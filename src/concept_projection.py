"""
concept_projection.py

Prompt-based concept directions and optional 1D calibration for interpretable
CLIP projections.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from .chexpert_utils import CHEXPERT_PATHOLOGIES

PROMPT_PHRASES = {
    "No Finding": {
        "positive": "no acute cardiopulmonary abnormality",
        "negative": "abnormal chest x-ray with at least one cardiopulmonary finding",
    },
    "Enlarged Cardiomediastinum": {
        "positive": "enlarged cardiomediastinal silhouette",
        "negative": "normal cardiomediastinal silhouette",
    },
    "Cardiomegaly": {
        "positive": "cardiomegaly",
        "negative": "no cardiomegaly",
    },
    "Lung Opacity": {
        "positive": "lung opacity",
        "negative": "clear lungs without focal opacity",
    },
    "Lung Lesion": {
        "positive": "lung lesion or pulmonary nodule",
        "negative": "no focal lung lesion",
    },
    "Edema": {
        "positive": "pulmonary edema",
        "negative": "no pulmonary edema",
    },
    "Consolidation": {
        "positive": "focal consolidation",
        "negative": "no focal consolidation",
    },
    "Pneumonia": {
        "positive": "pneumonia",
        "negative": "no evidence of pneumonia",
    },
    "Atelectasis": {
        "positive": "atelectasis",
        "negative": "no atelectasis",
    },
    "Pneumothorax": {
        "positive": "pneumothorax",
        "negative": "no pneumothorax",
    },
    "Pleural Effusion": {
        "positive": "pleural effusion",
        "negative": "no pleural effusion",
    },
    "Pleural Other": {
        "positive": "pleural abnormality",
        "negative": "no pleural abnormality",
    },
    "Fracture": {
        "positive": "fracture",
        "negative": "no acute fracture",
    },
    "Support Devices": {
        "positive": "support devices such as tubes, lines, or pacer leads",
        "negative": "no support devices",
    },
}

POSITIVE_TEMPLATES = [
    "chest x-ray with {phrase}",
    "radiology image demonstrating {phrase}",
    "frontal chest radiograph showing {phrase}",
]

NEGATIVE_TEMPLATES = [
    "chest x-ray with {phrase}",
    "radiology image demonstrating {phrase}",
    "frontal chest radiograph showing {phrase}",
]


def build_prompt_spec(pathology_cols: list[str] | None = None) -> dict[str, dict[str, list[str]]]:
    labels = pathology_cols or CHEXPERT_PATHOLOGIES
    prompt_spec = {}
    for label in labels:
        phrases = PROMPT_PHRASES[label]
        prompt_spec[label] = {
            "positive": [template.format(phrase=phrases["positive"]) for template in POSITIVE_TEMPLATES],
            "negative": [template.format(phrase=phrases["negative"]) for template in NEGATIVE_TEMPLATES],
        }
    return prompt_spec


def build_concept_matrix(
    prompt_spec: dict[str, dict[str, list[str]]],
    model,
    tokenizer: Callable[[list[str]], torch.Tensor],
    device: str,
) -> tuple[np.ndarray, list[str]]:
    pathology_cols = list(prompt_spec.keys())
    directions = []
    for label in pathology_cols:
        prompts = prompt_spec[label]
        positive_embeddings = _encode_prompts(prompts["positive"], model, tokenizer, device)
        negative_embeddings = _encode_prompts(prompts["negative"], model, tokenizer, device)
        directions.append(compute_direction_from_embeddings(positive_embeddings, negative_embeddings))

    matrix = np.vstack(directions).astype(np.float32)
    return matrix, pathology_cols


def compute_direction_from_embeddings(
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
) -> np.ndarray:
    pos_mean = _normalize_vector(positive_embeddings.mean(axis=0))
    neg_mean = _normalize_vector(negative_embeddings.mean(axis=0))
    return _normalize_vector(pos_mean - neg_mean).astype(np.float32)


def score_embeddings(embeddings: np.ndarray, concept_matrix: np.ndarray) -> np.ndarray:
    return embeddings @ concept_matrix.T


def fit_1d_calibrators(
    raw_scores: np.ndarray,
    label_frame: pd.DataFrame,
    pathology_cols: list[str],
) -> dict[str, dict[str, object]]:
    calibrators: dict[str, dict[str, object]] = {}
    for index, label in enumerate(pathology_cols):
        labels = pd.to_numeric(label_frame[label], errors="coerce")
        valid_mask = labels.isin([0.0, 1.0]).to_numpy()

        if valid_mask.sum() == 0:
            calibrators[label] = {"kind": "constant", "probability": 0.0}
            continue

        features = raw_scores[valid_mask, index : index + 1]
        targets = labels.loc[valid_mask].astype(int).to_numpy()

        if np.unique(targets).size < 2:
            calibrators[label] = {
                "kind": "constant",
                "probability": float(targets[0]) if len(targets) else 0.0,
            }
            continue

        model = LogisticRegression(max_iter=1000)
        model.fit(features, targets)
        calibrators[label] = {"kind": "logistic", "model": model}

    return calibrators


def predict_calibrated_probabilities(
    raw_scores: np.ndarray,
    pathology_cols: list[str],
    calibrators: dict[str, dict[str, object]],
) -> np.ndarray:
    probabilities = np.zeros((raw_scores.shape[0], len(pathology_cols)), dtype=np.float32)
    for index, label in enumerate(pathology_cols):
        calibrator = calibrators[label]
        if calibrator["kind"] == "constant":
            probabilities[:, index] = float(calibrator["probability"])
            continue

        model = calibrator["model"]
        probabilities[:, index] = model.predict_proba(raw_scores[:, index : index + 1])[:, 1]
    return probabilities


def build_blind_pair_concept_df(
    blind_pairs_df: pd.DataFrame,
    test_probabilities: np.ndarray,
    train_probabilities: np.ndarray,
    pathology_cols: list[str],
    top_k: int = 5,
    test_raw_scores: np.ndarray | None = None,
    train_raw_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    for pair in blind_pairs_df.itertuples(index=False):
        q_idx = int(pair.query_index)
        n_idx = int(pair.neighbor_index)

        if test_raw_scores is None or train_raw_scores is None:
            raise ValueError("Raw concept scores are required for pair-level concept explanations.")

        query_scores = test_raw_scores[q_idx]
        neighbor_scores = train_raw_scores[n_idx]
        score_deltas = query_scores - neighbor_scores

        query_probs = test_probabilities[q_idx]
        neighbor_probs = train_probabilities[n_idx]
        probability_deltas = query_probs - neighbor_probs

        ranked = np.argsort(np.abs(score_deltas))[::-1][:top_k]
        top_deltas = [
            {
                "concept": pathology_cols[idx],
                "score_delta": round(float(score_deltas[idx]), 6),
                "query_score": round(float(query_scores[idx]), 6),
                "neighbor_score": round(float(neighbor_scores[idx]), 6),
                "probability_delta": round(float(probability_deltas[idx]), 6),
                "query_probability": round(float(query_probs[idx]), 6),
                "neighbor_probability": round(float(neighbor_probs[idx]), 6),
            }
            for idx in ranked
        ]

        row = {
            "pair_id": pair.pair_id,
            "query_uid": pair.query_uid,
            "neighbor_uid": pair.neighbor_uid,
            "query_index": q_idx,
            "neighbor_index": n_idx,
            "query_filename": pair.query_filename,
            "neighbor_filename": pair.neighbor_filename,
            "query_concept_scores": json.dumps(_vector_to_dict(pathology_cols, query_scores)),
            "neighbor_concept_scores": json.dumps(_vector_to_dict(pathology_cols, neighbor_scores)),
            "concept_score_deltas": json.dumps(_vector_to_dict(pathology_cols, score_deltas)),
            "query_concept_probabilities": json.dumps(_vector_to_dict(pathology_cols, query_probs)),
            "neighbor_concept_probabilities": json.dumps(_vector_to_dict(pathology_cols, neighbor_probs)),
            "concept_probability_deltas": json.dumps(_vector_to_dict(pathology_cols, probability_deltas)),
            "top_score_deltas": json.dumps(top_deltas),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_blind_pair_concepts(
    blind_pair_concepts_df: pd.DataFrame,
    blind_pairs_df: pd.DataFrame,
    pathology_cols: list[str],
) -> pd.DataFrame:
    """
    Summarize raw concept-score deltas by blind type for downstream plotting.
    """
    pair_types = blind_pairs_df[["pair_id", "blind_type"]].drop_duplicates()
    merged = blind_pair_concepts_df[["pair_id", "concept_score_deltas", "top_score_deltas"]].merge(
        pair_types,
        on="pair_id",
        how="left",
    )
    merged = merged.dropna(subset=["blind_type"]).reset_index(drop=True)

    summary_rows = []
    for blind_type, group in merged.groupby("blind_type"):
        pair_count = len(group)
        if pair_count == 0:
            continue

        delta_matrix = np.zeros((pair_count, len(pathology_cols)), dtype=np.float32)
        top1_counts = {label: 0 for label in pathology_cols}
        top5_counts = {label: 0 for label in pathology_cols}

        for row_index, row in enumerate(group.itertuples(index=False)):
            deltas = json.loads(row.concept_score_deltas)
            delta_matrix[row_index] = [float(deltas.get(label, 0.0)) for label in pathology_cols]

            top_deltas = json.loads(row.top_score_deltas)
            if top_deltas:
                top1_counts[top_deltas[0]["concept"]] += 1
                for item in top_deltas:
                    top5_counts[item["concept"]] += 1

        mean_abs = np.abs(delta_matrix).mean(axis=0)
        mean_signed = delta_matrix.mean(axis=0)

        for concept_index, concept in enumerate(pathology_cols):
            summary_rows.append(
                {
                    "blind_type": blind_type,
                    "concept": concept,
                    "mean_abs_delta": round(float(mean_abs[concept_index]), 6),
                    "mean_signed_delta": round(float(mean_signed[concept_index]), 6),
                    "delta_kind": "raw_score",
                    "top1_frequency_pct": round(float(top1_counts[concept] / pair_count * 100.0), 3),
                    "top5_frequency_pct": round(float(top5_counts[concept] / pair_count * 100.0), 3),
                    "pair_count": pair_count,
                }
            )

    return pd.DataFrame(summary_rows)


def _encode_prompts(prompts: list[str], model, tokenizer, device: str) -> np.ndarray:
    tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        features = model.encode_text(tokens)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features.detach().cpu().numpy()


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _vector_to_dict(pathology_cols: list[str], values: np.ndarray) -> dict[str, float]:
    return {label: round(float(value), 6) for label, value in zip(pathology_cols, values)}
