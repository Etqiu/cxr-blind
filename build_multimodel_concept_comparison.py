"""
Compare blind-pair concept explanations between CLIP and RAD-DINO spaces.

To reduce CLIP-text circularity, this script builds concept directions
independently inside each image embedding space using simple CheXpert-labeled
image prototypes:

    direction_k = normalize(mean(pos_k) - mean(neg_k))

The same blind pairs are then scored in both spaces and compared side by side.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent / "modal_downloads" / "indiana_artifacts"
IMAGE_BASE = BASE / "representative_images"
CHEXPERT_PATH = Path(__file__).resolve().parent.parent / "cxr-blind-retrieval" / "data" / "chexpert_labels.csv"
BLIND_TYPES = ["Type 1", "Type 2", "Type 3"]
MODELS = {
    "clip": "image_embeddings.pkl",
    "dino": "raddino_embeddings.pkl",
}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blind_pairs = pd.read_csv(BASE / "blind_pairs_analysis.csv")
    chexpert = pd.read_csv(CHEXPERT_PATH)
    chexpert["filename"] = chexpert["image_path"].astype(str).map(lambda value: os.path.basename(value.replace("\\", "/")))
    return blind_pairs, chexpert, pd.read_csv(BASE / "representative_blind_pairs_standardized.csv")


def load_embedding_dict(name: str) -> dict[str, np.ndarray]:
    with (BASE / MODELS[name]).open("rb") as file_obj:
        emb_dict = pickle.load(file_obj)
    return {key: np.asarray(value, dtype=np.float32) for key, value in emb_dict.items()}


def build_embedding_frame(emb_dict: dict[str, np.ndarray], chexpert: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pathology_cols = [column for column in chexpert.columns if column not in {"uid", "image_path", "filename"}]
    df = chexpert[["filename", *pathology_cols]].copy()
    df["embedding"] = df["filename"].map(emb_dict)
    df = df.dropna(subset=["embedding"]).reset_index(drop=True)
    return df, pathology_cols


def build_prototype_concept_matrix(df: pd.DataFrame, pathology_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.stack(df["embedding"].to_list()).astype(np.float32)
    embeddings = _normalize_rows(embeddings)

    directions = []
    for label in pathology_cols:
        values = pd.to_numeric(df[label], errors="coerce")
        positive_mask = (values == 1.0).to_numpy()
        negative_mask = (values == 0.0).to_numpy()

        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            directions.append(np.zeros(embeddings.shape[1], dtype=np.float32))
            continue

        pos_mean = _normalize_vector(embeddings[positive_mask].mean(axis=0))
        neg_mean = _normalize_vector(embeddings[negative_mask].mean(axis=0))
        directions.append(_normalize_vector(pos_mean - neg_mean))

    concept_matrix = np.vstack(directions).astype(np.float32)
    scores = embeddings @ concept_matrix.T
    return concept_matrix, scores


def build_score_lookup(filenames: list[str], scores: np.ndarray) -> dict[str, np.ndarray]:
    return {filename: score.astype(np.float32) for filename, score in zip(filenames, scores)}


def pair_level_comparison(
    blind_pairs: pd.DataFrame,
    pathology_cols: list[str],
    clip_scores: dict[str, np.ndarray],
    dino_scores: dict[str, np.ndarray],
    clip_stds: np.ndarray,
    dino_stds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for pair in blind_pairs.itertuples(index=False):
        q_file = pair.query_filename
        n_file = pair.neighbor_filename
        if q_file not in clip_scores or n_file not in clip_scores or q_file not in dino_scores or n_file not in dino_scores:
            continue

        clip_delta = clip_scores[q_file] - clip_scores[n_file]
        dino_delta = dino_scores[q_file] - dino_scores[n_file]

        clip_scale = clip_delta / np.where(clip_stds == 0.0, 1.0, clip_stds)
        dino_scale = dino_delta / np.where(dino_stds == 0.0, 1.0, dino_stds)

        clip_rank = np.argsort(np.abs(clip_scale))[::-1][:5]
        dino_rank = np.argsort(np.abs(dino_scale))[::-1][:5]

        clip_top = [
            {
                "concept": pathology_cols[index],
                "standardized_delta": round(float(clip_scale[index]), 6),
                "raw_score_delta": round(float(clip_delta[index]), 6),
            }
            for index in clip_rank
        ]
        dino_top = [
            {
                "concept": pathology_cols[index],
                "standardized_delta": round(float(dino_scale[index]), 6),
                "raw_score_delta": round(float(dino_delta[index]), 6),
            }
            for index in dino_rank
        ]

        clip_top1 = clip_top[0]["concept"] if clip_top else None
        dino_top1 = dino_top[0]["concept"] if dino_top else None

        rows.append(
            {
                "pair_id": pair.pair_id,
                "blind_type": pair.blind_type,
                "query_filename": q_file,
                "neighbor_filename": n_file,
                "clip_raw_deltas": json.dumps(_vector_to_dict(pathology_cols, clip_delta)),
                "clip_standardized_deltas": json.dumps(_vector_to_dict(pathology_cols, clip_scale)),
                "dino_raw_deltas": json.dumps(_vector_to_dict(pathology_cols, dino_delta)),
                "dino_standardized_deltas": json.dumps(_vector_to_dict(pathology_cols, dino_scale)),
                "clip_top_standardized_deltas": json.dumps(clip_top),
                "dino_top_standardized_deltas": json.dumps(dino_top),
                "clip_top1_concept": clip_top1,
                "dino_top1_concept": dino_top1,
                "top1_agree": bool(clip_top1 == dino_top1),
            }
        )

    return pd.DataFrame(rows)


def summary_tables(pair_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for blind_type, group in pair_df.groupby("blind_type"):
        pair_count = len(group)
        if pair_count == 0:
            continue
        summary_rows.append(
            {
                "blind_type": blind_type,
                "pair_count": pair_count,
                "top1_agreement_pct": round(float(group["top1_agree"].mean() * 100.0), 3),
                "unique_clip_top1": int(group["clip_top1_concept"].nunique()),
                "unique_dino_top1": int(group["dino_top1_concept"].nunique()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    concept_rows = []
    for blind_type, group in pair_df.groupby("blind_type"):
        pair_count = len(group)
        clip_counts = group["clip_top1_concept"].value_counts()
        dino_counts = group["dino_top1_concept"].value_counts()
        concepts = sorted(set(clip_counts.index) | set(dino_counts.index))
        for concept in concepts:
            concept_rows.append(
                {
                    "blind_type": blind_type,
                    "concept": concept,
                    "clip_top1_pct": round(float(clip_counts.get(concept, 0) / pair_count * 100.0), 3),
                    "dino_top1_pct": round(float(dino_counts.get(concept, 0) / pair_count * 100.0), 3),
                    "difference_pct": round(float((clip_counts.get(concept, 0) - dino_counts.get(concept, 0)) / pair_count * 100.0), 3),
                    "pair_count": pair_count,
                }
            )
    concept_df = pd.DataFrame(concept_rows)
    return summary_df, concept_df


def representative_table(pair_df: pd.DataFrame, representative_df: pd.DataFrame) -> pd.DataFrame:
    merged = representative_df[["pair_id", "blind_type", "query_filename", "neighbor_filename"]].merge(
        pair_df,
        on=["pair_id", "blind_type", "query_filename", "neighbor_filename"],
        how="left",
    )
    return merged


def make_agreement_figure(summary_df: pd.DataFrame, concept_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#f8fafc")

    ax1 = axes[0]
    summary_plot = summary_df.set_index("blind_type").reindex(BLIND_TYPES)
    bars = ax1.bar(summary_plot.index, summary_plot["top1_agreement_pct"], color=["#4f46e5", "#f59e0b", "#dc2626"], width=0.6)
    for bar, value in zip(bars, summary_plot["top1_agreement_pct"].fillna(0.0)):
        ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylim(0, max(100, float(summary_plot["top1_agreement_pct"].max()) + 10))
    ax1.set_title("CLIP vs DINO Top-1 Concept Agreement", fontweight="bold")
    ax1.set_ylabel("Agreement on same top concept (%)")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    overall = concept_df.groupby("concept")[["clip_top1_pct", "dino_top1_pct"]].mean().sort_values("clip_top1_pct", ascending=True)
    ax2.barh(overall.index, overall["clip_top1_pct"], color="#2563eb", alpha=0.75, label="CLIP")
    ax2.barh(overall.index, overall["dino_top1_pct"], color="#ea580c", alpha=0.55, label="DINO")
    ax2.set_title("Average Top-1 Concept Frequency by Model", fontweight="bold")
    ax2.set_xlabel("Mean top-1 frequency across blind types (%)")
    ax2.legend(frameon=False, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Blind-Pair Concept Comparison Across Embedding Spaces", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(BASE / "multimodel_concept_agreement.png", dpi=180)
    plt.close(fig)


def make_representative_comparison_figure(rep_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(rep_df),
        6,
        figsize=(24, 5.2 * len(rep_df)),
        facecolor="#f8fafc",
        gridspec_kw={"width_ratios": [0.85, 0.85, 1.45, 1.45, 1.45, 1.45]},
    )
    if len(rep_df) == 1:
        axes = np.asarray([axes])

    for row_index, row in enumerate(rep_df.itertuples(index=False)):
        clip_raw = json.loads(row.clip_raw_deltas)
        clip_std = json.loads(row.clip_standardized_deltas)
        dino_raw = json.loads(row.dino_raw_deltas)
        dino_std = json.loads(row.dino_standardized_deltas)

        clip_top = json.loads(row.clip_top_standardized_deltas)[:3]
        dino_top = json.loads(row.dino_top_standardized_deltas)[:3]
        concept_order = []
        for item in clip_top + dino_top:
            concept = item["concept"]
            if concept not in concept_order:
                concept_order.append(concept)
        concept_order = sorted(
            concept_order,
            key=lambda concept: max(abs(float(clip_std.get(concept, 0.0))), abs(float(dino_std.get(concept, 0.0)))),
            reverse=True,
        )[:6]
        concept_order = concept_order[::-1]

        y = np.arange(len(concept_order))
        height = 0.36

        _plot_image_panel(
            axes[row_index, 0],
            IMAGE_BASE / row.query_filename,
            f"{row.blind_type}: query\n{row.query_filename}",
        )
        _plot_image_panel(
            axes[row_index, 1],
            IMAGE_BASE / row.neighbor_filename,
            f"{row.blind_type}: neighbor\n{row.neighbor_filename}",
        )

        _plot_dual_barh(
            axes[row_index, 2],
            y,
            concept_order,
            [0.0] * len(concept_order),
            [float(clip_raw[c]) for c in concept_order],
            "CLIP prototype raw delta",
            xlabel="Query - neighbor raw score",
            left_label="0",
            right_label="CLIP",
        )

        _plot_dual_barh(
            axes[row_index, 3],
            y,
            concept_order,
            [0.0] * len(concept_order),
            [float(clip_std[c]) for c in concept_order],
            "CLIP prototype standardized delta",
            xlabel="(Query - neighbor) / sigma",
            left_label="0",
            right_label="CLIP",
        )

        _plot_dual_barh(
            axes[row_index, 4],
            y,
            concept_order,
            [0.0] * len(concept_order),
            [float(dino_raw[c]) for c in concept_order],
            "DINO prototype raw delta",
            xlabel="Query - neighbor raw score",
            left_label="0",
            right_label="DINO",
        )

        top_concept = json.loads(row.dino_top_standardized_deltas)[0]["concept"] if json.loads(row.dino_top_standardized_deltas) else "NA"
        top_std = json.loads(row.dino_top_standardized_deltas)[0]["standardized_delta"] if json.loads(row.dino_top_standardized_deltas) else 0.0
        _plot_dual_barh(
            axes[row_index, 5],
            y,
            concept_order,
            [0.0] * len(concept_order),
            [float(dino_std[c]) for c in concept_order],
            f"DINO prototype standardized delta\nTop DINO driver: {top_concept} ({top_std:+.2f} SD)",
            xlabel="(Query - neighbor) / sigma",
            left_label="0",
            right_label="DINO",
        )

    fig.suptitle(
        "Representative Blind Pairs: Prototype Concept Comparison in CLIP and DINO Spaces",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "Concept directions are built from positive-vs-negative CheXpert image prototypes separately in each embedding space.",
        ha="center",
        fontsize=11,
        color="#374151",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
    fig.savefig(BASE / "representative_blind_pairs_clip_vs_dino_prototypes.png", dpi=180)
    plt.close(fig)


def _plot_image_panel(ax, image_path: Path, title: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if image_path.exists():
        image = mpimg.imread(image_path)
        ax.imshow(image, cmap="gray")
    else:
        ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center", fontsize=11, color="#6b7280")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_dual_barh(
    ax,
    y: np.ndarray,
    concepts: list[str],
    left_values: list[float],
    right_values: list[float],
    title: str,
    xlabel: str,
    left_label: str,
    right_label: str,
) -> None:
    height = 0.34
    ax.barh(y - height / 2, left_values, height, color="#94a3b8", label=left_label)
    ax.barh(y + height / 2, right_values, height, color="#2563eb" if right_label == "CLIP" else "#ea580c", label=right_label)
    ax.axvline(0.0, color="#6b7280", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(concepts)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)


def _vector_to_dict(pathology_cols: list[str], values: np.ndarray) -> dict[str, float]:
    return {label: round(float(value), 6) for label, value in zip(pathology_cols, values)}


def main() -> None:
    blind_pairs, chexpert, representative_df = load_inputs()
    clip_emb = load_embedding_dict("clip")
    dino_emb = load_embedding_dict("dino")

    clip_df, pathology_cols = build_embedding_frame(clip_emb, chexpert)
    dino_df, dino_cols = build_embedding_frame(dino_emb, chexpert)
    if pathology_cols != dino_cols:
        raise ValueError("CLIP and DINO pathology columns do not align.")

    _, clip_scores = build_prototype_concept_matrix(clip_df, pathology_cols)
    _, dino_scores = build_prototype_concept_matrix(dino_df, pathology_cols)

    clip_stds = np.where(clip_scores.std(axis=0) < 1e-8, 1.0, clip_scores.std(axis=0))
    dino_stds = np.where(dino_scores.std(axis=0) < 1e-8, 1.0, dino_scores.std(axis=0))

    clip_lookup = build_score_lookup(clip_df["filename"].tolist(), clip_scores)
    dino_lookup = build_score_lookup(dino_df["filename"].tolist(), dino_scores)

    pair_df = pair_level_comparison(blind_pairs, pathology_cols, clip_lookup, dino_lookup, clip_stds, dino_stds)
    summary_df, concept_df = summary_tables(pair_df)
    rep_df = representative_table(pair_df, representative_df)

    pair_df.to_csv(BASE / "multimodel_blind_pair_concepts.csv", index=False)
    summary_df.to_csv(BASE / "multimodel_concept_agreement_summary.csv", index=False)
    concept_df.to_csv(BASE / "multimodel_concept_top1_by_type.csv", index=False)
    rep_df.to_csv(BASE / "multimodel_representative_pairs.csv", index=False)
    make_agreement_figure(summary_df, concept_df)
    make_representative_comparison_figure(rep_df)

    print("Wrote multimodel concept comparison artifacts to", BASE)
    print("\nAgreement summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
