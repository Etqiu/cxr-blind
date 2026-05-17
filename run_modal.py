"""
run_modal.py

Main orchestrator for running the combined CXR blind retrieval pipeline on Modal.
Supports MIMIC preparation and prompt-based concept projections for blind-pair
interpretation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = "cxr-combined-pipeline"
MIMIC_VOLUME_NAME = "radiology-data"
INDIANA_VOLUME_NAME = "radiology-archive"
DEFAULT_GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "radiology-report-488501")
MIMIC_REPORTS_BUCKET = "mimic-cxr-2.0.0.physionet.org"
MIMIC_JPG_BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
RAW_MIMIC_ROOT = Path("/mnt/radiology-data/_mimic_source")
RAW_MIMIC_REPORTS_ROOT = RAW_MIMIC_ROOT / "reports" / "files"
RAW_MIMIC_TABLES_ROOT = RAW_MIMIC_ROOT / "tables"
INDIANA_MOUNT_ROOT = Path("/mnt/radiology-archive")

app = modal.App(APP_NAME)
local_src_dir = os.path.join(os.path.dirname(__file__), "src")


def _maybe_load_google_creds() -> str | None:
    cred_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if not cred_path.exists():
        return ""
    return cred_path.read_text()


LOCAL_GOOGLE_CREDS = _maybe_load_google_creds()
GOOGLE_CREDS_SECRET = modal.Secret.from_dict({"GOOGLE_CREDS": LOCAL_GOOGLE_CREDS})

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers<4.40.0",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "Pillow",
        "requests",
        "open_clip_torch",
        "faiss-cpu",
        "radgraph",
        "google-cloud-storage",
    )
    .add_local_dir(local_src_dir, remote_path="/root/src")
)

try:
    mimic_volume = modal.Volume.from_name(MIMIC_VOLUME_NAME, create_if_missing=True)
except Exception:
    mimic_volume = modal.Volume.from_name(MIMIC_VOLUME_NAME)

try:
    indiana_volume = modal.Volume.from_name(INDIANA_VOLUME_NAME, create_if_missing=True)
except Exception:
    indiana_volume = modal.Volume.from_name(INDIANA_VOLUME_NAME)

def _configure_google_credentials() -> None:
    creds_str = os.environ.get("GOOGLE_CREDS", "")
    if not creds_str:
        raise RuntimeError(
            "Missing GOOGLE_CREDS secret. Run `gcloud auth application-default login` locally "
            "before dispatching MIMIC preparation."
        )

    cred_path = Path("/tmp/gcp_creds.json")
    cred_path.write_text(creds_str)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)


def _resolve_indiana_base_path() -> str:
    direct = INDIANA_MOUNT_ROOT / "archive"
    nested = direct / "archive"
    if (direct / "indiana_reports.csv").exists() and (direct / "indiana_projections.csv").exists():
        return str(direct)
    if (nested / "indiana_reports.csv").exists() and (nested / "indiana_projections.csv").exists():
        return str(nested)
    return str(direct)


@app.function(
    image=image,
    volumes={"/mnt/radiology-data": mimic_volume},
    timeout=86400,
    secrets=[GOOGLE_CREDS_SECRET],
)
def prepare_mimic_dataset(project_id: str = DEFAULT_GCP_PROJECT, mode: str = "auto"):
    import pandas as pd
    from google.cloud import storage
    from src.mimic_prep import (
        build_mimic_projections_dataframe,
        discover_local_mimic_images,
        normalize_mimic_chexpert_dataframe,
        parse_report_sections,
    )

    if mode not in {"auto", "always", "never"}:
        raise ValueError(f"Unsupported prepare mode: {mode}")

    base_path = Path("/mnt/radiology-data")
    local_images_df = discover_local_mimic_images(base_path)
    if local_images_df.empty:
        raise FileNotFoundError(
            f"No MIMIC JPGs were found under {base_path}. Expected a downloaded JPG subset first."
        )

    local_study_ids = sorted(local_images_df["study_id"].unique().tolist())
    if mode == "never":
        return {
            "status": "skipped",
            "reason": "prepare=never",
            "study_count": len(local_study_ids),
            "image_count": len(local_images_df),
        }

    if mode == "auto" and _mimic_outputs_are_current(base_path, local_study_ids):
        return {
            "status": "skipped",
            "reason": "prepared_outputs_already_match_local_subset",
            "study_count": len(local_study_ids),
            "image_count": len(local_images_df),
        }

    _configure_google_credentials()
    client = storage.Client(project=project_id)

    RAW_MIMIC_TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_path = RAW_MIMIC_TABLES_ROOT / "mimic-cxr-2.0.0-metadata.csv.gz"
    chexpert_path = RAW_MIMIC_TABLES_ROOT / "mimic-cxr-2.0.0-chexpert.csv.gz"

    jpg_bucket = client.bucket(MIMIC_JPG_BUCKET, user_project=project_id)
    _download_blob_if_missing(jpg_bucket, metadata_path.name, metadata_path)
    _download_blob_if_missing(jpg_bucket, chexpert_path.name, chexpert_path)

    reports_bucket = client.bucket(MIMIC_REPORTS_BUCKET, user_project=project_id)
    study_rows = (
        local_images_df[["patient_group", "subject_id", "study_dir", "study_id"]]
        .drop_duplicates()
        .sort_values(["study_id", "subject_id"])
    )

    downloaded_reports = 0
    report_rows = []
    missing_reports = []
    for row in study_rows.itertuples(index=False):
        blob_name = f"files/{row.patient_group}/p{row.subject_id}/{row.study_dir}.txt"
        blob = reports_bucket.blob(blob_name)
        try:
            report_text = blob.download_as_text()
        except Exception:
            missing_reports.append(row.study_id)
            continue

        findings, impression, full_text = parse_report_sections(report_text)
        report_rows.append(
            {
                "uid": row.study_id,
                "findings": findings,
                "impression": impression,
                "full_text": full_text,
            }
        )
        downloaded_reports += 1

    reports_df = pd.DataFrame(report_rows, columns=["uid", "findings", "impression", "full_text"])
    reports_df = reports_df.sort_values("uid").reset_index(drop=True) if not reports_df.empty else reports_df
    if reports_df.empty:
        raise FileNotFoundError(
            "Could not retrieve any MIMIC reports for the locally mounted image subset. "
            "This usually means the report bucket access or path assumptions are wrong."
        )

    if missing_reports:
        print(
            f"Warning: skipped {len(missing_reports)} studies whose reports were unavailable "
            f"for this subset. Example study IDs: {missing_reports[:5]}"
        )

    available_study_ids = set(reports_df["uid"].astype(str))
    local_images_df = local_images_df[local_images_df["study_id"].astype(str).isin(available_study_ids)].copy()
    local_images_df = local_images_df.sort_values(["study_id", "filename"]).reset_index(drop=True)

    metadata_df = pd.read_csv(metadata_path, compression="gzip")
    projections_df = build_mimic_projections_dataframe(local_images_df, metadata_df)

    official_chexpert_df = pd.read_csv(chexpert_path, compression="gzip")
    mimic_chexpert_df = normalize_mimic_chexpert_dataframe(official_chexpert_df, available_study_ids)
    if mimic_chexpert_df.empty:
        raise FileNotFoundError(
            "No official MIMIC CheXpert rows overlapped with the report-backed image subset."
        )

    reports_out = base_path / "mimic_reports.csv"
    projections_out = base_path / "mimic_projections.csv"
    chexpert_out = base_path / "mimic_chexpert.csv"

    reports_df.to_csv(reports_out, index=False)
    projections_df.to_csv(projections_out, index=False)
    mimic_chexpert_df.to_csv(chexpert_out, index=False)
    mimic_volume.commit()

    return {
        "status": "prepared",
        "study_count": len(available_study_ids),
        "image_count": len(local_images_df),
        "report_rows": len(reports_df),
        "projection_rows": len(projections_df),
        "chexpert_rows": len(mimic_chexpert_df),
        "downloaded_reports": downloaded_reports,
        "missing_reports": len(missing_reports),
    }


@app.function(
    image=image,
    gpu="T4",
    volumes={
        "/mnt/radiology-data": mimic_volume,
        "/mnt/radiology-archive": indiana_volume,
    },
    timeout=86400,
)
def run_pipeline_on_modal(
    dataset: str = "indiana",
    prepare_mode: str = "auto",
    interpretability: str = "concepts",
):
    import gc
    import pickle

    import numpy as np
    import open_clip
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    if dataset not in {"indiana", "mimic"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if prepare_mode not in {"auto", "always", "never"}:
        raise ValueError(f"Unsupported prepare mode: {prepare_mode}")
    if interpretability not in {"concepts", "off"}:
        raise ValueError(f"Unsupported interpretability mode: {interpretability}")

    os.environ["DATASET_TYPE"] = "mimic" if dataset == "mimic" else "indiana"
    os.environ["RADIOLOGY_BASE_PATH"] = (
        "/mnt/radiology-data" if dataset == "mimic" else _resolve_indiana_base_path()
    )
    os.environ["EXPECT_OFFICIAL_MIMIC_CHEXPERT"] = (
        "1" if dataset == "mimic" and prepare_mode != "never" else "0"
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from src import config
    from src.analysis import build_consensus, connect_blindtype_to_deviation
    from src.chexpert_utils import align_chexpert_labels, load_chexpert_table
    from src.concept_projection import (
        build_blind_pair_concept_df,
        build_concept_matrix,
        build_prompt_spec,
        fit_1d_calibrators,
        predict_calibrated_probabilities,
        score_embeddings,
        summarize_blind_pair_concepts,
    )
    from src.data_loader import load_data
    from src.embeddings import (
        compute_hf_vision_embeddings,
        compute_openclip_image_embeddings,
        compute_text_embeddings,
    )
    from src.radgraph_utils import extract_entities, load_radgraph
    from src.retrieval import build_faiss_index, label_consistent_blind, mask_to_indices, retrieve_top_k

    print("====== MODAL GPU APP STARTING ======")
    print(f"Dataset: {dataset}")
    print(f"Prepare mode: {prepare_mode}")
    print(f"Interpretability: {interpretability}")
    print(f"Mounted Base Path: {config.BASE_PATH}")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    clip_model = None
    clip_preprocess = None
    clip_tokenizer = None

    def ensure_clip_components():
        nonlocal clip_model, clip_preprocess, clip_tokenizer
        if clip_model is None:
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(config.BIOMEDCLIP_MODEL)
            clip_model.to(config.DEVICE).eval()
            clip_tokenizer = open_clip.get_tokenizer(config.BIOMEDCLIP_MODEL)
        return clip_model, clip_preprocess, clip_tokenizer

    print("\n--- 1. Loading dataset ---")
    df_clean = load_data(config.REPORTS_FILE, config.PROJECTIONS_FILE)

    print("\n--- 2. BioMedCLIP Image Embeddings ---")
    clip_cache = config.ARTIFACTS_DIR / "image_embeddings.pkl"
    if clip_cache.exists():
        print(f"Loading {clip_cache}")
        with clip_cache.open("rb") as file_obj:
            clip_emb_dict = pickle.load(file_obj)
    else:
        model, preprocess, _ = ensure_clip_components()
        clip_emb_dict = compute_openclip_image_embeddings(
            df_clean["filename"].dropna().unique(),
            config.IMAGES_DIR,
            preprocess,
            model,
            config.DEVICE,
            config.OPENCLIP_BATCH_SIZE,
            config.NUM_IMAGE_WORKERS,
        )
        with clip_cache.open("wb") as file_obj:
            pickle.dump(clip_emb_dict, file_obj)

    df_clean["image_embedding"] = df_clean["filename"].map(clip_emb_dict)
    df_clean = df_clean.dropna(subset=["image_embedding"]).copy()
    if clip_model is not None:
        del clip_model
        clip_model = None
    clip_preprocess = None
    clip_tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n--- 3. RAD-DINO Image Embeddings ---")
    dino_cache = config.ARTIFACTS_DIR / "raddino_embeddings.pkl"
    if dino_cache.exists():
        print(f"Loading {dino_cache}")
        with dino_cache.open("rb") as file_obj:
            dino_emb_dict = pickle.load(file_obj)
    else:
        processor = AutoImageProcessor.from_pretrained(config.RADDINO_MODEL)
        dino_model = AutoModel.from_pretrained(config.RADDINO_MODEL).to(config.DEVICE).eval()
        dino_emb_dict = compute_hf_vision_embeddings(
            df_clean["filename"].dropna().unique(),
            config.IMAGES_DIR,
            processor,
            dino_model,
            config.DEVICE,
            config.RAD_DINO_BATCH_SIZE,
            config.NUM_IMAGE_WORKERS,
            dino_cache,
            10,
        )
        del dino_model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df_clean["raddino_embedding"] = df_clean["filename"].map(dino_emb_dict)
    df_clean = df_clean.dropna(subset=["raddino_embedding"]).copy()

    print("\n--- 4. BioMed-RoBERTa Text Embeddings ---")
    text_cache = config.ARTIFACTS_DIR / "text_embeddings.pkl"
    if text_cache.exists():
        print(f"Loading {text_cache}")
        with text_cache.open("rb") as file_obj:
            text_emb = pickle.load(file_obj)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.BIOMED_ROBERTA_MODEL)
        text_model = AutoModel.from_pretrained(config.BIOMED_ROBERTA_MODEL).to(config.DEVICE).eval()
        text_emb = compute_text_embeddings(
            df_clean["full_text"].tolist(),
            text_model,
            tokenizer,
            config.DEVICE,
            config.TEXT_BATCH_SIZE,
        )
        with text_cache.open("wb") as file_obj:
            pickle.dump(text_emb, file_obj)
        del text_model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    df_clean["text_embedding"] = list(text_emb)

    train = df_clean.copy()
    test = df_clean.copy()
    print(f"\n--- 5. Using Full Dataset for Replication: {len(train)} Images ---")

    print("\n--- 6. FAISS Retrieval & Labeling ---")
    train_img_embs = np.stack(train["image_embedding"]).astype("float32")
    test_img_embs = np.stack(test["image_embedding"]).astype("float32")
    train_dino_embs = np.stack(train["raddino_embedding"]).astype("float32")
    test_dino_embs = np.stack(test["raddino_embedding"]).astype("float32")

    img_index = build_faiss_index(train_img_embs)
    d_clip_full, i_clip_full = retrieve_top_k(img_index, test_img_embs, config.RETRIEVAL_K + 1)
    d_clip = d_clip_full[:, 1:]
    i_clip = i_clip_full[:, 1:]

    consistent_mask, blind_mask = label_consistent_blind(
        d_clip,
        i_clip,
        test_dino_embs,
        train_dino_embs,
        config.CLIP_THRESHOLD,
        config.DINO_THRESHOLD,
    )
    test_cons_neighbors = mask_to_indices(i_clip, consistent_mask)
    test_blind_neighbors = mask_to_indices(i_clip, blind_mask)

    concept_scores = None
    concept_probabilities = None
    concept_labels = None
    if interpretability == "concepts":
        print("\n--- 7. Concept Projection & Calibration ---")
        chexpert_table, concept_labels = load_chexpert_table(df_clean)
        aligned_chexpert = align_chexpert_labels(df_clean, chexpert_table, concept_labels)

        prompt_spec = build_prompt_spec(concept_labels)
        if config.CONCEPT_MATRIX_CACHE.exists():
            with config.CONCEPT_MATRIX_CACHE.open("rb") as file_obj:
                matrix_cache = pickle.load(file_obj)
            if matrix_cache.get("pathology_cols") == concept_labels:
                concept_matrix = np.asarray(matrix_cache["matrix"], dtype=np.float32)
            else:
                matrix_cache = None
                concept_matrix = None
        else:
            matrix_cache = None
            concept_matrix = None

        if concept_matrix is None:
            model, _, tokenizer = ensure_clip_components()
            concept_matrix, concept_labels = build_concept_matrix(prompt_spec, model, tokenizer, config.DEVICE)
            with config.CONCEPT_MATRIX_CACHE.open("wb") as file_obj:
                pickle.dump(
                    {
                        "pathology_cols": concept_labels,
                        "matrix": concept_matrix,
                        "prompt_spec": prompt_spec,
                    },
                    file_obj,
                )
            config.CONCEPT_PROMPTS_CACHE.write_text(json.dumps(prompt_spec, indent=2))
        elif not config.CONCEPT_PROMPTS_CACHE.exists():
            config.CONCEPT_PROMPTS_CACHE.write_text(json.dumps(prompt_spec, indent=2))

        if clip_model is not None:
            del clip_model
            clip_model = None
        clip_preprocess = None
        clip_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        cached_scores = None
        if config.CONCEPT_SCORES_CACHE.exists():
            with config.CONCEPT_SCORES_CACHE.open("rb") as file_obj:
                cached_scores = pickle.load(file_obj)
            same_filenames = cached_scores.get("filenames") == df_clean["filename"].tolist()
            same_uids = cached_scores.get("uids") == df_clean["uid"].astype(str).tolist()
            same_labels = cached_scores.get("pathology_cols") == concept_labels
            if not (same_filenames and same_uids and same_labels):
                cached_scores = None

        if cached_scores is not None:
            concept_scores = np.asarray(cached_scores["raw_scores"], dtype=np.float32)
            concept_probabilities = np.asarray(cached_scores["probabilities"], dtype=np.float32)
            print(f"Loaded cached concept scores from {config.CONCEPT_SCORES_CACHE}")
        else:
            concept_scores = score_embeddings(train_img_embs, concept_matrix)
            calibrators = fit_1d_calibrators(concept_scores, aligned_chexpert, concept_labels)
            concept_probabilities = predict_calibrated_probabilities(
                concept_scores,
                concept_labels,
                calibrators,
            )
            with config.CONCEPT_CALIBRATORS_CACHE.open("wb") as file_obj:
                pickle.dump({"pathology_cols": concept_labels, "calibrators": calibrators}, file_obj)
            with config.CONCEPT_SCORES_CACHE.open("wb") as file_obj:
                pickle.dump(
                    {
                        "uids": df_clean["uid"].astype(str).tolist(),
                        "filenames": df_clean["filename"].tolist(),
                        "pathology_cols": concept_labels,
                        "raw_scores": concept_scores,
                        "probabilities": concept_probabilities,
                    },
                    file_obj,
                )

    print("\n--- 8. RadGraph Entity Extraction ---")
    consolidated_rg_cache = config.ARTIFACTS_DIR / "full_dataset_radgraph_entities.pkl"
    if consolidated_rg_cache.exists():
        with consolidated_rg_cache.open("rb") as file_obj:
            full_ents = pickle.load(file_obj)
        train_ents = full_ents
        test_ents = full_ents
    else:
        rg_model = load_radgraph(config.RADGRAPH_MODEL)
        full_ents = extract_entities(df_clean["full_text"].tolist(), rg_model, config.RADGRAPH_BATCH_SIZE)
        with consolidated_rg_cache.open("wb") as file_obj:
            pickle.dump(full_ents, file_obj)
        train_ents = full_ents
        test_ents = full_ents

    print("\n--- 9. Consensus & CheXpert Analysis ---")
    test = build_consensus(test, train, i_clip, test_cons_neighbors, test_blind_neighbors, test_ents, train_ents)
    blind_pairs_df = connect_blindtype_to_deviation(test, train, df_clean, test_ents, train_ents)

    test.to_csv(config.DEVIATION_RESULTS_ARTIFACT, index=False)
    blind_pairs_df.to_csv(config.BLIND_PAIRS_ARTIFACT, index=False)

    if interpretability == "concepts" and concept_scores is not None and concept_probabilities is not None:
        blind_pair_concepts_df = build_blind_pair_concept_df(
            blind_pairs_df,
            concept_probabilities,
            concept_probabilities,
            concept_labels,
            top_k=5,
            test_raw_scores=concept_scores,
            train_raw_scores=concept_scores,
        )
        blind_pair_concepts_df.to_csv(config.BLIND_PAIR_CONCEPTS_ARTIFACT, index=False)

        concept_summary_df = summarize_blind_pair_concepts(
            blind_pair_concepts_df,
            blind_pairs_df,
            concept_labels,
        )
        concept_summary_df.to_csv(config.CONCEPT_DELTA_SUMMARY_ARTIFACT, index=False)

    print("\n--- 10. Visualizations & Summary ---")
    from collections import Counter

    import matplotlib.pyplot as plt
    import pandas as pd

    type_counts = blind_pairs_df["blind_type"].value_counts()
    print("\nBlind Retrieval Types Breakdown:")
    print(type_counts)

    print("\nTop Deviant RadGraph Entities per Pathology:")
    for pathology, group in blind_pairs_df.groupby("primary_pathology"):
        missing_list = []
        for entities in group["missing_entities"]:
            missing_list.extend(entities)
        if missing_list:
            counts = Counter(missing_list)
            print(f"\n-- {pathology} --")
            for entity, count in counts.most_common(5):
                print(f"  {count}x : {entity}")

    fig, ax = plt.subplots(figsize=(7, 5))
    types = ["Type 1", "Type 2", "Type 3"]
    counts = [type_counts.get(label, 0) for label in types]
    bars = ax.bar(types, counts, color=["#4878CF", "#6ACC65", "#D65F5F"], width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Blind Retrieval Pairs by Error Type")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plot_path = config.ARTIFACTS_DIR / "blind_pair_types.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    all_missing = []
    for entities in blind_pairs_df["missing_entities"]:
        all_missing.extend(entities)

    top_entities = [entity for entity, _ in Counter(all_missing).most_common(10)]
    freq_data = []
    for blind_type in types:
        type_df = blind_pairs_df[blind_pairs_df["blind_type"] == blind_type]
        type_total = len(type_df)
        type_missing = []
        for entities in type_df["missing_entities"]:
            type_missing.extend(entities)
        type_counts_dict = Counter(type_missing)

        for entity in top_entities:
            count = type_counts_dict.get(entity, 0)
            proportion = (count / type_total * 100) if type_total > 0 else 0
            freq_data.append(
                {
                    "Entity": entity,
                    "Type": blind_type,
                    "Count": count,
                    "Prevalence (%)": round(proportion, 1),
                }
            )

    freq_df = pd.DataFrame(freq_data)
    table_path = config.ARTIFACTS_DIR / "top_missing_entities_by_type.csv"
    freq_df.to_csv(table_path, index=False)
    print(f"Table saved to {table_path}")

    pivot_df = freq_df.pivot(index="Entity", columns="Type", values="Prevalence (%)").fillna(0)
    x = np.arange(len(top_entities))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, pivot_df["Type 1"], width, label="Type 1", color="#4878CF")
    ax.bar(x, pivot_df["Type 2"], width, label="Type 2", color="#6ACC65")
    ax.bar(x + width, pivot_df["Type 3"], width, label="Type 3", color="#D65F5F")
    ax.set_ylabel("Prevalence (% of Pairs Missing Entity)")
    ax.set_title("Top Missing RadGraph Entities by Blind Error Type (Normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(top_entities, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    grouped_bar_path = config.ARTIFACTS_DIR / "missing_entities_grouped_bar.png"
    fig.savefig(grouped_bar_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(pivot_df.values, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(types)))
    ax.set_yticks(np.arange(len(top_entities)))
    ax.set_xticklabels(types)
    ax.set_yticklabels(top_entities)
    ax.set_title("Heatmap: Prevalence (%) of Missing Entities per Error Type")
    for row_index in range(len(top_entities)):
        for col_index in range(len(types)):
            value = pivot_df.values[row_index, col_index]
            color = "black" if value < pivot_df.values.max() / 2 else "white"
            ax.text(col_index, row_index, f"{value:.1f}%", ha="center", va="center", color=color)
    fig.colorbar(cax, label="Prevalence (%)")
    plt.tight_layout()
    heatmap_path = config.ARTIFACTS_DIR / "missing_entities_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    if interpretability == "concepts" and concept_scores is not None and concept_probabilities is not None:
        summary_df = pd.read_csv(config.CONCEPT_DELTA_SUMMARY_ARTIFACT)
        heatmap_df = summary_df.pivot(
            index="concept",
            columns="blind_type",
            values="mean_abs_delta",
        ).fillna(0.0)
        heatmap_df = heatmap_df.reindex(index=concept_labels, fill_value=0.0)
        heatmap_df = heatmap_df.reindex(columns=types, fill_value=0.0)

        fig, ax = plt.subplots(figsize=(8, 7))
        cax = ax.imshow(heatmap_df.values, cmap="OrRd", aspect="auto")
        ax.set_xticks(np.arange(len(heatmap_df.columns)))
        ax.set_yticks(np.arange(len(heatmap_df.index)))
        ax.set_xticklabels(list(heatmap_df.columns))
        ax.set_yticklabels(list(heatmap_df.index))
        ax.set_title("Mean Absolute Raw Concept Score Delta by Blind Type")

        max_val = float(heatmap_df.values.max()) if heatmap_df.size else 0.0
        threshold = max_val / 2.0 if max_val > 0 else 0.0
        for row_index in range(len(heatmap_df.index)):
            for col_index in range(len(heatmap_df.columns)):
                value = float(heatmap_df.values[row_index, col_index])
                color = "white" if value > threshold else "black"
                ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(cax, label="Mean |Raw Score Delta|")
        plt.tight_layout()
        fig.savefig(config.CONCEPT_DELTA_HEATMAP_ARTIFACT, dpi=150)
        plt.close(fig)

    blind_pairs_df["query_uid_str"] = blind_pairs_df["query_uid"].astype(str)
    test["uid_str"] = test["uid"].astype(str)
    dev_df = pd.merge(
        blind_pairs_df,
        test[["uid_str", "radgraph_deviation_full"]],
        left_on="query_uid_str",
        right_on="uid_str",
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_data = []
    labels = []
    for blind_type in types:
        values = dev_df[dev_df["blind_type"] == blind_type]["radgraph_deviation_full"].dropna().values
        plot_data.append(values if len(values) > 0 else [0])
        labels.append(blind_type)

    boxplot = ax.boxplot(plot_data, patch_artist=True, labels=labels, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(boxplot["boxes"], ["#4878CF", "#6ACC65", "#D65F5F"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("RadGraph Deviation from Consensus")
    ax.set_title("Absolute RadGraph Deviation vs. Error Type")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    dev_plot_path = config.ARTIFACTS_DIR / "radgraph_deviation_vs_type.png"
    fig.savefig(dev_plot_path, dpi=150)
    plt.close(fig)

    print("\n====== PIPELINE COMPLETE ======")
    print(f"Results, tables, and plots saved to {config.ARTIFACTS_DIR}")


@app.local_entrypoint()
def main(
    dataset: str = "indiana",
    prepare: str = "auto",
    interpretability: str = "concepts",
    project_id: str = DEFAULT_GCP_PROJECT,
):
    if dataset not in {"indiana", "mimic"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if prepare not in {"auto", "always", "never"}:
        raise ValueError(f"Unsupported prepare mode: {prepare}")
    if interpretability not in {"concepts", "off"}:
        raise ValueError(f"Unsupported interpretability mode: {interpretability}")

    if dataset == "mimic" and prepare != "never":
        print(f"Preparing MIMIC metadata on Modal using project {project_id}...")
        prep_summary = prepare_mimic_dataset.remote(project_id=project_id, mode=prepare)
        print(f"MIMIC prep summary: {prep_summary}")

    print(
        f"Triggering the serverless pipeline on Modal for dataset={dataset}, "
        f"prepare={prepare}, interpretability={interpretability}..."
    )
    run_pipeline_on_modal.remote(
        dataset=dataset,
        prepare_mode=prepare,
        interpretability=interpretability,
    )


def _download_blob_if_missing(bucket, blob_name: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    bucket.blob(blob_name).download_to_filename(str(destination))


def _mimic_outputs_are_current(base_path: Path, local_study_ids: list[str]) -> bool:
    import pandas as pd

    expected = set(local_study_ids)
    if not expected:
        return False

    targets = {
        "reports": base_path / "mimic_reports.csv",
        "projections": base_path / "mimic_projections.csv",
        "chexpert": base_path / "mimic_chexpert.csv",
    }
    if not all(path.exists() and path.stat().st_size > 0 for path in targets.values()):
        return False

    reports = pd.read_csv(targets["reports"])
    projections = pd.read_csv(targets["projections"])
    chexpert = pd.read_csv(targets["chexpert"])

    report_uids = set(reports["uid"].astype(str)) if "uid" in reports.columns else set()
    projection_uids = set(projections["uid"].astype(str)) if "uid" in projections.columns else set()
    chexpert_uids = set(chexpert["uid"].astype(str)) if "uid" in chexpert.columns else set()
    return report_uids == expected and projection_uids == expected and chexpert_uids == expected
