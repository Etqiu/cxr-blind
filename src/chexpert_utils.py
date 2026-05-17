"""
chexpert_utils.py

Heuristic-based CheXpert pathology classification for evaluating Type 1/2/3 blind retrieval errors.
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

# A simplified, generic mapping based purely on string matching of `Problems` text from indiana dataset.
CHEXPERT_PATHOLOGIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

from pathlib import Path
from . import config

def normalize_uid(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text.removeprefix("s")


def load_chexpert_table(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load a normalized study-level CheXpert table with columns `uid` + canonical labels.
    """
    pathology_cols = CHEXPERT_PATHOLOGIES

    if config.CHEXPERT_FILE.exists():
        print(f"Loading official CheXpert labels from {config.CHEXPERT_FILE}...")
        df_chexpert = pd.read_csv(config.CHEXPERT_FILE)
        return normalize_chexpert_table(df_chexpert, pathology_cols), pathology_cols

    if config.DATASET_TYPE == "mimic" and config.REQUIRE_OFFICIAL_MIMIC_CHEXPERT:
        raise FileNotFoundError(
            f"Expected official MIMIC CheXpert labels at {config.CHEXPERT_FILE}, but the file is missing."
        )

    print(
        f"File {config.CHEXPERT_FILE} not found. Falling back to heuristic text mapping "
        "(Type 2 errors will be 0)..."
    )
    heuristic_rows = []
    unique_rows = df_clean.drop_duplicates(subset=["uid"])
    for _, row in unique_rows.iterrows():
        uid = normalize_uid(row["uid"])
        heuristic_rows.append({"uid": uid, **heuristic_label_row(row)})

    return pd.DataFrame(heuristic_rows, columns=["uid", *pathology_cols]), pathology_cols


def normalize_chexpert_table(df_chexpert: pd.DataFrame, pathology_cols: list[str]) -> pd.DataFrame:
    normalized = df_chexpert.copy()
    if "uid" not in normalized.columns:
        if "study_id" in normalized.columns:
            normalized["uid"] = normalized["study_id"]
        else:
            raise KeyError("CheXpert data must contain either 'uid' or 'study_id'.")

    normalized["uid"] = normalized["uid"].map(normalize_uid)
    for pathology in pathology_cols:
        if pathology not in normalized.columns:
            normalized[pathology] = 0.0

    normalized[pathology_cols] = (
        normalized[pathology_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    normalized = normalized[["uid", *pathology_cols]].drop_duplicates(subset=["uid"], keep="first")
    return normalized.reset_index(drop=True)


def align_chexpert_labels(
    df_clean: pd.DataFrame,
    chexpert_table: pd.DataFrame,
    pathology_cols: list[str],
) -> pd.DataFrame:
    """
    Expand a study-level CheXpert table to match the row order of df_clean.
    """
    aligned = df_clean[["uid"]].copy()
    aligned["uid"] = aligned["uid"].map(normalize_uid)
    merged = aligned.merge(chexpert_table, on="uid", how="left")
    merged[pathology_cols] = merged[pathology_cols].fillna(0.0)
    return merged[["uid", *pathology_cols]]


def load_chexpert_labels(df_clean: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Returns: lookup mapping `uid` to its approximated CheXpert pathology dictionary.
    Will attempt to load the actual `chexpert_labels.csv` to capture Type 2 (-1.0) errors.
    If the file is missing, it will gracefully fallback to the basic heuristic regex parser.
    """
    chexpert_table, pathology_cols = load_chexpert_table(df_clean)
    lookup = {}
    for _, row in chexpert_table.iterrows():
        lookup[row["uid"]] = {pathology: float(row.get(pathology, 0.0)) for pathology in pathology_cols}
    return lookup, pathology_cols


def heuristic_label_row(row: pd.Series) -> dict[str, float]:
    combined = " ".join(
        str(row.get(field, "")).lower()
        for field in ["Problems", "MeSH", "findings", "impression", "full_text"]
    )

    labels = {pathology: 0.0 for pathology in CHEXPERT_PATHOLOGIES}
    if "normal" in combined or "unremarkable" in combined:
        labels["No Finding"] = 1.0
    if "cardiomegaly" in combined or "enlarged heart" in combined:
        labels["Cardiomegaly"] = 1.0
    if "opacity" in combined or "infiltrate" in combined:
        labels["Lung Opacity"] = 1.0
    if "nodule" in combined or "mass" in combined or "lesion" in combined:
        labels["Lung Lesion"] = 1.0
    if "edema" in combined or "failure" in combined:
        labels["Edema"] = 1.0
    if "consolidation" in combined:
        labels["Consolidation"] = 1.0
    if "pneumonia" in combined:
        labels["Pneumonia"] = 1.0
    if "atelectasis" in combined:
        labels["Atelectasis"] = 1.0
    if "pneumothorax" in combined:
        labels["Pneumothorax"] = 1.0
    if "effusion" in combined:
        labels["Pleural Effusion"] = 1.0
    if "fracture" in combined:
        labels["Fracture"] = 1.0
    if "device" in combined or "tube" in combined or "line" in combined or "catheter" in combined:
        labels["Support Devices"] = 1.0
    return labels

def characterize_blind_pairs(labels_q: dict, labels_n: dict, pathology_cols: list) -> str:
    """
    Assign a blind type to a single query–neighbour pair.
    Type 1 — Same primary pathology, different secondary labels
    Type 2 — Same pathology, different severity (approx: positive/uncertain mismatch, skipped due to heuristic mapping providing strictly binary 0.0/1.0).
    Type 3 — Different pathologies entirely
    """
    pos_q = {p for p in pathology_cols if labels_q.get(p) == 1.0}
    pos_n = {p for p in pathology_cols if labels_n.get(p) == 1.0}
    unc_q = {p for p in pathology_cols if labels_q.get(p) == -1.0} # Our heuristic doesn't generate -1
    unc_n = {p for p in pathology_cols if labels_n.get(p) == -1.0}

    shared_pos = pos_q & pos_n
    pos_unc_mismatch = (pos_q & unc_n) | (unc_q & pos_n)

    if pos_unc_mismatch:
        return "Type 2"

    if not shared_pos:
        return "Type 3"

    differ = any(
        labels_q.get(p) != labels_n.get(p)
        for p in pathology_cols
        if p not in shared_pos
    )
    return "Type 1" if differ else "Type 1"
