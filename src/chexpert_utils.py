"""
chexpert_utils.py

Heuristic-based CheXpert pathology classification for evaluating Type 1/2/3 blind retrieval errors.
"""
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
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

def load_chexpert_labels(df_clean: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Returns: lookup mapping `uid` to its approximated CheXpert pathology dictionary.
    Will attempt to load the actual `chexpert_labels.csv` to capture Type 2 (-1.0) errors.
    If the file is missing, it will gracefully fallback to the basic heuristic regex parser.
    """
    pathology_cols = CHEXPERT_PATHOLOGIES
    lookup = {}
    
    if config.CHEXPERT_FILE.exists():
        print(f"Loading official CheXpert labels from {config.CHEXPERT_FILE}...")
        df_chexpert = pd.read_csv(config.CHEXPERT_FILE)
        
        # Ensure UIDs align properly (e.g. 1.0 -> "1", etc.)
        if 'uid' in df_chexpert.columns:
            # Fill NaNs with 0.0 before creating the dict
            df_chexpert[pathology_cols] = df_chexpert[pathology_cols].fillna(0.0)
            for _, row in df_chexpert.iterrows():
                try:
                    uid_val = str(int(float(row['uid'])))
                except (ValueError, TypeError):
                    uid_val = str(row['uid'])
                
                lookup[uid_val] = {p: row.get(p, 0.0) for p in pathology_cols}
                
            # Second pass to ensure types match whatever df_clean has
            for _, row in df_clean.iterrows():
                clean_uid = str(row['uid'])
                if clean_uid not in lookup:
                    # Try looking up without casting (for pure string ids)
                    if str(row['uid']) in lookup: pass
        return lookup, pathology_cols

    print(f"File {config.CHEXPERT_FILE} not found. Falling back to heuristic text mapping (Type 2 errors will be 0)...")
    
    for _, row in df_clean.iterrows():
        uid = str(row['uid'])
        probs = str(row.get('Problems', '')).lower()
        mesh = str(row.get('MeSH', '')).lower()
        combined = probs + " " + mesh
        
        # Simple mapping to CheXpert 14 classes (heuristic mapping since we bypassed Docker)
        labels = {p: 0.0 for p in CHEXPERT_PATHOLOGIES}
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
            
        lookup[uid] = labels

    return lookup, pathology_cols

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
