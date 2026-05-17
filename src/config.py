"""
config.py

Global configuration variables, paths, and hyperparameters for pipeline execution.
"""
import os
from pathlib import Path
import torch

# Base Paths (default to Modal volume path, overridable via environment variable)
BASE_PATH_STR = os.getenv("RADIOLOGY_BASE_PATH", "/mnt/radiology-data/archive")
BASE_PATH = Path(BASE_PATH_STR)

DATASET_TYPE = os.getenv("DATASET_TYPE", "indiana")
REQUIRE_OFFICIAL_MIMIC_CHEXPERT = os.getenv("EXPECT_OFFICIAL_MIMIC_CHEXPERT", "0") == "1"

# Artifacts & Images
ARTIFACTS_DIR = BASE_PATH / "modal_artifacts"

if DATASET_TYPE == "mimic":
    IMAGES_DIR = BASE_PATH
    REPORTS_FILE = BASE_PATH / "mimic_reports.csv"
    # We may not have a projections file exactly like IU for MIMIC immediately, 
    # but we point to a dummy or mapping if data_loader requires it.
    PROJECTIONS_FILE = BASE_PATH / "mimic_projections.csv" 
    CHEXPERT_FILE = BASE_PATH / "mimic_chexpert.csv"
else:
    IMAGES_DIR = BASE_PATH / "images" / "images_normalized"
    REPORTS_FILE = BASE_PATH / "indiana_reports.csv"
    PROJECTIONS_FILE = BASE_PATH / "indiana_projections.csv"
    CHEXPERT_FILE = BASE_PATH / "chexpert_labels.csv"

DEVIATION_RESULTS_ARTIFACT = ARTIFACTS_DIR / "deviation_results.csv"
BLIND_PAIRS_ARTIFACT = ARTIFACTS_DIR / "blind_pairs_analysis.csv"
BLIND_PAIR_CONCEPTS_ARTIFACT = ARTIFACTS_DIR / "blind_pair_concepts.csv"
CONCEPT_DELTA_SUMMARY_ARTIFACT = ARTIFACTS_DIR / "concept_delta_summary.csv"
CONCEPT_DELTA_HEATMAP_ARTIFACT = ARTIFACTS_DIR / "concept_delta_heatmap.png"
CONCEPT_MATRIX_CACHE = ARTIFACTS_DIR / "concept_matrix.pkl"
CONCEPT_PROMPTS_CACHE = ARTIFACTS_DIR / "concept_prompts.json"
CONCEPT_CALIBRATORS_CACHE = ARTIFACTS_DIR / "concept_calibrators.pkl"
CONCEPT_SCORES_CACHE = ARTIFACTS_DIR / "concept_scores.pkl"

# Batch Sizes & Workers
NUM_IMAGE_WORKERS = max(2, min(8, os.cpu_count() or 2))
OPENCLIP_BATCH_SIZE = 128
TEXT_BATCH_SIZE = 128
RAD_DINO_BATCH_SIZE = 24
RADGRAPH_BATCH_SIZE = 64

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def configure_torch_for_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

# Models
BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMED_ROBERTA_MODEL = "allenai/biomed_roberta_base"
RADDINO_MODEL = "microsoft/rad-dino"
RADGRAPH_MODEL = "radgraph-xl"

# Retrieval & Thresholds
RETRIEVAL_K = 10
CLIP_THRESHOLD = 0.85
DINO_THRESHOLD = 0.60
