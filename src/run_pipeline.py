"""
run_pipeline.py

Local execution script for running the combined pipeline natively.
"""
import argparse
import os
import pickle
from pathlib import Path
import torch
import numpy as np

from src import config
from src.data_loader import load_data, split_patient_data
from src.embeddings import compute_openclip_image_embeddings, compute_hf_vision_embeddings, compute_text_embeddings
from src.retrieval import build_faiss_index, retrieve_top_k, label_consistent_blind, mask_to_indices
from src.radgraph_utils import load_radgraph, extract_entities
from src.analysis import build_consensus, connect_blindtype_to_deviation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default=config.BASE_PATH_STR)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    
    os.environ["RADIOLOGY_BASE_PATH"] = args.base_path
    
    print(f"Loading data from {args.base_path}...")
    df_clean = load_data(config.REPORTS_FILE, config.PROJECTIONS_FILE)
    
    train_df, test_df = split_patient_data(df_clean)
    print(f"Train/Test split complete. Train: {len(train_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    main()
