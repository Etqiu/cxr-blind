"""
run_modal.py

Main orchestrator script to execute the combined CXR blind retrieval pipeline on Modal's serverless GPU infrastructure.
Handles remote caching, embeddings, FAISS retrieval, RadGraph parsing, and visualization generation.
"""

import modal
import os

# Create the Modal app
app = modal.App("cxr-combined-pipeline")

# Explicitly attach the local 'src' directory to the Docker image
local_src_dir = os.path.join(os.path.dirname(__file__), "src")

# Define the remote environment (Docker image + dependencies)
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
        "radgraph"
    )
    .add_local_dir(local_src_dir, remote_path="/root/src")
)

# Connect to the Modal Volume containing the dataset
# Adjust the Volume name if yours is different from 'radiology-data'
try:
    volume = modal.Volume.from_name("radiology-data", create_if_missing=True)
except Exception:
    volume = modal.Volume.from_name("radiology-data")

@app.function(
    image=image,
    gpu="T4",  # Requesting a GPU
    volumes={"/mnt/radiology-data": volume},
    timeout=86400  # Allow long execution times (up to 24hours)
)
def run_pipeline_on_modal():
    """
    This function acts exactly like the notebook, but runs on Modal servers.
    Modal automatically synchronizes your local 'src' directory so we can import it.
    """
    import os
    import sys
    import pickle
    import numpy as np

    # Ensure config uses the modal mounted path explicitly
    os.environ["RADIOLOGY_BASE_PATH"] = "/mnt/radiology-data/archive"
    
    # Now we can import our source cleanly!
    from src import config
    from src.data_loader import load_data, split_patient_data
    from src.embeddings import compute_openclip_image_embeddings, compute_hf_vision_embeddings, compute_text_embeddings
    from src.retrieval import build_faiss_index, retrieve_top_k, label_consistent_blind, mask_to_indices
    from src.radgraph_utils import load_radgraph, extract_entities
    from src.analysis import build_consensus, connect_blindtype_to_deviation
    import open_clip
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    
    print("====== MODAL GPU APP STARTING ======")
    print(f"Mounted Base Path: {config.BASE_PATH}")
    
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Load Data ---
    print("\n--- 1. Loading dataset ---")
    df_clean = load_data(config.REPORTS_FILE, config.PROJECTIONS_FILE)
    
    # --- Step 3: BioMedCLIP ---
    print("\n--- 2. BioMedCLIP Image Embeddings ---")
    clip_cache = config.ARTIFACTS_DIR / "image_embeddings.pkl"
    if clip_cache.exists():
        print(f"Loading {clip_cache}")
        with open(clip_cache, "rb") as f:
            clip_emb_dict = pickle.load(f)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(config.BIOMEDCLIP_MODEL)
        model.to(config.DEVICE).eval()
        clip_emb_dict = compute_openclip_image_embeddings(
            df_clean['filename'].dropna().unique(), config.IMAGES_DIR, preprocess, model, config.DEVICE,
            config.OPENCLIP_BATCH_SIZE, config.NUM_IMAGE_WORKERS
        )
        with open(clip_cache, "wb") as f:
            pickle.dump(clip_emb_dict, f)
            
    df_clean['image_embedding'] = df_clean['filename'].map(clip_emb_dict)
    df_clean = df_clean.dropna(subset=['image_embedding']).copy()

    # --- Step 4: RAD-DINO ---
    print("\n--- 3. RAD-DINO Image Embeddings ---")
    dino_cache = config.ARTIFACTS_DIR / "raddino_embeddings.pkl"
    if dino_cache.exists():
        print(f"Loading {dino_cache}")
        with open(dino_cache, "rb") as f:
            dino_emb_dict = pickle.load(f)
    else:
        processor = AutoImageProcessor.from_pretrained(config.RADDINO_MODEL)
        d_model = AutoModel.from_pretrained(config.RADDINO_MODEL).to(config.DEVICE).eval()
        dino_emb_dict = compute_hf_vision_embeddings(
            df_clean['filename'].dropna().unique(), config.IMAGES_DIR, processor, d_model, config.DEVICE,
            config.RAD_DINO_BATCH_SIZE, config.NUM_IMAGE_WORKERS, dino_cache, 10
        )
    df_clean['raddino_embedding'] = df_clean['filename'].map(dino_emb_dict)
    df_clean = df_clean.dropna(subset=['raddino_embedding']).copy()

    # --- Step 5: BioMed-RoBERTa Text ---
    print("\n--- 4. BioMed-RoBERTa Text Embeddings ---")
    text_cache = config.ARTIFACTS_DIR / "text_embeddings.pkl"
    if text_cache.exists():
        print(f"Loading {text_cache}")
        with open(text_cache, "rb") as f:
            text_emb = pickle.load(f)
    else:
        t_tokenizer = AutoTokenizer.from_pretrained(config.BIOMED_ROBERTA_MODEL)
        t_model = AutoModel.from_pretrained(config.BIOMED_ROBERTA_MODEL).to(config.DEVICE).eval()
        text_emb = compute_text_embeddings(
            df_clean['full_text'].tolist(), t_model, t_tokenizer, config.DEVICE, config.TEXT_BATCH_SIZE
        )
        with open(text_cache, "wb") as f:
            pickle.dump(text_emb, f)
    df_clean['text_embedding'] = list(text_emb)

    # --- Train/Test Setup ---
    # The original cxr-blind-retrieval pipeline evaluated *every image* against 
    # the entire database without a predefined Train/Test split!
    # To rigorously replicate it, we use `df_clean` for both queries & index.
    train = df_clean.copy()
    test = df_clean.copy()
    print(f"\n--- 5. Using Full Dataset for Replication: {len(train)} Images ---")

    # --- Step 7 & 8: Retrieval & Labeling ---
    print("\n--- 6. FAISS Retrieval & Labeling ---")
    train_img_embs = np.stack(train['image_embedding']).astype('float32')
    test_img_embs = np.stack(test['image_embedding']).astype('float32')
    train_dino_embs = np.stack(train['raddino_embedding']).astype('float32')
    test_dino_embs = np.stack(test['raddino_embedding']).astype('float32')

    img_index = build_faiss_index(train_img_embs)
    
    # Retrieve top K + 1 (the 1st slot is the exact image itself matched against its own index)
    D_clip_full, I_clip_full = retrieve_top_k(img_index, test_img_embs, config.RETRIEVAL_K + 1)
    
    # Strip off the self-match at rank 0 (the query retrieving itself distance=1.0)
    D_clip = D_clip_full[:, 1:]
    I_clip = I_clip_full[:, 1:]
    
    c_mask, b_mask = label_consistent_blind(D_clip, I_clip, test_dino_embs, train_dino_embs, config.CLIP_THRESHOLD, config.DINO_THRESHOLD)
    test_cons_neighbors = mask_to_indices(I_clip, c_mask)
    test_blind_neighbors = mask_to_indices(I_clip, b_mask)

    # --- Step 9: RadGraph ---
    print("\n--- 7. RadGraph Entity Extraction ---")
    consolidated_rg_cache = config.ARTIFACTS_DIR / "full_dataset_radgraph_entities.pkl"
    if consolidated_rg_cache.exists():
        with open(consolidated_rg_cache, "rb") as f:
            full_ents = pickle.load(f)
        train_ents = full_ents
        test_ents = full_ents
    else:
        rg_model = load_radgraph(config.RADGRAPH_MODEL)
        full_ents = extract_entities(df_clean['full_text'].tolist(), rg_model, config.RADGRAPH_BATCH_SIZE)
        with open(consolidated_rg_cache, "wb") as f: pickle.dump(full_ents, f)
        train_ents = full_ents
        test_ents = full_ents

    # --- Step 10 & 11: Consensus & CheXpert ---
    print("\n--- 8. Consensus & CheXpert Analysis ---")
    test = build_consensus(test, train, I_clip, test_cons_neighbors, test_blind_neighbors, test_ents, train_ents)
    blind_pairs_df = connect_blindtype_to_deviation(test, train, df_clean, test_ents, train_ents)
    
    results_path = config.ARTIFACTS_DIR / "deviation_results.csv"
    test.to_csv(results_path, index=False)
    blind_pairs_df.to_csv(config.ARTIFACTS_DIR / "blind_pairs_analysis.csv", index=False)
    
    print("\n--- 9. Visualizations & Step 7 Summary ---")
    import matplotlib.pyplot as plt
    from collections import Counter
    
    type_counts = blind_pairs_df['blind_type'].value_counts()
    print("\nBlind Retrieval Types Breakdown:")
    print(type_counts)
    
    print("\nTop Deviant RadGraph Entities per Pathology:")
    for pathology, group in blind_pairs_df.groupby('primary_pathology'):
        missing_list = []
        for m_ents in group['missing_entities']:
            missing_list.extend(m_ents)
        if missing_list:
            c = Counter(missing_list)
            print(f"\n-- {pathology} --")
            for ent, count in c.most_common(5):
                print(f"  {count}x : {ent}")
                
    # Save the Bar Chart
    fig, ax = plt.subplots(figsize=(7, 5))
    types = ['Type 1', 'Type 2', 'Type 3']
    counts = [type_counts.get(t, 0) for t in types]
    bars = ax.bar(types, counts, color=["#4878CF", "#6ACC65", "#D65F5F"], width=0.5)
    for bar, num in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(num), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Blind Retrieval Pairs by Error Type")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plot_path = config.ARTIFACTS_DIR / "blind_pair_types.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # 1. Gather all missing entities to find the global top 10
    import pandas as pd
    import numpy as np
    
    all_missing = []
    for m_ents in blind_pairs_df['missing_entities']:
        all_missing.extend(m_ents)
    
    top_entities = [ent for ent, _ in Counter(all_missing).most_common(10)]
    
    # 2. Build frequency table (Entity vs Type)
    freq_data = []
    for t in types:
        type_df = blind_pairs_df[blind_pairs_df['blind_type'] == t]
        type_missing = []
        for m_ents in type_df['missing_entities']:
            type_missing.extend(m_ents)
        type_counts_dict = Counter(type_missing)
        
        for ent in top_entities:
            freq_data.append({
                "Entity": ent,
                "Type": t,
                "Count": type_counts_dict.get(ent, 0)
            })
            
    freq_df = pd.DataFrame(freq_data)
    
    # 3. Save as CSV Table
    table_path = config.ARTIFACTS_DIR / "top_missing_entities_by_type.csv"
    freq_df.to_csv(table_path, index=False)
    print(f"Table saved to {table_path}")
    
    # Pivot for plotting
    pivot_df = freq_df.pivot(index="Entity", columns="Type", values="Count").fillna(0)
    
    # 4. Grouped Bar Plot
    x = np.arange(len(top_entities))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, pivot_df['Type 1'], width, label='Type 1', color="#4878CF")
    ax.bar(x, pivot_df['Type 2'], width, label='Type 2', color="#6ACC65")
    ax.bar(x + width, pivot_df['Type 3'], width, label='Type 3', color="#D65F5F")
    
    ax.set_ylabel('Missing Entity Count')
    ax.set_title('Top Missing RadGraph Entities by Blind Error Type')
    ax.set_xticks(x)
    ax.set_xticklabels(top_entities, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    grouped_bar_path = config.ARTIFACTS_DIR / "missing_entities_grouped_bar.png"
    fig.savefig(grouped_bar_path, dpi=150)
    plt.close(fig)

    # 5. Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(pivot_df.values, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(len(types)))
    ax.set_yticks(np.arange(len(top_entities)))
    ax.set_xticklabels(types)
    ax.set_yticklabels(top_entities)
    ax.set_title("Heatmap: Missing RadGraph Entities per Error Type")
    
    # Annotate Heatmap
    for i in range(len(top_entities)):
        for j in range(len(types)):
            val = int(pivot_df.values[i, j])
            ax.text(j, i, str(val), ha="center", va="center", color="black" if val < pivot_df.values.max()/2 else "white")
            
    fig.colorbar(cax)
    plt.tight_layout()
    heatmap_path = config.ARTIFACTS_DIR / "missing_entities_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    
    print("\n====== PIPELINE COMPLETE ======")
    print(f"Results, tables, and plots saved to {config.ARTIFACTS_DIR}")

@app.local_entrypoint()
def main():
    print("Triggering the Serverless execution on Modal...")
    run_pipeline_on_modal.remote()
