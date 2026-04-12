# Combined CXR Pipeline

End-to-End Pipeline adapted from `seagate_data_loader_modal_gpu.ipynb` and `cxr-blind-retrieval`. Engineered specifically for Modal GPU Notebook execution targeting the IU Chest X-Ray dataset.

## Features Included
1. **Parallel Vectorized Embeddings**: Batch embedding over `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, `allenai/biomed_roberta_base` and `microsoft/rad-dino`. Cache-aware to persist checkpoints to Modal volume.
2. **FAISS Indexing**: Uses `IndexFlatIP` over L2 normalized embeddings for exact dense matrix dot-products directly on train features.
3. **RadGraph Entity Extractor**: Pulls clinical tokens from radiology report text (`radgraph-xl`).
4. **CheXpert Heuristic Analysis**: Computes blind and consistent neighbor boundaries and correlates them to specific pathology misalignments.

## Running on Modal

This pipeline assumes the underlying volume data (`indiana_reports.csv` and the `images/` directory) is mounted at `/mnt/radiology-data/archive`.

### 1. From the Interactive Notebook:
Upload or mount `combined_pipeline/combined_pipeline.ipynb` inside your Modal Jupyter container and execute top to bottom. Caching semantics will safely intercept multiple restarts.

### 2. From CLI
If executing natively on the container via script instead of notebook:

```bash
pip install -r requirements.txt
export RADIOLOGY_BASE_PATH="/mnt/radiology-data/archive"
python -m src.run_pipeline
```

## Structure
- `combined_pipeline.ipynb`: Interactive end-to-end driver.
- `src/config.py`: Hardcoded path overrides and batch settings.
- `src/data_loader.py`: Pandas abstractions around the `reports` and `projections` dataframes.
- `src/embeddings.py`: OpenCLIP and HuggingFace AutoModels.
- `src/chexpert_utils.py`: Text-mapping heuristic algorithms bypassing the traditional Docker overhead.
- `src/retrieval.py`: Setup and querying of the FAISS index.
- `src/radgraph_utils.py`: `radgraph` NLP functions.
- `src/analysis.py`: Calculates deviation scores and links pathology categories.
