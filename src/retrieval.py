"""
retrieval.py

FAISS indexing and rapid cosine similarity search for finding top-k visual neighbors.
"""
import numpy as np
import faiss
import logging

log = logging.getLogger(__name__)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a cosine similarity FAISS index using L2-normalized embeddings.
    """
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build index with empty embeddings")
    
    # Ensure they are L2-normalized for IndexFlatIP (cosine match)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_top_k(index: faiss.IndexFlatIP, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the index for the top-k neighbors.
    Returns: distances (M, k), indices (M, k)
    """
    faiss.normalize_L2(query_embeddings)
    distances, indices = index.search(query_embeddings, k)
    return distances, indices

def label_consistent_blind(
    D_clip: np.ndarray,
    I_clip: np.ndarray,
    test_dino_embeddings: np.ndarray,
    train_dino_embeddings: np.ndarray,
    clip_threshold: float,
    dino_threshold: float
):
    """
    Computes consistency masks natively.
    Returns masks of shapes (M, k) for consistent and blind neighbors.
    """
    retrieved_train_dino = train_dino_embeddings[I_clip]
    # np.einsum for vectorized pairwise dot products
    retrieved_dino_sims = np.einsum('tkd,td->tk', retrieved_train_dino, test_dino_embeddings)

    consistent_mask = retrieved_dino_sims >= dino_threshold
    blind_mask = (~consistent_mask) & (D_clip >= clip_threshold)
    
    return consistent_mask, blind_mask

def mask_to_indices(I_clip: np.ndarray, mask: np.ndarray) -> list[list[int]]:
    """Convert a boolean mask back to lists of train integer indices for each test query."""
    return [
        I_clip[i][mask[i]].astype(int).tolist()
        for i in range(len(I_clip))
    ]
