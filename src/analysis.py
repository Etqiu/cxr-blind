"""
analysis.py

Computes RadGraph entity deviations, consensus modeling, and CheXpert pathology mappings for blind pairs.
"""
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Any
from .radgraph_utils import compute_deviation
from .chexpert_utils import load_chexpert_labels, characterize_blind_pairs
import gc

def build_consensus(
    test_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    I_clip_indices: np.ndarray, 
    test_consistent_neighbors: List[List[int]], 
    test_blind_neighbors: List[List[int]], 
    test_entities: List[frozenset], 
    train_entities: List[frozenset]
) -> pd.DataFrame:
    """
    Construct consensus and output a decorated DataFrame.
    Calculates RadGraph Deviation components from retrieved neighbors.
    """
    deviations_full, unsupported_full, missing_full = [], [], []
    deviations_consistent, unsupported_consistent, missing_consistent = [], [], []
    
    for i in range(len(test_df)):
        target_entities = set(test_entities[i])

        top_full = I_clip_indices[i][:3].astype(int).tolist()
        d_full, u_full, m_full = compute_deviation(top_full, target_entities, train_entities)
        deviations_full.append(d_full)
        unsupported_full.append(u_full)
        missing_full.append(m_full)

        top_consistent = test_consistent_neighbors[i][:3]
        d_con, u_con, m_con = compute_deviation(top_consistent, target_entities, train_entities)
        deviations_consistent.append(d_con)
        unsupported_consistent.append(u_con)
        missing_consistent.append(m_con)

        if i % 250 == 0:
            gc.collect()

    test_df['retrieved_clip_neighbors'] = [tuple(x) for x in I_clip_indices.tolist()]
    test_df['consistent_neighbors'] = [tuple(c) for c in test_consistent_neighbors]
    test_df['blind_neighbors'] = [tuple(b) for b in test_blind_neighbors]

    test_df['radgraph_deviation_full'] = deviations_full
    test_df['unsupported_entities_full'] = unsupported_full
    test_df['missing_entities_full'] = missing_full

    test_df['radgraph_deviation_consistent'] = deviations_consistent
    test_df['unsupported_entities_consistent'] = unsupported_consistent
    test_df['missing_entities_consistent'] = missing_consistent

    return test_df

def connect_blindtype_to_deviation(
    test_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    df_clean: pd.DataFrame, 
    test_entities: List[frozenset], 
    train_entities: List[frozenset]
) -> pd.DataFrame:
    """
    Connect blind pair types to RadGraph deviation patterns and
    return summary of pathology -> top deviant entities.
    """
    chexpert_lookup, pathology_cols = load_chexpert_labels(df_clean)
    
    rows = []
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        q_uid = str(row['uid'])
        q_labels = chexpert_lookup.get(q_uid, {})
        target_ents = set(test_entities[i])
        
        blind_indices = row.get('blind_neighbors', [])
        for n_idx in blind_indices:
            n_row = train_df.iloc[n_idx]
            n_uid = str(n_row['uid'])
            n_labels = chexpert_lookup.get(n_uid, {})
            
            btype = characterize_blind_pairs(q_labels, n_labels, pathology_cols)
            
            n_ents = set(train_entities[n_idx])
            
            unsupported = target_ents - n_ents
            missing = n_ents - target_ents
            
            # primary query pathology approximation
            prim_pathology = next((p for p in pathology_cols if q_labels.get(p) == 1.0), "No Finding")
            
            rows.append({
                "query_uid": q_uid,
                "neighbor_uid": n_uid,
                "primary_pathology": prim_pathology,
                "blind_type": btype,
                "unsupported_entities": tuple(unsupported),
                "missing_entities": tuple(missing)
            })
            
    blind_pair_df = pd.DataFrame(rows)
    return blind_pair_df
