"""
radgraph_utils.py

Integration with RadGraph-XL to extract clinical entities and evaluate textual consistency.
"""
import pickle
from tqdm import tqdm
from pathlib import Path

def load_radgraph(model_type="radgraph-xl"):
    """
    Load RadGraph for downstream entity extraction / deviation scoring.
    """
    from radgraph import RadGraph
    radgraph = RadGraph(model_type=model_type)
    return radgraph

def extract_entities(text_list: list[str], model, batch_size=64) -> list[frozenset[str]]:
    """
    Uses radgraph model to extract sets of 'token|label' for clinical consensus.
    """
    ground_truth_entities = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="RadGraph batches"):
        batch_texts = text_list[i: i+batch_size]
        annotations = model(batch_texts)
        for batch_idx, text in enumerate(batch_texts):
            item_key = str(batch_idx)
            if item_key in annotations:
                report_data = annotations[item_key]
                entities_dict = report_data.get('entities', {})
                entity_set = set()
                for ent_id, ent_data in entities_dict.items():
                    token = ent_data['tokens']
                    label = ent_data['label']
                    formatted = f"{token}|{label}".lower()
                    entity_set.add(formatted)
                ground_truth_entities.append(frozenset(entity_set))
            else:
                ground_truth_entities.append(frozenset())
    return ground_truth_entities

def compute_deviation(top_neighbors_idx: list[int],
                      target_entities: set,
                      train_ground_truth_entities: list[frozenset[str]]) -> tuple[int, int, int]:
    """
    Compute specific deviation values for a sequence of neighbor indices
    against the test image's targets.
    Creates a visual consensus: entity must exist in at least min(2, len/2) 
    neighbors, and calculates what is missing and what is unsupported.
    """
    if len(top_neighbors_idx) == 0:
        return 0, 0, 0

    neighbor_entity_sets = [train_ground_truth_entities[idx] for idx in top_neighbors_idx]
    
    from collections import Counter
    all_ents = []
    for subset in neighbor_entity_sets:
        all_ents.extend(list(subset))

    entity_counts = Counter(all_ents)
    consensus_threshold = 2 if len(top_neighbors_idx) >= 2 else 1
    consensus_entities = {
        ent for ent, count in entity_counts.items()
        if count >= consensus_threshold
    }

    unsupported_entities = target_entities - consensus_entities
    missing_entities = consensus_entities - target_entities
    return (
        len(unsupported_entities) + len(missing_entities), # total deviation score
        len(unsupported_entities),
        len(missing_entities),
    )
