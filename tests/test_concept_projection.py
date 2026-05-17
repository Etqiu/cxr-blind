import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.concept_projection import (  # noqa: E402
    build_blind_pair_concept_df,
    build_prompt_spec,
    compute_direction_from_embeddings,
    fit_1d_calibrators,
    predict_calibrated_probabilities,
    score_embeddings,
    summarize_blind_pair_concepts,
)


class ConceptProjectionTests(unittest.TestCase):
    def test_build_prompt_spec_uses_canonical_labels(self):
        prompt_spec = build_prompt_spec()
        self.assertEqual(len(prompt_spec), 14)
        self.assertIn("Cardiomegaly", prompt_spec)
        self.assertGreaterEqual(len(prompt_spec["Cardiomegaly"]["positive"]), 3)

    def test_compute_direction_from_embeddings_returns_normalized_vector(self):
        positive = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
        negative = np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
        direction = compute_direction_from_embeddings(positive, negative)
        self.assertEqual(direction.shape, (2,))
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=5)

    def test_calibration_is_1d_and_predicts_probabilities(self):
        pathology_cols = ["Cardiomegaly", "Edema"]
        raw_scores = np.array(
            [
                [0.1, -0.5],
                [0.2, -0.2],
                [0.8, 0.1],
                [1.0, 0.6],
            ],
            dtype=np.float32,
        )
        labels = pd.DataFrame(
            {
                "Cardiomegaly": [0.0, 0.0, 1.0, 1.0],
                "Edema": [0.0, 0.0, 0.0, 0.0],
            }
        )

        calibrators = fit_1d_calibrators(raw_scores, labels, pathology_cols)
        self.assertEqual(calibrators["Cardiomegaly"]["kind"], "logistic")
        self.assertEqual(calibrators["Edema"]["kind"], "constant")

        probabilities = predict_calibrated_probabilities(raw_scores, pathology_cols, calibrators)
        self.assertEqual(probabilities.shape, raw_scores.shape)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_build_blind_pair_concept_df_orders_top_deltas(self):
        blind_pairs_df = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "query_uid": "1",
                    "neighbor_uid": "2",
                    "query_index": 0,
                    "neighbor_index": 1,
                    "query_filename": "q.jpg",
                    "neighbor_filename": "n.jpg",
                }
            ]
        )
        pathology_cols = ["A", "B", "C"]
        probabilities = np.array(
            [
                [0.9, 0.2, 0.4],
                [0.1, 0.8, 0.3],
            ],
            dtype=np.float32,
        )
        raw_scores = score_embeddings(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32),
        )

        concept_df = build_blind_pair_concept_df(
            blind_pairs_df,
            probabilities,
            probabilities,
            pathology_cols,
            top_k=2,
            test_raw_scores=raw_scores,
            train_raw_scores=raw_scores,
        )

        self.assertEqual(len(concept_df), 1)
        top_deltas = json.loads(concept_df.iloc[0]["top_score_deltas"])
        self.assertEqual(len(top_deltas), 2)
        self.assertEqual(top_deltas[0]["concept"], "B")
        self.assertGreaterEqual(abs(top_deltas[0]["score_delta"]), abs(top_deltas[1]["score_delta"]))
        score_deltas = json.loads(concept_df.iloc[0]["concept_score_deltas"])
        self.assertAlmostEqual(float(score_deltas["A"]), 1.0, places=6)
        self.assertAlmostEqual(float(score_deltas["B"]), -1.0, places=6)

    def test_summarize_blind_pair_concepts_builds_type_level_summary(self):
        blind_pairs_df = pd.DataFrame(
            [
                {"pair_id": "p1", "blind_type": "Type 1"},
                {"pair_id": "p2", "blind_type": "Type 2"},
            ]
        )
        blind_pair_concepts_df = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "concept_score_deltas": json.dumps({"A": 0.4, "B": -0.1}),
                    "top_score_deltas": json.dumps(
                        [
                            {"concept": "A", "score_delta": 0.4},
                            {"concept": "B", "score_delta": -0.1},
                        ]
                    ),
                },
                {
                    "pair_id": "p2",
                    "concept_score_deltas": json.dumps({"A": -0.2, "B": 0.6}),
                    "top_score_deltas": json.dumps(
                        [
                            {"concept": "B", "score_delta": 0.6},
                            {"concept": "A", "score_delta": -0.2},
                        ]
                    ),
                },
            ]
        )

        summary = summarize_blind_pair_concepts(blind_pair_concepts_df, blind_pairs_df, ["A", "B"])
        self.assertEqual(len(summary), 4)
        row = summary[(summary["blind_type"] == "Type 1") & (summary["concept"] == "A")].iloc[0]
        self.assertAlmostEqual(float(row["mean_abs_delta"]), 0.4, places=6)
        self.assertEqual(row["delta_kind"], "raw_score")
        self.assertAlmostEqual(float(row["top1_frequency_pct"]), 100.0, places=6)


if __name__ == "__main__":
    unittest.main()
