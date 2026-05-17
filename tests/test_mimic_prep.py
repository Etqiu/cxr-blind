import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mimic_prep import (  # noqa: E402
    build_mimic_projections_dataframe,
    build_mimic_reports_dataframe,
    discover_local_mimic_images,
    normalize_mimic_chexpert_dataframe,
    parse_report_sections,
)


class MimicPrepTests(unittest.TestCase):
    def test_parse_report_sections_handles_standard_headers(self):
        text = "FINDINGS: Mild bibasilar opacity.\nIMPRESSION: Small pleural effusion."
        findings, impression, full_text = parse_report_sections(text)
        self.assertEqual(findings, "Mild bibasilar opacity.")
        self.assertEqual(impression, "Small pleural effusion.")
        self.assertEqual(full_text, "Mild bibasilar opacity. Small pleural effusion.")

    def test_parse_report_sections_handles_impression_only(self):
        text = "  Impression : No acute cardiopulmonary abnormality. "
        findings, impression, full_text = parse_report_sections(text)
        self.assertEqual(findings, "")
        self.assertEqual(impression, "No acute cardiopulmonary abnormality.")
        self.assertEqual(full_text, "No acute cardiopulmonary abnormality.")

    def test_parse_report_sections_falls_back_to_full_text(self):
        text = "Portable chest radiograph shows low lung volumes without focal consolidation."
        findings, impression, full_text = parse_report_sections(text)
        self.assertEqual(findings, text)
        self.assertEqual(impression, "")
        self.assertEqual(full_text, text)

    def test_discover_reports_and_projection_join(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            image_dir = base / "physionet.org" / "files" / "mimic-cxr-jpg" / "2.1.0" / "files" / "p10" / "p10000032" / "s50414267"
            image_dir.mkdir(parents=True)
            (image_dir / "aaa111.jpg").write_bytes(b"a")
            (image_dir / "bbb222.jpg").write_bytes(b"b")

            local_images = discover_local_mimic_images(base)
            self.assertEqual(len(local_images), 2)
            self.assertEqual(local_images["study_id"].unique().tolist(), ["50414267"])

            reports_root = base / "_mimic_source" / "reports" / "files"
            report_path = reports_root / "p10" / "p10000032" / "s50414267.txt"
            report_path.parent.mkdir(parents=True)
            report_path.write_text("findings: clear lungs\nimpression: no acute disease")

            reports_df, missing = build_mimic_reports_dataframe(local_images, reports_root)
            self.assertEqual(missing, [])
            self.assertEqual(reports_df.iloc[0]["uid"], "50414267")
            self.assertEqual(reports_df.iloc[0]["impression"], "no acute disease")

            metadata_df = pd.DataFrame(
                [
                    {"study_id": 50414267, "dicom_id": "aaa111", "ViewPosition": "PA"},
                    {"study_id": 50414267, "dicom_id": "bbb222", "ViewPosition": "LATERAL"},
                ]
            )
            projections_df = build_mimic_projections_dataframe(local_images, metadata_df)
            self.assertEqual(projections_df["projection"].tolist(), ["PA", "LATERAL"])
            self.assertTrue(projections_df["filename"].str.endswith(".jpg").all())

    def test_normalize_mimic_chexpert_dataframe_filters_to_local_studies(self):
        chexpert_df = pd.DataFrame(
            [
                {"study_id": 50414267, "Cardiomegaly": 1.0, "No Finding": 0.0},
                {"study_id": 59999999, "Cardiomegaly": 0.0, "No Finding": 1.0},
            ]
        )
        normalized = normalize_mimic_chexpert_dataframe(chexpert_df, ["50414267"])
        self.assertEqual(normalized["uid"].tolist(), ["50414267"])
        self.assertIn("Cardiomegaly", normalized.columns)
        self.assertEqual(float(normalized.iloc[0]["Cardiomegaly"]), 1.0)


if __name__ == "__main__":
    unittest.main()
