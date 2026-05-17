"""
mimic_prep.py

Pure helpers for building the MIMIC study-level CSVs expected by the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .chexpert_utils import CHEXPERT_PATHOLOGIES

MIMIC_IMAGE_PATH_RE = re.compile(
    r"(?:^|/)(p\d{2})/(p\d+)/(s\d+)/([^/]+\.jpg)$",
    flags=re.IGNORECASE,
)


def normalize_uid(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text.removeprefix("s")


def discover_local_mimic_images(base_path: Path) -> pd.DataFrame:
    """
    Find all MIMIC JPGs mounted under base_path and extract study-level metadata
    from their relative paths.
    """
    records = []
    for image_path in sorted(base_path.rglob("*.jpg")):
        relative = image_path.relative_to(base_path).as_posix()
        match = MIMIC_IMAGE_PATH_RE.search(relative)
        if not match:
            continue

        patient_group, subject_id, study_dir, basename = match.groups()
        records.append(
            {
                "filename": relative,
                "patient_group": patient_group,
                "subject_id": subject_id.removeprefix("p"),
                "study_id": normalize_uid(study_dir),
                "study_dir": study_dir,
                "dicom_id": Path(basename).stem,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "filename",
                "patient_group",
                "subject_id",
                "study_id",
                "study_dir",
                "dicom_id",
            ]
        )

    df = pd.DataFrame.from_records(records).drop_duplicates(subset=["filename"])
    return df.sort_values(["study_id", "filename"]).reset_index(drop=True)


def parse_report_sections(text: str) -> tuple[str, str, str]:
    """
    Parse semi-structured MIMIC report text into findings and impression.
    Falls back to the full text when section headers are missing.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return "", "", ""

    header_re = re.compile(r"(?im)^\s*(findings|impression)\s*:\s*")
    matches = list(header_re.finditer(normalized))

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        section_name = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        section_text = _clean_section_text(normalized[start:end])
        if section_text:
            sections[section_name] = section_text

    findings = sections.get("findings", "")
    impression = sections.get("impression", "")
    fallback = _clean_section_text(normalized)

    if not findings and not impression:
        findings = fallback

    full_text = " ".join(part for part in [findings, impression] if part).strip()
    if not full_text:
        full_text = fallback

    return findings, impression, full_text


def build_mimic_reports_dataframe(
    local_images_df: pd.DataFrame,
    report_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build one row per study from the locally available report text files.
    """
    study_rows = (
        local_images_df[["patient_group", "subject_id", "study_id", "study_dir"]]
        .drop_duplicates()
        .sort_values(["study_id", "subject_id"])
    )

    rows = []
    missing_studies: list[str] = []

    for row in study_rows.itertuples(index=False):
        report_path = report_root / row.patient_group / f"p{row.subject_id}" / f"{row.study_dir}.txt"
        if not report_path.exists():
            missing_studies.append(row.study_id)
            continue

        findings, impression, full_text = parse_report_sections(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "uid": row.study_id,
                "findings": findings,
                "impression": impression,
                "full_text": full_text,
            }
        )

    reports_df = pd.DataFrame(rows, columns=["uid", "findings", "impression", "full_text"])
    reports_df = reports_df.sort_values("uid").reset_index(drop=True) if not reports_df.empty else reports_df
    return reports_df, missing_studies


def build_mimic_projections_dataframe(
    local_images_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join local images to the official metadata table to recover projections.
    """
    if "dicom_id" not in metadata_df.columns or "study_id" not in metadata_df.columns:
        raise KeyError("MIMIC metadata must contain 'dicom_id' and 'study_id'.")

    metadata = metadata_df.copy()
    metadata["dicom_id"] = metadata["dicom_id"].astype(str)
    metadata["study_id"] = metadata["study_id"].map(normalize_uid)
    if "ViewPosition" not in metadata.columns:
        metadata["ViewPosition"] = "UNKNOWN"

    metadata = metadata[["dicom_id", "study_id", "ViewPosition"]].drop_duplicates(
        subset=["study_id", "dicom_id"],
        keep="first",
    )

    merged = local_images_df.merge(
        metadata,
        on=["study_id", "dicom_id"],
        how="left",
    )
    merged["ViewPosition"] = merged["ViewPosition"].fillna("UNKNOWN")

    projections = merged.rename(columns={"study_id": "uid", "ViewPosition": "projection"})
    projections = projections[["uid", "filename", "projection"]]
    projections = projections.sort_values(["uid", "filename"]).reset_index(drop=True)
    return projections


def normalize_mimic_chexpert_dataframe(
    chexpert_df: pd.DataFrame,
    study_ids: Iterable[str],
) -> pd.DataFrame:
    """
    Filter the official MIMIC-JPG CheXpert table down to the locally available studies.
    """
    labels = chexpert_df.copy()
    if "uid" not in labels.columns:
        if "study_id" not in labels.columns:
            raise KeyError("MIMIC CheXpert data must contain 'study_id' or 'uid'.")
        labels["uid"] = labels["study_id"]

    labels["uid"] = labels["uid"].map(normalize_uid)
    for pathology in CHEXPERT_PATHOLOGIES:
        if pathology not in labels.columns:
            labels[pathology] = 0.0

    labels[CHEXPERT_PATHOLOGIES] = (
        labels[CHEXPERT_PATHOLOGIES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    study_id_set = {normalize_uid(study_id) for study_id in study_ids}
    labels = labels[labels["uid"].isin(study_id_set)]
    labels = labels[["uid", *CHEXPERT_PATHOLOGIES]].drop_duplicates(subset=["uid"], keep="first")
    return labels.sort_values("uid").reset_index(drop=True)


def _clean_section_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \n\t:")
