"""
data_loader.py

Loads and cleans raw reports/projections into pandas DataFrames, handles dataset splits.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(reports_file: Path, projections_file: Path):
    """
    Load reports and projections dataframes, merge them, and clean up.
    Returns:
        df: Merged dataframe.
    """
    if reports_file.exists():
        df_reports = pd.read_csv(reports_file)
    else:
        raise FileNotFoundError(f"Missing {reports_file}")

    if projections_file.exists():
        df_projections = pd.read_csv(projections_file)
    else:
        raise FileNotFoundError(f"Missing {projections_file}")

    df = df_reports.merge(df_projections, on='uid', how='left')

    df['findings'] = df['findings'].fillna("")
    df['impression'] = df['impression'].fillna("")
    df['full_text'] = (df['findings'] + " " + df['impression']).str.strip()
    
    df_clean = df[df['full_text'].str.len() > 0].copy()
    return df_clean

def split_patient_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset on 'uid' preserving patient-level separation without leakage.
    Returns:
        train, test: pd.DataFrame
    """
    unique_uids = df['uid'].unique()
    train_uids, test_uids = train_test_split(unique_uids, test_size=test_size, random_state=random_state)
    train = df[df['uid'].isin(train_uids)].copy()
    test = df[df['uid'].isin(test_uids)].copy()
    return train, test
