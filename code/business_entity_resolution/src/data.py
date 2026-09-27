import pathlib
import pandas as pd
from typing import Dict, Tuple

# Constants
DEFAULT_SEP = "\t"
EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

def _validate_columns(df: pd.DataFrame, source_name: str) -> None:
    """Ensure the dataframe has exactly the expected columns.

    Raises
    ------
    ValueError
        If the column set does not match ``EXPECTED_COLUMNS``.
    """
    cols = list(df.columns)
    if cols != EXPECTED_COLUMNS:
        raise ValueError(
            f"{source_name}: unexpected columns {cols}. Expected {EXPECTED_COLUMNS}."
        )

def load_source_tsv(path: pathlib.Path) -> pd.DataFrame:
    """Load a single source TSV file.

    Parameters
    ----------
    path: pathlib.Path
        Path to a ``*.tsv`` file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the four canonical columns.
    """
    if not path.is_file():
        raise FileNotFoundError(f"TSV file not found: {path}")
    df = pd.read_csv(path, sep=DEFAULT_SEP, dtype=str)
    _validate_columns(df, path.name)
    return df

def load_all(train_dir: pathlib.Path, test_dir: pathlib.Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load every training and test source file.

    Returns a tuple ``(train_data, test_data)`` where each element is a dict
    mapping ``source_name`` (e.g. ``"source1"``) to its DataFrame.
    """
    train_data = {}
    test_data = {}
    for src in ["source1", "source2", "source3"]:
        train_path = train_dir / f"train_{src}.tsv"
        test_path = test_dir / f"test_{src}.tsv"
        train_data[src] = load_source_tsv(train_path)
        test_data[src] = load_source_tsv(test_path)
    return train_data, test_data

def sanity_check(train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]) -> None:
    """Print a quick sanity report for each source.
    """
    print("=== Training data summary ===")
    for src, df in train_data.items():
        print(f"{src}: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1_048_576:.2f} MiB")
        print(df.head(5).to_string(index=False))
        print("---")
    print("=== Test data summary ===")
    for src, df in test_data.items():
        print(f"{src}: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1_048_576:.2f} MiB")
        print(df.head(5).to_string(index=False))
        print("---")

if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).resolve().parents[3]
    TRAIN_DIR = ROOT / "dataset" / "train"
    TEST_DIR = ROOT / "dataset" / "test"
    train, test = load_all(TRAIN_DIR, TEST_DIR)
    sanity_check(train, test)
