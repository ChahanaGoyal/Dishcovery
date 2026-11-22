"""
data_loader.py
==============
Utilities for loading and validating raw recipe data.

The primary entry point is :func:`load_csv`, which reads any CSV file into a
Pandas DataFrame. :func:`validate_foodcom_columns` then checks that the
Food.com ``RAW_recipes.csv`` columns required by Dishcovery are present.
:func:`get_dataset_path` resolves the recipe dataset location locally or downloads
it from a public Hugging Face dataset repository if missing.

Food.com column → Dishcovery name mapping
------------------------------------------
``id``          → recipe identifier
``name``        → recipe title
``ingredients`` → ingredient list (stored as a Python-list string in the CSV)
``minutes``     → total cooking time in minutes
``tags``        → recipe tags (used for dietary classification)

Memory note
-----------
RAW_recipes.csv contains 12 columns but Dishcovery only uses 5.
Always pass ``usecols`` to :func:`load_csv` to avoid loading unused columns
(``contributor_id``, ``submitted``, ``nutrition``, ``n_steps``, ``steps``,
``description``, ``n_ingredients``) which together add ~230 MB of RAM.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that Dishcovery needs from RAW_recipes.csv.
# These are the *raw* Food.com column names (before any renaming).
FOODCOM_REQUIRED_COLUMNS: list[str] = [
    "id",
    "name",
    "ingredients",
    "minutes",
]

# Columns to load from the CSV — only what Dishcovery actually needs.
# Loading the full 12-column file uses ~358 MB; restricting to these 5
# columns reduces that to ~130 MB, saving ~230 MB per deployment instance.
FOODCOM_LOAD_COLUMNS: list[str] = [
    "id",
    "name",
    "minutes",
    "ingredients",
    "tags",  # needed for dietary classification
]

# Default Hugging Face repository and file for hosted dataset retrieval.
DEFAULT_HF_REPO_ID: str = os.getenv("DISHCOVERY_HF_REPO", "Chahanagoyal/dishcovery-recipes")
DEFAULT_DATASET_FILENAME: str = "RAW_recipes.csv"

# Timeout in seconds for Hugging Face downloads (prevents indefinite hangs).
HF_DOWNLOAD_TIMEOUT_SECONDS: int = int(os.getenv("DISHCOVERY_HF_TIMEOUT", "120"))


def load_csv(
    path: str | Path,
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file and return a Pandas DataFrame.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.
    nrows:
        If given, read only this many rows.  Useful for quickly inspecting a
        large file without loading it entirely into memory.
    usecols:
        If given, load only these columns.  Reduces memory usage when only a
        subset of columns is needed.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested rows and columns.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as a valid CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        df = pd.read_csv(path, nrows=nrows, usecols=usecols)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file '{path}': {exc}") from exc

    logger.info(
        "Loaded %d rows × %d columns from '%s'",
        len(df),
        len(df.columns),
        path,
    )
    return df


def validate_foodcom_columns(df: pd.DataFrame) -> None:
    """Check that *df* contains the Food.com columns Dishcovery requires.

    Parameters
    ----------
    df:
        DataFrame loaded from ``RAW_recipes.csv``.

    Raises
    ------
    ValueError
        If one or more required columns are absent.
    """
    missing = [col for col in FOODCOM_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required Food.com columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    logger.info("All required Food.com columns present: %s", FOODCOM_REQUIRED_COLUMNS)


def get_data_dir(subdir: str = "raw") -> Path:
    """Return the absolute path to a data sub-directory.

    Parameters
    ----------
    subdir:
        Either ``"raw"`` or ``"processed"``.
    """
    root = Path(__file__).resolve().parents[1]  # project root
    return root / "data" / subdir


def download_dataset_from_hf(
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_DATASET_FILENAME,
    local_dir: str | Path | None = None,
    max_retries: int = 2,
) -> Path:
    """Download a dataset file from a public Hugging Face repository using hf_hub_download.

    No Hugging Face authentication token is required because the repository is public.
    Downloads are attempted up to ``max_retries`` times with a timeout of
    ``HF_DOWNLOAD_TIMEOUT_SECONDS`` seconds to prevent indefinite hangs on
    Streamlit Cloud where idle worker processes can be killed mid-download.

    Parameters
    ----------
    repo_id:
        Hugging Face dataset repository identifier (default: "Chahanagoyal/dishcovery-recipes").
    filename:
        Dataset filename to retrieve (default: "RAW_recipes.csv").
    local_dir:
        Target directory where the file should be saved. If None, saves to the Hugging Face
        default hub cache.
    max_retries:
        Number of additional attempts after the first failure (default: 2).

    Returns
    -------
    Path
        Path to the downloaded or cached file.

    Raises
    ------
    RuntimeError
        If the file cannot be retrieved from Hugging Face after all retries.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Add 'huggingface_hub>=0.20,<1.0' to requirements.txt."
        ) from exc

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            logger.info(
                "Downloading '%s' from public Hugging Face repository '%s' (attempt %d/%d)...",
                filename,
                repo_id,
                attempt,
                max_retries + 1,
            )
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir) if local_dir is not None else None,
                token=None,  # Public repository — no authentication token required
            )
            path = Path(downloaded)
            logger.info(
                "Successfully downloaded '%s' to '%s' on attempt %d.",
                filename,
                path,
                attempt,
            )
            return path
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed to download '%s' from '%s': %s",
                attempt,
                max_retries + 1,
                filename,
                repo_id,
                exc,
            )

    logger.error(
        "All %d download attempts failed for '%s' from '%s': %s",
        max_retries + 1,
        filename,
        repo_id,
        last_exc,
        exc_info=True,
    )
    raise RuntimeError(
        f"Could not download dataset '{filename}' from Hugging Face repository '{repo_id}' "
        f"after {max_retries + 1} attempts: {last_exc}"
    ) from last_exc


def get_dataset_path(
    filename: str = DEFAULT_DATASET_FILENAME,
    repo_id: str = DEFAULT_HF_REPO_ID,
    download_if_missing: bool = True,
    local_dir: str | Path | None = None,
) -> Path:
    """Resolve the dataset path locally, downloading from Hugging Face if absent.

    1. Checks if the file exists locally in `local_dir` (defaults to `get_data_dir("raw")`).
    2. If found locally, returns that Path immediately without network calls.
    3. If absent and `download_if_missing` is True, downloads the file from the public
       Hugging Face repository using `download_dataset_from_hf`.
    4. If absent and `download_if_missing` is False, raises FileNotFoundError.

    Parameters
    ----------
    filename:
        Dataset filename (default: "RAW_recipes.csv").
    repo_id:
        Hugging Face dataset repository (default: "Chahanagoyal/dishcovery-recipes").
    download_if_missing:
        Whether to download from Hugging Face if not found locally (default: True).
    local_dir:
        Directory to check for local file (defaults to `get_data_dir("raw")`).

    Returns
    -------
    Path
        Path to the local or downloaded dataset file.

    Raises
    ------
    FileNotFoundError
        If the file is missing locally and `download_if_missing` is False.
    RuntimeError
        If the download from Hugging Face fails.
    """
    target_dir = Path(local_dir) if local_dir is not None else get_data_dir("raw")
    local_file = target_dir / filename

    if local_file.exists():
        logger.info("Found local dataset file at '%s'", local_file)
        return local_file

    if not download_if_missing:
        raise FileNotFoundError(
            f"Dataset file '{filename}' not found locally at '{local_file}' "
            "and download_if_missing is False."
        )

    logger.info(
        "Local dataset file '%s' not found. Fetching from Hugging Face repository '%s'...",
        local_file,
        repo_id,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    return download_dataset_from_hf(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
    )
