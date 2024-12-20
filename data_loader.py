import logging
from typing import Optional, List
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_data(
    path: str,
    dataset_type: str,
    name: Optional[str] = None,
    columns: Optional[List[str]] = None,
):
    """
    Load and return a dataset from a given CSV file path using Hugging Face datasets.

    Depending on `dataset_type`, it selects columns appropriate for unlabeled or
    labeled datasets. If `columns` are provided, they override the default selection.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the dataset.
    dataset_type : str
        Type of dataset to load.
        - "unlabeled" for datasets containing 'query', 'dish'.
        - "labeled" for datasets containing 'query', 'dish', 'label'.
    name : str, optional
        A name to associate with the returned dataset, mainly for identification.
    columns : list of str, optional
        Columns to select after loading the dataset. If None, defaults are chosen based on `dataset_type`.

    Returns
    -------
    datasets.Dataset or datasets.DatasetDict or dict
        The loaded dataset. If `name` is provided, returns a dict with `name` as the key.
        If multiple splits exist in the CSV, a `datasets.DatasetDict` is returned.

    Raises
    ------
    ValueError
        If an invalid `dataset_type` is provided or required columns are missing.
    """
    # Determine default columns based on dataset_type
    if columns is None:
        if dataset_type == "unlabeled":
            columns = ["query", "dish"]
        elif dataset_type == "labeled":
            columns = ["query", "dish", "label"]
        else:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. Allowed values are 'unlabeled' or 'labeled'."
            )

    logger.info(f"Loading dataset from {path} with dataset_type={dataset_type}")
    dataset = load_dataset("csv", data_files=path)

    for split in dataset:
        missing_columns = [
            col for col in columns if col not in dataset[split].column_names
        ]
        if missing_columns:
            raise ValueError(
                f"The following required columns are missing from the dataset: {missing_columns}"
            )

    dataset = dataset.select_columns(columns)
    logger.info(f"Dataset loaded successfully from {path}")

    return dataset if name is None else {name: dataset}
