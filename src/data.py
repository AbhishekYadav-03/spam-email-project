"""
Data module for spam email classifier.

This module handles:
- Loading the spam dataset
- Preprocessing and feature engineering
- Train/test splitting
- Text vectorization

Dataset:
    The SMS Spam Collection is a set of SMS tagged messages that have been
    collected for SMS Spam research. It contains 5,572 messages (4825 ham,
    747 spam).
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Set up logging
logger = logging.getLogger(__name__)


def load_dataset(data_path: str = "spam.csv") -> pd.DataFrame:
    """
    Load the SMS spam dataset from CSV file.

    The dataset is expected to have the following format:
    - First column (v1): Label (ham/spam)
    - Second column (v2): Message text
    - Additional columns are ignored

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['label', 'message']
        where label is binary (0=ham, 1=spam).

    Raises
    ------
    FileNotFoundError
        If the specified data file does not exist.
    """
    file_path = Path(data_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please download the SMS Spam Collection dataset "
            "or specify the correct path."
        )

    # Load only the first two columns as per original format
    df = pd.read_csv(file_path, encoding="latin-1", usecols=[0, 1])

    # Rename columns for clarity
    df.columns = ["label", "message"]

    # Map labels to binary values
    label_map = {"ham": 0, "spam": 1}
    df["label"] = df["label"].map(label_map)

    # Drop rows with unrecognised labels or missing text before casting
    before = len(df)
    df = df.dropna(subset=["label", "message"])
    if len(df) != before:
        logger.warning("Dropped %d rows with unknown labels or missing text", before - len(df))
    df["label"] = df["label"].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Loaded dataset with {len(df)} messages")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return df


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing for spam classification.

    This function performs:
    - Lowercasing
    - Removal of extra whitespace
    - Simple punctuation handling

    For production use, consider adding:
    - Stopword removal
    - Stemming/Lemmatization
    - URL removal
    - Special character cleanup

    Parameters
    ----------
    text : str
        Raw text message

    Returns
    -------
    str
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def prepare_data(
    data_path: str = "spam.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepare and split the dataset for model training.

    This function:
    1. Loads the dataset
    2. Preprocesses text messages
    3. Splits into train/test sets (stratified on the label)

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the dataset.
    test_size : float
        Proportion of dataset to include in the test split.
    random_state : int
        Controls the shuffling applied to the data before applying the split.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test) where ``X_*`` are message Series
        and ``y_*`` are label Series (0=ham, 1=spam).
    """
    logger.info(" Preparing data for model training...")

    # Load dataset
    df = load_dataset(data_path)

    # Apply basic preprocessing to all messages
    df["message"] = df["message"].apply(preprocess_text)

    # Features and labels
    X = df["message"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Data split: {len(X_train)} training samples, "
        f"{len(X_test)} test samples"
    )

    return X_train, X_test, y_train, y_test


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Compute statistics about the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset with 'label' and 'message' columns.

    Returns
    -------
    dict
        Dictionary containing dataset statistics:
        - total_messages: int
        - spam_count: int
        - ham_count: int
        - spam_ratio: float (proportion of spam messages)
        - avg_message_length: float
        - vocab_size_estimate: int (unique words across all messages)
    """
    total_messages = len(df)
    spam_count = (df["label"] == 1).sum()
    ham_count = (df["label"] == 0).sum()
    spam_ratio = spam_count / total_messages if total_messages > 0 else 0.0

    # Calculate average message length
    avg_length = df["message"].apply(len).mean()

    # Estimate vocabulary size (unique words)
    all_words = " ".join(df["message"].astype(str)).split()
    vocab_size = len(set(all_words))

    return {
        "total_messages": total_messages,
        "spam_count": int(spam_count),
        "ham_count": int(ham_count),
        "spam_ratio": round(spam_ratio, 4),
        "avg_message_length": round(avg_length, 2),
        "vocab_size_estimate": vocab_size,
    }