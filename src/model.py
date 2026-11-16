"""
Model module for spam email classifier.

This module handles:
- Text vectorization using CountVectorizer or TF-IDF
- Model training (Naive Bayes, SVM, Logistic Regression)
- Model evaluation with appropriate metrics
- Model persistence (save/load)

Models Supported:
    - Multinomial Naive Bayes (baseline, fast, works well with text)
    - Linear SVM (high performance, slower training)
    - Logistic Regression (good baseline, probabilistic outputs)
"""

import logging
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Set up logging
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "naive_bayes": {
        "classifier": "MultinomialNB",
        "params": {},
        "description": "Multinomial Naive Bayes - fast baseline for text classification",
    },
    "linear_svm": {
        "classifier": "LinearSVC",
        "params": {"C": 1.0},
        "description": "Linear SVM - high performance for text classification",
    },
    "logistic_regression": {
        "classifier": "LogisticRegression",
        "params": {"max_iter": 1000},
        "description": "Logistic Regression - probabilistic text classification",
    },
}

# Metrics to compute for evaluation
EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1"]


class SpamClassifier:
    """
    A comprehensive spam classification pipeline.

    This class encapsulates the entire machine learning pipeline:
    - Text vectorization (CountVectorizer or TF-IDF)
    - Model training and evaluation
    - Model persistence
    - Prediction capabilities

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted ML pipeline containing vectorizer and classifier
    model_name : str
        Name of the trained model
    metrics : Dict[str, float]
        Evaluation metrics on test data
    """

    def __init__(
        self,
        model_type: str = "naive_bayes",
        use_tfidf: bool = False,
        vectorizer_max_features: Optional[int] = None,
    ):
        """
        Initialize the spam classifier.

        Parameters
        ----------
        model_type : str
            Type of model to use. Options: 'naive_bayes', 'linear_svm',
            'logistic_regression'.
        use_tfidf : bool
            Whether to use TF-IDF weighting instead of raw counts.
        vectorizer_max_features : int or None
            Maximum number of features for the vectorizer. None means use all.
        """
        self.model_type = model_type
        self.use_tfidf = use_tfidf
        self.vectorizer_max_features = vectorizer_max_features
        self.pipeline: Optional[Any] = None
        self.metrics: Dict[str, float] = {}
        self.model_name = ""

        # Validate model type
        if model_type not in DEFAULT_MODEL_CONFIG:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Choose from: {list(DEFAULT_MODEL_CONFIG.keys())}"
            )

    def _build_pipeline(self) -> Pipeline:
        """
        Build the ML pipeline based on configuration.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Configured ML pipeline with vectorizer and classifier.
        """
        model_config = DEFAULT_MODEL_CONFIG[self.model_type]

        classifier_map = {
            "MultinomialNB": MultinomialNB,
            "LinearSVC": LinearSVC,
            "LogisticRegression": LogisticRegression,
        }
        classifier_cls = classifier_map[model_config["classifier"]]

        vectorizer_cls = TfidfVectorizer if self.use_tfidf else CountVectorizer
        vectorizer = vectorizer_cls(
            max_features=self.vectorizer_max_features,
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )

        return Pipeline(
            [
                ("vectorizer", vectorizer),
                ("classifier", classifier_cls(**model_config["params"])),
            ]
        )

    def train(self, X_train: pd.Series, y_train: pd.Series) -> "SpamClassifier":
        """
        Train the model on training data.

        Parameters
        ----------
        X_train : pd.Series
            Training messages.
        y_train : pd.Series
            Training labels (0=ham, 1=spam).

        Returns
        -------
        SpamClassifier
            Self instance for method chaining.
        """
        logger.info(f"Training {self.model_type} model...")

        self.pipeline = self._build_pipeline()

        # Train the model
        self.pipeline.fit(X_train, y_train)

        self.model_name = f"{self.model_type}_{'tfidf' if self.use_tfidf else 'count'}"

        logger.info(f"Model {self.model_name} trained successfully")

        return self

    def evaluate(
        self, X_test: pd.Series, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Computes the following metrics:
        - Accuracy
        - Precision
        - Recall
        - F1-score

        Parameters
        ----------
        X_test : pd.Series
            Test messages.
        y_test : pd.Series
            Test labels (0=ham, 1=spam).

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet. Call 'train' first.")

        logger.info("Evaluating model performance...")

        # Make predictions
        y_pred = self.pipeline.predict(X_test)

        # Calculate metrics
        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }

        # Log detailed classification report
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

        return self.metrics

    def cross_validate(
        self, X: pd.Series, y: pd.Series, cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation to assess model robustness.

        Parameters
        ----------
        X : pd.Series
            All messages.
        y : pd.Series
            All labels.
        cv : int
            Number of cross-validation folds.

        Returns
        -------
        Dict[str, float]
            Average cross-validation metrics.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet.")

        logger.info(f"Performing {cv}-fold cross-validation...")

        scoring = {"accuracy": "accuracy", "f1": "f1_macro"}

        cv_results = {}
        for metric in scoring.keys():
            scores = cross_val_score(
                self.pipeline, X, y, scoring=scoring[metric], cv=cv
            )
            cv_results[metric] = {
                "mean": round(scores.mean(), 4),
                "std": round(scores.std(), 4),
                "scores": scores.tolist(),
            }

        logger.info(f"Cross-validation results: {cv_results}")
        return cv_results

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict whether a given text is spam or not.

        Parameters
        ----------
        text : str
            Input text message to classify.

        Returns
        -------
        Tuple[str, float]
            (prediction, confidence) where:
            - prediction: "spam" or "ham"
            - confidence: float between 0 and 1 (probability of spam)
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # Vectorize once and reuse the matrix for both prediction and scoring
        # (the previous implementation transformed the text twice).
        vectorizer = self.pipeline.named_steps["vectorizer"]
        classifier = self.pipeline.named_steps["classifier"]
        features = vectorizer.transform([text])

        predicted = classifier.predict(features)[0]
        classes = list(getattr(classifier, "classes_", [0, 1]))
        spam_index = classes.index(1) if 1 in classes else len(classes) - 1

        if hasattr(classifier, "predict_proba"):
            confidence = float(classifier.predict_proba(features)[0][spam_index])
        elif hasattr(classifier, "decision_function"):
            decision = float(np.ravel(classifier.decision_function(features))[0])
            confidence = float(1 / (1 + np.exp(-decision)))
        else:  # pragma: no cover - all supported models expose a score
            confidence = 0.5
            logger.warning("Classifier exposes no score; defaulting to 0.5")

        if isinstance(predicted, str):
            is_spam = predicted.strip().lower() == "spam"
        else:
            is_spam = int(predicted) == 1

        prediction = "Spam" if is_spam else "Not Spam (Ham)"

        return prediction, confidence

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        model_path : str
            Path where the model will be saved (joblib format).
        """
        if self.pipeline is None:
            raise ValueError("No trained model to save.")

        model_file = Path(model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, model_file)
        logger.info(f"Model saved to {model_file}")

    @classmethod
    def load_model(cls, model_path: str) -> "SpamClassifier":
        """
        Load a saved model from disk.

        Parameters
        ----------
        model_path : str
            Path to the saved model file.

        Returns
        -------
        SpamClassifier
            Instance with loaded model.
        """
        instance = cls()
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        instance.pipeline = joblib.load(model_file)
        instance.model_name = f"loaded_{model_file.stem}"
        logger.info(f"Model loaded from {model_file}")

        return instance

    @staticmethod
    def get_model_info() -> Dict[str, Dict]:
        """
        Get information about available model types.

        Returns
        -------
        Dict[str, Dict]
            Dictionary containing model configuration for each type.
        """
        return DEFAULT_MODEL_CONFIG.copy()


def train_model(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    model_type: str = "naive_bayes",
    use_tfidf: bool = False,
) -> Tuple[SpamClassifier, Dict[str, float]]:
    """
    Convenience function to train and evaluate a model.

    Parameters
    ----------
    X_train : pd.Series
        Training messages.
    y_train : pd.Series
        Training labels.
    X_test : pd.Series
        Test messages.
    y_test : pd.Series
        Test labels.
    model_type : str
        Type of model to train.
    use_tfidf : bool
        Whether to use TF-IDF weighting.

    Returns
    -------
    Tuple[SpamClassifier, Dict[str, float]]
        Trained model and evaluation metrics.
    """
    logger.info(f"Starting training with {model_type} model")

    # Create and train model
    classifier = SpamClassifier(model_type=model_type, use_tfidf=use_tfidf)
    classifier.train(X_train, y_train)

    # Evaluate model
    metrics = classifier.evaluate(X_test, y_test)

    return classifier, metrics


def compare_models(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    model_types: Optional[list] = None,
    use_tfidf: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models and return their metrics.

    Parameters
    ----------
    X_train : pd.Series
        Training messages.
    y_train : pd.Series
        Training labels.
    X_test : pd.Series
        Test messages.
    y_test : pd.Series
        Test labels.
    model_types : list or None
        List of model types to compare. If None, uses all available models.
    use_tfidf : bool
        Whether to use TF-IDF weighting for all models.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping model names to their evaluation metrics.
    """
    if model_types is None:
        model_types = list(DEFAULT_MODEL_CONFIG.keys())

    results = {}

    for model_type in model_types:
        try:
            classifier, metrics = train_model(
                X_train, y_train, X_test, y_test,
                model_type=model_type, use_tfidf=use_tfidf
            )
            results[model_type] = metrics
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    return results