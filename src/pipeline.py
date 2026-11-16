"""
Spam Classifier Training Pipeline

Runs the complete end-to-end training process:
1. Load and preprocess data
2. Train model(s)
3. Evaluate performance
4. Save model and vectorizer
5. Generate evaluation report

Usage:
    python -m src.pipeline
    python -m src.pipeline --quick

Or as module:
    from src.pipeline import run_pipeline
    results = run_pipeline()
"""

import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np

from src.data import load_dataset, prepare_data, get_dataset_stats
from src.model import train_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths are anchored to the project root so the pipeline works regardless of
# the current working directory (e.g. when invoked from Streamlit).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "spam.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "spam_model.pkl"
RESULTS_SAVE_PATH = PROJECT_ROOT / "training_results.json"


def run_pipeline(
    data_path: str = DATA_PATH,
    model_type: str = "naive_bayes",
    use_tfidf: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    save_results: bool = True,
) -> dict:
    """
    Run the complete training pipeline.

    Parameters
    ----------
    data_path : str
        Path to the dataset CSV file.
    model_type : str
        Type of model to train. Options: naive_bayes, linear_svm, logistic_regression.
    use_tfidf : bool
        Whether to use TF-IDF weighting.
    test_size : float
        Proportion of data for testing.
    random_state : int
        Random seed for reproducibility.
    save_results : bool
        Whether to save results to JSON file.

    Returns
    -------
    dict
        Dictionary containing complete pipeline results.
    """
    logger.info("=" * 70)
    logger.info("SMS SPAM CLASSIFIER - TRAINING PIPELINE")
    logger.info("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "data_path": str(data_path),
        "model_type": model_type,
        "use_tfidf": use_tfidf,
        "test_size": test_size,
        "random_state": random_state,
    }

    try:
        # ==========================================
        # Step 1: Load and Analyze Dataset
        # ==========================================
        logger.info("\n[1/5] Loading and analyzing dataset...")

        df = load_dataset(data_path)
        dataset_stats = get_dataset_stats(df)

        results["dataset_stats"] = {
            "total_messages": dataset_stats["total_messages"],
            "spam_count": dataset_stats["spam_count"],
            "ham_count": dataset_stats["ham_count"],
            "spam_ratio": dataset_stats["spam_ratio"],
            "avg_message_length": dataset_stats["avg_message_length"],
            "vocab_size_estimate": dataset_stats["vocab_size_estimate"],
        }

        logger.info(f"   Total messages: {dataset_stats['total_messages']}")
        logger.info(f"   Spam messages: {dataset_stats['spam_count']}")
        logger.info(f"   Ham messages: {dataset_stats['ham_count']}")
        logger.info(f"   Spam ratio: {dataset_stats['spam_ratio']:.2%}")
        logger.info(f"   Avg message length: {dataset_stats['avg_message_length']:.1f} chars")

        # ==========================================
        # Step 2: Prepare Data
        # ==========================================
        logger.info("\n[2/5] Preparing data (splitting and vectorizing)...")

        X_train, X_test, y_train, y_test = prepare_data(
            data_path=data_path,
            test_size=test_size,
            random_state=random_state,
        )

        results["data_split"] = {
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "spam_ratio_train": float((y_train == 1).mean()),
            "spam_ratio_test": float((y_test == 1).mean()),
        }

        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Test samples: {len(X_test)}")

        # ==========================================
        # Step 3: Train Model
        # ==========================================
        logger.info(f"\n[3/5] Training {model_type} model{' with TF-IDF' if use_tfidf else ''}...")

        classifier, metrics = train_model(
            X_train, y_train, X_test, y_test,
            model_type=model_type,
            use_tfidf=use_tfidf,
        )

        results["model_type"] = model_type
        results["use_tfidf"] = use_tfidf
        results["metrics"] = metrics

        logger.info(f"   Model: {model_type}")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1']:.4f}")

        # ==========================================
        # Step 4: Cross-Validation
        # ==========================================
        logger.info("\n[4/5] Performing cross-validation...")

        cv_results = classifier.cross_validate(X_train, y_train, cv=5)
        results["cross_validation"] = {
            "cv_mean_accuracy": cv_results["accuracy"]["mean"],
            "cv_std_accuracy": cv_results["accuracy"]["std"],
            "cv_mean_f1": cv_results["f1"]["mean"],
            "cv_std_f1": cv_results["f1"]["std"],
        }

        logger.info(f"   CV Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        logger.info(f"   CV F1: {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")

        # ==========================================
        # Step 5: Save Model
        # ==========================================
        logger.info("\n[5/5] Saving model...")

        classifier.save_model(str(MODEL_SAVE_PATH))
        results["model_path"] = str(MODEL_SAVE_PATH)

        # Save feature names if using TF-IDF
        if use_tfidf:
            vectorizer = classifier.pipeline.named_steps["vectorizer"]
            feature_names = vectorizer.get_feature_names_out()[:20]  # Top 20 features
            results["top_features"] = feature_names.tolist()
            logger.info(f"   Top {len(feature_names)} features: {feature_names}")

        logger.info(f"   Model saved to: {MODEL_SAVE_PATH}")

        # ==========================================
        # Final Summary
        # ==========================================
        if save_results:
            _save_results(results)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE - All steps finished successfully")
        logger.info("=" * 70)

        return results

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def _save_results(results: dict) -> None:
    """
    Save pipeline results to JSON file.

    Parameters
    ----------
    results : dict
        Dictionary containing all pipeline results.
    """
    results_file = Path(RESULTS_SAVE_PATH)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types to Python native types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    serializable_results = convert_types(results)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")


def quick_train():
    """
    Quick training function for rapid prototyping.

    Trains a Naive Bayes model with default parameters
    and prints a summary of results.
    """
    logger.info("Quick training mode...")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data_path=str(DATA_PATH))
    stats = get_dataset_stats(load_dataset(str(DATA_PATH)))

    # Train Naive Bayes model (fast, no TF-IDF needed for baseline)
    classifier, metrics = train_model(
        X_train, y_train, X_test, y_test,
        model_type="naive_bayes",
        use_tfidf=False,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("QUICK TRAINING SUMMARY")
    print("=" * 60)
    print(
        f"Dataset: {stats['total_messages']} messages "
        f"({stats['ham_count']} ham, {stats['spam_count']} spam)"
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("\nModel: Multinomial Naive Bayes")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1']:.2%}")
    print("=" * 60 + "\n")

    # Save model
    classifier.save_model(str(MODEL_SAVE_PATH))
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    import sys

    print("\nSMS Spam Classifier - Training Pipeline")
    print("=" * 50)

    # Check if user wants quick train or full pipeline
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_train()
    else:
        # Run full pipeline with default parameters
        results = run_pipeline()

        # Print final summary
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Model: {results['model_type']}")
        print(f"Using TF-IDF: {results['use_tfidf']}")
        print(f"Accuracy:    {results['metrics']['accuracy']:.2%}")
        print(f"Precision:   {results['metrics']['precision']:.2%}")
        print(f"Recall:      {results['metrics']['recall']:.2%}")
        print(f"F1 Score:    {results['metrics']['f1']:.2%}")
        print(f"CV Accuracy: {results['cross_validation']['cv_mean_accuracy']:.2%} ± "
              f"{results['cross_validation']['cv_std_accuracy']:.2%}")
        print(f"Dataset: {results['dataset_stats']['total_messages']} messages")
        print("=" * 50 + "\n")