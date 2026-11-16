"""Streamlit user interface for the Spam Email Classifier.

The module provides a self-contained web app that can:

* train (or retrain) a classifier directly from the browser;
* classify a user supplied message as spam / ham;
* display evaluation metrics of the active model;
* offer curated example messages for quick testing.

Run it with::

    streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "spam.csv"
MODEL_PATH = PROJECT_ROOT / "spam_model.pkl"
RESULTS_PATH = PROJECT_ROOT / "training_results.json"

SPAM_EXAMPLES = [
    "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry",
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461",
    "Congratulations! You've won a free iPhone. Click here!",
    "Text GRAVEL to 69888 for real Dogging locations",
    "Your Mobile No. was awarded £2000 Bonus Caller Prize GUARANTEED",
]

HAM_EXAMPLES = [
    "Hey, are we still on for the meeting at 10 AM?",
    "I'm gonna be home soon and I don't want to talk about this stuff anymore tonight, k?",
    "Ok lar... Joking wif u oni...",
    "Happy birthday to me! Thanks for the wishes everyone",
    "Don't forget to pick up milk on the way home",
]


def initialize_session_state() -> None:
    """Create the session-state keys used across reruns."""
    if "metrics" not in st.session_state:
        # Reuse the metrics from the last training run so the dashboard is
        # populated on first load instead of staying blank until a retrain.
        metrics: Dict[str, float] = {}
        if RESULTS_PATH.exists():
            try:
                import json

                metrics = json.loads(RESULTS_PATH.read_text(encoding="utf-8")).get(
                    "metrics", {}
                )
            except (OSError, ValueError):
                logger.warning("Could not read %s", RESULTS_PATH, exc_info=True)
        st.session_state.metrics = metrics

    defaults: Dict[str, Any] = {
        "message_input": "",
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def load_classifier(model_path: str, mtime: float) -> Optional[Any]:
    """Load the persisted classifier, cached per file version.

    Parameters
    ----------
    model_path : str
        Absolute path to the joblib artefact.
    mtime : float
        Modification time of the artefact. Including it in the cache key
        guarantees a retrained model is picked up instead of a stale cache hit.

    Returns
    -------
    Optional[SpamClassifier]
        The fitted classifier, or ``None`` if the artefact is missing/invalid.
    """
    from src.model import SpamClassifier

    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return SpamClassifier.load_model(str(path))
    except Exception as exc:  # pragma: no cover - depends on artefact contents
        logger.exception("Failed to load model from %s", path)
        st.error(f"Could not load the saved model: {exc}")
        return None


def get_classifier() -> Optional[Any]:
    """Return the currently persisted classifier, if any."""
    if not MODEL_PATH.exists():
        return None
    return load_classifier(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def train_model_from_ui(model_type: str, use_tfidf: bool, test_size: float) -> None:
    """Train a model using the shared pipeline and refresh the UI state."""
    from src.pipeline import MODEL_SAVE_PATH, run_pipeline

    with st.spinner("Training the model. This takes a few seconds..."):
        results = run_pipeline(
            data_path=str(DATA_PATH),
            model_type=model_type,
            use_tfidf=use_tfidf,
            test_size=test_size,
            save_results=True,
        )

    # Drop the cached artefact so the freshly written file is used.
    load_classifier.clear()
    st.session_state.metrics = results["metrics"]
    st.session_state.last_result = None
    st.success(f"Model trained and saved to `{MODEL_SAVE_PATH}`.")


def classify_message(classifier: Any, text: str) -> Tuple[str, float]:
    """Classify ``text`` with ``classifier``.

    Returns
    -------
    Tuple[str, float]
        ``(label, p_spam)`` where ``p_spam`` is the probability (or a
        probability-like score for LinearSVC) of the message being spam.
    """
    return classifier.predict(text)


def render_sidebar() -> Dict[str, Any]:
    """Render training controls and example messages in the sidebar.

    Returns
    -------
    Dict[str, Any]
        The selected training configuration and current classifier.
    """
    st.sidebar.title("⚙️ Settings")

    model_type = st.sidebar.selectbox(
        "Model",
        options=["naive_bayes", "linear_svm", "logistic_regression"],
        format_func=lambda m: m.replace("_", " ").title(),
        help="Multinomial Naive Bayes is the fastest baseline.",
    )
    use_tfidf = st.sidebar.checkbox(
        "Use TF-IDF weighting", value=False, help="TF-IDF instead of raw counts."
    )
    test_size = st.sidebar.slider(
        "Test split", min_value=0.10, max_value=0.40, value=0.20, step=0.05
    )

    classifier = get_classifier()
    label = "Retrain model" if classifier is not None else "Train model"
    train_clicked = st.sidebar.button(label, type="primary", use_container_width=True)
    if train_clicked:
        train_model_from_ui(model_type, use_tfidf, test_size)
        classifier = get_classifier()

    st.sidebar.divider()
    st.sidebar.subheader("🧪 Quick examples")

    category = st.sidebar.radio(
        "Category",
        options=["Spam", "Ham (legitimate)"],
        horizontal=True,
    )
    pool = SPAM_EXAMPLES if category == "Spam" else HAM_EXAMPLES
    selected = st.sidebar.selectbox("Example message", options=pool)
    if st.sidebar.button("Use this example", use_container_width=True):
        # Bound to the text_area via key="message_input"; safe to set before
        # the widget is instantiated on the next rerun.
        st.session_state.message_input = selected
        st.rerun()

    return {"classifier": classifier}


def render_metrics(metrics: Dict[str, float]) -> None:
    """Render evaluation metrics as metric cards."""
    if not metrics:
        return

    st.subheader("📊 Model performance")
    cols = st.columns(4)
    labels = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 score"),
    ]
    for col, (key, text) in zip(cols, labels):
        if key in metrics:
            col.metric(text, f"{metrics[key]:.2%}")


def render_prediction_interface(classifier: Any) -> None:
    """Render the message input and prediction result."""
    st.subheader("Classify a message")

    text = st.text_area(
        "Message",
        key="message_input",
        height=140,
        placeholder="Paste an SMS or email here...",
    )
    clicked = st.button("Classify", type="primary")

    if not clicked:
        return
    if not text.strip():
        st.warning("Please enter a message first.")
        return

    with st.spinner("Analyzing..."):
        label, p_spam = classify_message(classifier, text)

    st.divider()

    if label.lower().startswith("spam"):
        st.error("🚨 **Spam detected**")
        st.write(f"Probability of spam: **{p_spam:.1%}**")
    else:
        st.success("✅ **Not spam (ham)**")
        st.write(f"Probability of spam: **{p_spam:.1%}**")

    with st.expander("Details"):
        st.write(f"**Prediction:** {label}")
        st.write(f"**p(spam):** {p_spam:.4f}")
        st.caption(
            "For LinearSVC the score is a sigmoid of the decision function, "
            "so treat it as a ranking score rather than a calibrated probability."
        )


def render_empty_state() -> None:
    """Explain the first-run state when no model artefact exists."""
    st.info(
        "No trained model found yet. Use **Train model** in the sidebar, "
        "or run `python -m src.pipeline` from a terminal."
    )


def main() -> None:
    """Compose and run the Streamlit application."""
    st.set_page_config(
        page_title="Spam Email Classifier",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    st.title("📧 Spam Email Classifier")
    st.caption(
        "Multinomial Naive Bayes / Linear SVM / Logistic Regression trained on "
        "the SMS Spam Collection dataset."
    )

    settings = render_sidebar()
    classifier = settings["classifier"]

    left, right = st.columns([2, 1])

    with left:
        if classifier is not None:
            render_prediction_interface(classifier)
        else:
            render_empty_state()

    with right:
        render_metrics(st.session_state.metrics)
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown(
            """
            - **Dataset:** SMS Spam Collection — 5,572 messages
            - **Features:** bag-of-words / TF-IDF (1–2 grams)
            - **Metrics:** accuracy, precision, recall, F1
            - **Training:** `python -m src.pipeline`
            """
        )


if __name__ == "__main__":
    main()
