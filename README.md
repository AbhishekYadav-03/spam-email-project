# Spam Email Classifier

A complete machine learning project for classifying SMS/email messages as spam or legitimate (ham).

## Project Structure

```
spam-email-classifier/
├── app.py                      # Streamlit entry point (thin wrapper)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore
├── spam.csv                    # Dataset (SMS Spam Collection)
└── src/
    ├── __init__.py
    ├── data.py                 # Data loading and preprocessing
    ├── model.py                # Model training and evaluation
    ├── pipeline.py             # Training pipeline (CLI)
    └── ui.py                   # Streamlit user interface
```

## Dataset

The project uses the **SMS Spam Collection Dataset** containing:
- **5,572 messages** total
- **4,825 ham** (legitimate) messages
- **747 spam** messages
- Source: UCI Machine Learning Repository / Kamishima et al. (2009)

The dataset is included as `spam.csv` with columns:
- `v1`: Label (ham/spam)
- `v2`: Message text

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd spam-email-classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install Kaggle API for dataset download:**
   ```bash
   pip install kaggle
   # Configure API token at ~/.kaggle/kaggle.json
   ```

## Usage

### 1. Train a Model

Run the training pipeline from the project root:

```bash
python -m src.pipeline
```

Quick baseline only (Naive Bayes):

```bash
python -m src.pipeline --quick
```

This writes `spam_model.pkl` and `training_results.json` to the project root.

### 2. Launch the Streamlit UI

```bash
streamlit run app.py
```

The web interface offers:

- **Message classification** with a spam probability
- **Model selection**: Naive Bayes, Linear SVM, or Logistic Regression
- **TF-IDF weighting** toggle
- **Train / retrain** directly from the sidebar
- **Example messages** for quick testing
- **Model performance metrics**: accuracy, precision, recall, F1-score

### 3. Compare Multiple Models

```python
from src.data import prepare_data
from src.model import compare_models

X_train, X_test, y_train, y_test = prepare_data("spam.csv")
results = compare_models(X_train, y_train, X_test, y_test)
print(results)
```

## Available Models

| Model | Type | Best For |
|-------|------|----------|
| **Multinomial Naive Bayes** | Probabilistic | Fast baseline, works well with high-dimensional text |
| **Linear SVM** | Margin-based | Higher performance on text classification, slower training |
| **Logistic Regression** | Probabilistic | Probabilistic outputs, good baseline with interpretability |

## Pipeline Workflow

1. **Data Loading**: Loads and preprocesses the SMS Spam Collection dataset
2. **Text Vectorization**: Converts text to numerical features using CountVectorizer or TfidfVectorizer
3. **Train/Test Split**: 80/20 split with stratification to maintain class balance
4. **Model Training**: Fits the selected classifier on training data
5. **Evaluation**: Computes accuracy, precision, recall, and F1-score
6. **Cross-Validation**: 5-fold cross-validation for robustness assessment
7. **Model Persistence**: Saves trained pipeline using joblib

## Configuration

### Hyperparameters

Key parameters that can be adjusted:

- `max_features`: Limit the number of features (None = use all)
- `stop_words`: Remove English stop words (`"english"`)
- `ngram_range`: Consider unigrams and bigrams `(1, 2)`
- `min_df`: Ignore terms that appear in < 2 documents
- `max_df`: Ignore terms that appear in > 95% of documents

### Model Configuration

Configured in `src/model.py` `DEFAULT_MODEL_CONFIG` dictionary. Each model type has:
- Specific classifier class
- Default hyperparameters
- Human-readable description

## Examples

### Classifying a Message via Python

```python
from src.model import SpamClassifier

# Load trained model
classifier = SpamClassifier.load_model("spam_model.pkl")

# Predict
text = "Congratulations! You've won a free iPhone. Click here!"
prediction, confidence = classifier.predict(text)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

### Streamlit App Features

| Feature | Description |
|---------|-------------|
| Message Input | Type or paste any message for classification |
| Example Library | Predefined spam and ham examples |
| Model Settings | Switch between model types and TF-IDF |
| Performance Metrics | View accuracy, precision, recall, F1 |
| Model Persistence | Load/save trained models |

## Results Interpretation

### Evaluation Metrics

- **Accuracy**: Overall correctness (TP + TN / Total)
- **Precision**: Of predicted spam, how many are actually spam (TP / (TP + FP))
  - *Important for spam: High precision means few false alarms*
- **Recall**: Of actual spam, how many we detected (TP / (TP + FN))
  - *Important for spam: High recall means catching most spam*
- **F1 Score**: Harmonic mean of precision and recall

### Confusion Matrix Context

For spam classification, typically:
- **High Recall** is prioritized (catch more spam, even if some ham gets flagged)
- **Precision** is also important (avoid flagging legitimate messages as spam)

## File Descriptions

| File | Purpose |
|------|---------|
| `spam.csv` | SMS Spam Collection dataset |
| `requirements.txt` | Python dependencies |
| `src/data.py` | Data loading, preprocessing, statistics |
| `src/model.py` | Model training, evaluation, persistence |
| `src/ui.py` | Streamlit user interface |
| `src/pipeline.py` | End-to-end training pipeline (CLI) |
| `app.py` | Streamlit entry point that launches `src/ui.py` |
| `README.md` | Project documentation (this file) |

## Future Enhancements

- [ ] Deep learning models (CNN, RNN/LSTM for sequence modeling)
- [ ] Advanced text preprocessing (lemmatization, URL/mention removal)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Batch classification of multiple messages
- [ ] Integration with email clients
- [ ] Real-time spam filtering dashboard
- [ ] Mobile-responsive Streamlit theme

## License

This project is open source and available for modification and distribution.

---

**Built with**: Python, scikit-learn, pandas, numpy, Streamlit, joblib