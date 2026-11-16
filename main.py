import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


X = df['message']
y = df['label']


vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


def predict_spam(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

print(predict_spam("Congratulations! You've won a free iPhone. Click here!"))
print(predict_spam("Hey, are we still on for the meeting at 10 AM?"))