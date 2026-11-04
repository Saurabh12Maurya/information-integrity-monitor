from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "training_data.csv"
MODEL_PATH = ROOT / "review_priority_model.joblib"

data = pd.read_csv(DATA_PATH)
X = data["text"].astype(str)
y = data["label"].astype(str)

# This is a demonstration model trained on synthetic examples.
# The labels represent triage priority, not whether a claim is true or false.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("DEMONSTRATION MODEL EVALUATION")
print("--------------------------------")
print(f"Test examples: {len(X_test)}")
print(f"Accuracy on this small synthetic test split: {accuracy_score(y_test, predictions):.2%}")
print()
print(classification_report(y_test, predictions, zero_division=0))
print("Important: these scores are only for the synthetic demonstration dataset.")
print("They do not establish real-world accuracy or whether any claim is true.")

joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH.name}")
