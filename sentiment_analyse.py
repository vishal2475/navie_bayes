import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#  Load dataset
file_path = "C:\\Users\\C3STREAMLAND\\Downloads\\archive (3)\\movie.csv"
df = pd.read_csv(file_path)

#  View first few rows
print("First few rows:")
print(df.head())
print("\nColumn names:", df.columns)

#  Use correct column names (based on your dataset)
df.columns = ['text', 'label']  # if they are already correct, skip this

#  Drop rows where label is missing (important!)
df = df.dropna(subset=['label'])

#  If label is still string like "positive"/"negative", convert:
if df['label'].dtype == 'object':
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})

#  Drop rows where label mapping failed (in case some values weren't "positive"/"negative")
df = df.dropna(subset=['label'])

#  Define features and labels
texts = df['text']
labels = df['label'].astype(int)  # Make sure labels are integers

#  Convert text to number using Bag-of-Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

#  Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

#  Train model
model = MultinomialNB()
model.fit(X_train, y_train)

#  Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\n Model Accuracy:", accuracy)

# Try a custom review
sample_review = ["This movie was terrible and boring."]
sample_vector = vectorizer.transform(sample_review)
result = model.predict(sample_vector)

print("Review Sentiment:", "Positive " if result[0] == 1 else "Negative ")
