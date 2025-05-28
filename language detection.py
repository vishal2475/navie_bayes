import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\\Users\\C3STREAMLAND\\Downloads\\german_english_dataset (1).csv")


texts = pd.concat([data['German'], data['English']], ignore_index=True)

labels = ['german'] * len(data['German']) + ['english'] * len(data['English'])



vectorizer = TfidfVectorizer()         


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)


model = MultinomialNB()
model.fit(X_train, Y_train)

prediction = model.predict(X_test)


print("Predictions:", prediction)
print("Actual:     ", Y_test)
print("Accuracy:", accuracy_score(Y_test, prediction))

