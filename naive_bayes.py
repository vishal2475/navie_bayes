from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

texts = ["I love this movie", "I hate this movie", "This is amazing", "Terrible movie"]
labels = [1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

model = MultinomialNB()
model.fit(X_train, y_train)
print(X_train)

predictions = model.predict(X_test)
print(predictions)
