from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


texts = [
    "I loved the movie",
    "This film was great",        
    "Absolutely fantastic",      
    "I hated the movie",          
    "This film was terrible",     
    "Worst experience ever"       
]

labels = [1, 1, 1, 0, 0, 0]  

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))
