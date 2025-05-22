from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

texts = [
    "Win cash now", 
    "Hello friend, how are you?",
    "Claim your free prize",
    "Let's catch up tomorrow",
    "Click here to earn money",
    "Team meeting at 5 PM",
    "Congratulations! You've won",
    "Reminder: Your project meeting is tomorrow",
    "Get rich quick with this trick",
    "See you at the workshop"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

new_email = ["Get FREE cashback by clicking this link now"]
new_vec = vectorizer.transform(new_email).toarray()
prediction = model.predict(new_vec)

print(" Input Email:", new_email[0])
if prediction[0] == 1:
    print(" Prediction: Spam Email")
else:
    print(" Prediction: Not Spam")
