from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

texts = [
    "Win a free iPhone",        
    "Let's meet for lunch",     
    "Congratulations, you won!", 
    "Call me when you're free",
    "Get your free lottery now"  
]
labels = [1, 0, 1, 0, 1]  

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.4)


model = BernoulliNB()
model.fit(X_train, y_train)

user_input = input("Enter your message: ")

user_vector = vectorizer.transform([user_input])

prediction = model.predict(user_vector)

if prediction[0] == 1:
    print("❗ This is a SPAM message.")
else:
    print("✅ This is a NOT SPAM message.")
 


