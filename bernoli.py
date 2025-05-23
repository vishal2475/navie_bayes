import pandas as pd 
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\Users\C3STREAMLAND\Downloads\SMSSpamCollection (5)", encoding='latin1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

texts = df['message'].tolist()
labels = df['label'].tolist()

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.4)

model = BernoulliNB()
model.fit(X_train, y_train)

user_input = input("Enter your message: ")

user_vector = vectorizer.transform([user_input])
prediction = model.predict(user_vector)

if prediction[0] == 1:
    print("This is a SPAM message.")
else:
    print("This is a NOT SPAM message.")
