 from sklearn.naive_bayes import MultinomialNB
  Meaning:
This line imports the Multinomial Naive Bayes model from the sklearn library.
This model is best suited for text classification where we count how often words appear.

..... from sklearn.feature_extraction.text import CountVectorizer
  Meaning:
Imports a tool called CountVectorizer which converts text into numbers by counting how many times each word appears.

..... from sklearn.model_selection import train_test_split
  Meaning:
This function helps split your dataset into training and testing sets, so that the model can learn on one part and be tested on the other.

......texts = [..]
  Meaning:
These are your input texts (like email or messages) that you want to classify.

......labels = [1, 0, 1, 0]
  Meaning:
These are the target labels for each text.

1 = Positive (like or good)

0 = Negative (dislike or bad)

.... vectorizer = CountVectorizer()
  Meaning:
Creates a CountVectorizer object which will be used to convert the text into numeric word count format.

.... X = vectorizer.fit_transform(texts)
  Meaning:
This line converts the text data into word count matrix (numerical format) using the vectorizeer..
