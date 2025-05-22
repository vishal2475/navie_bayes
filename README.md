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





Summary of the Spam Detection Code using Bernoulli Naive Bayes
This Python script is a simple spam classifier built using the Bernoulli Naive Bayes algorithm from Scikit-learn. It works as follows:

Text Data: A list of sample messages is provided, labeled as spam (1) or not spam (0).

Vectorization: The messages are converted into binary vectors (1 if a word is present, 0 if not) using CountVectorizer(binary=True).

Train-Test Split: The dataset is split into training and testing sets.

Model Training: A BernoulliNB model is trained on the binary vectors.

User Input: The user can enter a new message.

Prediction: The model predicts whether the message is spam or not.

Output: The result is printed as either “SPAM” or “NOT SPAM”.

This approach is good for situations where the presence or absence of certain keywords is more important than how often they appear.








Spam Detection using TF-IDF and Gaussian Naive Bayes – Text Summary
Purpose:
This code is used to detect whether a given email/message is spam or not spam using machine learning.

Libraries Used:

TfidfVectorizer: Converts text into numerical TF-IDF values (word importance).

GaussianNB: A classifier used for continuous data (like TF-IDF values).

train_test_split: Splits the data into training and testing sets.

accuracy_score: Measures the prediction accuracy.

Dataset:

texts: A list of 10 messages (spam and non-spam).

labels: A list of corresponding labels (1 = spam, 0 = not spam).

Text Preprocessing:

TF-IDF vectorization is used to convert the messages into numeric feature vectors.

These vectors are converted into arrays for use in the model.

Model Training:

The dataset is split into training and test sets (70% train, 30% test).

GaussianNB is trained on the training set.

Prediction and Evaluation:

The model predicts labels for test data.

Accuracy is printed to show how well the model performs.

New Email Prediction:

A new message is input and transformed using the same vectorizer.

The trained model predicts whether the message is spam or not spam.

The prediction result is printed.

Output Example:

kotlin
Copy
Edit
Input Email: Get FREE cashback by clicking this link now
Prediction: Spam Email










Sentiment Analysis using TF-IDF and Multinomial Naive Bayes – Text Summary
Purpose:
This code is designed to classify text messages (movie reviews) as positive or negative using machine learning.

Libraries Used:

TfidfVectorizer: Converts text data into numerical values based on word importance (TF-IDF).

MultinomialNB: Naive Bayes classifier suitable for discrete features like word counts or TF-IDF values.

train_test_split: Splits the dataset into training and testing sets.

accuracy_score: Evaluates how accurate the model’s predictions are.

Dataset:

texts: A list of 6 movie reviews.

labels: A list of sentiment values:

1 = positive review

0 = negative review

Text Preprocessing:

TfidfVectorizer is used to convert the reviews into numerical vectors based on word frequency and importance.

These vectors are used as features for the machine learning model.

Train-Test Split:

The dataset is split into 67% training data and 33% testing data.

Model Training:

The MultinomialNB classifier is trained on the training data to learn how to classify the reviews.

Prediction & Evaluation:

The model makes predictions on the test set.

Actual labels and predicted labels are printed.

The model's accuracy is displayed using accuracy_score.

Example Output (may vary each time due to random split):

makefile
Copy
Edit
Predictions: [1 0]
Actual: [1 0]
Accuracy: 1.0

