# Text Classification Projects with Naive Bayes

This project shows how to use machine learning to check if a message is spam or not, or if a review is positive or negative.

---

## Project 1: Spam Detection using Multinomial Naive Bayes

### What it does:
Checks how many times each word appears in the message and uses that to decide if it's spam.

###  Tools Used:
- `MultinomialNB`: For checking word counts.
- `CountVectorizer`: Changes text into numbers by counting words.
- `train_test_split`: Splits the data into training and testing.

### Steps:
1. Messages are stored in a list.
2. Each message is labeled (1 = spam, 0 = not spam).
3. Messages are changed into numbers (word counts).
4. Model is trained on this data.
5. User enters a new message to check if it is spam or not.

---

## Project 2: Spam Detection using Bernoulli Naive Bayes

### What it does:
Checks **if a word is present** (yes or no), not how many times it appears.

###  Tools Used:
- `BernoulliNB`: For yes/no word checking.
- `CountVectorizer(binary=True)`: Changes text to 1s and 0s.
- `train_test_split`: Splits the data.

### Steps:
1. Convert messages to binary (1 = word is there, 0 = not there).
2. Train the model.
3. Check if a new message is spam or not.

---

## Project 3: Spam Detection using Gaussian Naive Bayes with TF-IDF

### What it does:
Uses word importance (not just count) to find out if a message is spam.

###  Tools Used:
- `TfidfVectorizer`: Gives importance to words.
- `GaussianNB`: Works well with continuous numbers like TF-IDF.
- `accuracy_score`: Checks how correct the model is.

### Steps:
1. Messages are turned into TF-IDF numbers.
2. Data is split into training and testing.
3. Model is trained.
4. New message is checked for spam or not.



---

## Project 4: Sentiment Analysis using Multinomial Naive Bayes

###  What it does:
Checks if a movie review is **positive** or **negative**.

###  Tools Used:
- `TfidfVectorizer`: Converts text into word importance numbers.
- `MultinomialNB`: Best for TF-IDF values.
- `accuracy_score`: Shows how good the model is.

### Steps:
1. Reviews are stored in a list.
2. Labels: 1 = positive, 0 = negative.
3. Text is turned into numbers (TF-IDF).
4. Model is trained.
5. Shows predictions and accuracy.

### Example:

Predicted: [1, 0]
Actual: [1, 0]
Accuracy: 100%
