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







### Spam Message Classifier
### Objective
* The goal of this project is to detect whether a given SMS message is SPAM or NOT SPAM using a machine learning model.

* Tools and Libraries Used
* Python – Programming language

* pandas – For reading and processing the dataset

* scikit-learn – For machine learning algorithms

* CountVectorizer – To convert text messages into numbers

* Bernoulli Naive Bayes – Classification model

### Dataset Information
* The dataset contains SMS messages and their corresponding labels (spam or ham).

### Columns used:

* v1 – Label (either 'ham' or 'spam')

* v2 – The actual SMS message content

* Steps Involved
1. Import Libraries
Import all necessary libraries such as pandas, CountVectorizer, train_test_split, and BernoulliNB.

2. Load the Dataset
* The dataset is read using pandas. The correct file path is provided, and only the required columns are selected.

3. Rename Columns
* The columns v1 and v2 are renamed to label and message for better understanding.

4. Convert Labels to Numbers
* The label column is mapped as follows:

* ham becomes 0

* spam becomes 1

5. Vectorize the Messages
* Text data (messages) is converted into numerical format using CountVectorizer. This helps the machine learning model understand the text.

6. Split the Data
* The dataset is split into training and testing sets using train_test_split.

7. Train the Model
* The Bernoulli Naive Bayes model is trained using the training set.

8. User Input and Prediction
* A user can enter a message. The model will predict if it is SPAM or NOT SPAM based on what it learned from the dataset.

### Sample Output
* Input: "Congratulations! You won a prize!"
* Output: SPAM

* Input: "Hi, are we still meeting tomorrow?"
* Output: NOT SPAM

### Notes
* Make sure the dataset file path is correct.

* Use proper encoding like 'latin-1' if the file contains special characters.

* Required libraries should be installed before running the project.+







#  IMDb Sentiment Analysis using Multinomial Naive Bayes

This project classifies IMDb movie reviews as **Positive ** or **Negative ** using the **Multinomial Naive Bayes** algorithm. The reviews are taken from a local dataset (`movie.csv`) and processed using Natural Language Processing techniques.

---

##  Dataset

- **File Path:**  
  `C:\Users\C3STREAMLAND\Downloads\archive (3)\movie.csv`

- **Expected Columns:**  
  - `text`: Contains the movie review  
  - `label`: Contains the sentiment (either `positive` or `negative`)

---

##  Requirements

Install the required Python libraries using pip:

- `pandas` for data handling  
- `scikit-learn` for machine learning and evaluation

```bash
pip install pandas scikit-learn
```

# Project Steps
 * Load the CSV file containing the movie reviews.

   ##  Requirements

Install the required Python libraries using pip:

- `pandas` for data handling  
- `scikit-learn` for machine learning and evaluation

```bash
pip install pandas scikit-learn
```

# Project Steps
 * Load the CSV file containing the movie reviews.

## Clean the data:

* Remove any rows with missing labels.

* Convert text labels like positive and negative into numeric form (1 and 0).

* Convert the text data into numbers using CountVectorizer (Bag-of-Words method).

* Split the data into training and testing sets.

* Train the Naive Bayes model using the training data.

* Evaluate the model's accuracy using the test data.

* Try predicting the sentiment of a custom review to test the model.

## Sample Output
* Displays the first few rows from the dataset.

* Shows the accuracy of the trained model (for example: 87%).

* Prints whether a given review is positive or negative.



# Resume Analyzer – Explanation of Python Libraries

## Libraries Used and Their Purpose:

### 1. `sklearn.feature_extraction.text.TfidfVectorizer`
- **Purpose:** Converts text into numerical vectors.
- **Explanation:** Identifies important words in the resume and converts them into a format suitable for machine learning.

### 2. `sklearn.naive_bayes.MultinomialNB`
- **Purpose:** Used for text classification.
- **Explanation:** Predicts the job role based on the words in the resume using the Multinomial Naive Bayes algorithm.

### 3. `sklearn.model_selection.train_test_split`
- **Purpose:** Splits the dataset into training and testing parts.
- **Explanation:** Helps the model learn from one part of the data (training) and evaluate on the other (testing).

### 4. `sklearn.metrics.accuracy_score`
- **Purpose:** Checks how accurate the model’s predictions are.
- **Explanation:** Compares predicted job roles with actual labels and calculates accuracy.

### 5. `tkinter`
- **Purpose:** Creates the Graphical User Interface (GUI).
- **Explanation:** Used to build the visual part of the app like windows, buttons, and text boxes.

### 6. `tkinter.messagebox`
- **Purpose:** Shows the prediction result in a popup.
- **Explanation:** Displays the predicted job role using a message box popup.

---
## Requirements

To run this project, you need to install the following Python libraries:


bash
pip install scikit-learn
pip install pandas
pip install tk

---
## Summary Table:

| Library Name               | Explanation                                                    |
|---------------------------|----------------------------------------------------------------|
| `TfidfVectorizer`         | Converts resume text to vector form by assigning importance    |
| `MultinomialNB`           | Predicts job role based on the resume using classification     |
| `train_test_split`        | Divides data into training and testing sets                    |
| `accuracy_score`          | Evaluates if the prediction is correct                         |
| `tkinter`                 | Used to create the GUI elements like window, buttons, etc.     |
| `messagebox`              | Displays the prediction result in a popup                      |
  


