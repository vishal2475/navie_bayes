import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset of code snippets and their language
data = {
    "code": [
        "def add(a, b): return a + b",                 # Python
        "print('Hello World')",                        # Python
        "public class HelloWorld { public static void main(String[] args) { System.out.println('Hello'); } }",  # Java
        "int sum(int a, int b) { return a + b; }",     # C++
        "console.log('Hello');",                       # JavaScript
        "let x = function(a, b) { return a + b; };",   # JavaScript
        "#include<iostream> using namespace std; int main() { cout << 'Hi'; }",  # C++
        "System.out.println('Hi')",                    # Java
        "for i in range(10): print(i)",                # Python
        "function greet() { alert('Hello'); }"         # JavaScript
    ],
    "language": [
        "Python", "Python", "Java", "C++", "JavaScript",
        "JavaScript", "C++", "Java", "Python", "JavaScript"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df['code'], df['language'], test_size=0.3, random_state=42)

# Convert code to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

