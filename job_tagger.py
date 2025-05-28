from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Step 1: Manually defined dataset
texts = [
    "Experienced Python developer with strong skills in Django and REST APIs.",
    "Data scientist with expertise in machine learning, deep learning, and data visualization.",
    "Frontend developer with experience in React, JavaScript, and HTML/CSS.",
    "Experienced in handling HR responsibilities including recruitment and payroll management.",
    "Skilled Java backend developer with experience in Spring Boot and Microservices.",
    "Worked on content creation, copywriting, and SEO optimization for websites.",
    "Cloud engineer experienced in AWS, DevOps practices, and container orchestration.",
    "Strong project management skills, Agile methodologies, and client communication.",
    "Experience in customer support and troubleshooting technical issues via chat.",
    "UX/UI designer with a focus on user-centric design and prototyping."
]

labels = [
    "Python Developer",
    "Data Scientist",
    "Frontend Developer",
    "HR Executive",
    "Java Developer",
    "Content Writer",
    "Cloud Engineer",
    "Project Manager",
    "Customer Support",
    "UX Designer"
]

# Step 2: Preprocessing
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(texts)
y = labels

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Step 4: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Evaluate (Optional)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Step 6: GUI
window = tk.Tk()
window.title("Resume Analyzer")
window.geometry("500x400")

label = tk.Label(window, text="Paste Resume Text Below:")
label.pack(pady=10)

text_box = tk.Text(window, height=10, width=50)
text_box.pack()

def analyze_resume():
    input_text = text_box.get("1.0", tk.END)
    input_vector = vectorizer.transform([input_text])
    result = model.predict(input_vector)
    messagebox.showinfo("Prediction", f"Predicted Job Role: {result[0]}")

button = tk.Button(window, text="Analyze", command=analyze_resume)
button.pack(pady=20)

window.mainloop()