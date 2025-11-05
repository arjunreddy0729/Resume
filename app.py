from flask import Flask, render_template, request
import pickle
import fitz  # PyMuPDF
import docx
import re
import os

import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model_res.pkl', 'rb'))

# Example job category mapping (update these based on your dataset)
job_labels = {
    0: "Data Analyst",
    1: "Software Engineer",
    2: "Machine Learning Engineer",
    3: "HR Specialist",
    4: "Marketing Executive",
    5: "Business Analyst",
    6: "Project Manager"
}

# Skill keywords to detect (you can expand this list)
SKILLS = [
    "python", "java", "sql", "excel", "machine learning", "data analysis",
    "tensorflow", "pandas", "flask", "communication", "leadership",
    "project management", "teamwork", "javascript", "power bi", "tableau"
]

def extract_text(file):
    """Extract text from pdf, docx, or txt file."""
    filename = file.filename
    if filename.endswith('.pdf'):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text("text")
        return text
    elif filename.endswith('.docx'):
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        return ""

def extract_skills(text):
    """Simple keyword matching for skill extraction."""
    text_lower = text.lower()
    found_skills = [skill for skill in SKILLS if skill in text_lower]
    return list(set(found_skills))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['resume']
    if not file:
        return render_template('result.html', prediction="No file uploaded")

    text = extract_text(file)

    if not text.strip():
        return render_template('result.html', prediction="Invalid or empty file")

    # Predict category and probability
    pred_probs = model.predict_proba([text])[0]
    pred_class = np.argmax(pred_probs)
    confidence = round(pred_probs[pred_class] * 100, 2)

    # Get readable label
    job_title = job_labels.get(pred_class, "Unknown Category")

    # Extract skills
    found_skills = extract_skills(text)

    return render_template(
        'result.html',
        prediction=job_title,
        confidence=confidence,
        skills=found_skills
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
