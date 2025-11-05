from flask import Flask, render_template, request, send_file
import pickle
import fitz  # PyMuPDF
import docx
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open('model_res.pkl', 'rb'))

# Example job category mapping
job_labels = {
    0: "Data Analyst",
    1: "Software Engineer",
    2: "Machine Learning Engineer",
    3: "HR Specialist",
    4: "Marketing Executive",
    5: "Business Analyst",
    6: "Project Manager"
}

# Skills to detect
SKILLS = [
    "python", "java", "sql", "excel", "machine learning", "data analysis",
    "tensorflow", "pandas", "flask", "communication", "leadership",
    "project management", "teamwork", "javascript", "power bi", "tableau"
]


# -----------------------------
# Utility Functions
# -----------------------------
def extract_text(file):
    """Extract text from pdf, docx, or txt file."""
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text("text")
        return text
    elif filename.endswith('.docx'):
        docx_file = docx.Document(file)
        return " ".join([para.text for para in docx_file.paragraphs])
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        return ""


def extract_skills(text):
    """Simple keyword matching for skill extraction."""
    text_lower = text.lower()
    found_skills = [skill for skill in SKILLS if skill in text_lower]
    return list(set(found_skills))


def job_match_score(resume_text, jd_text):
    """Compute similarity between resume text and job description."""
    if not jd_text.strip():
        return None
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['resume']
    jd_text = request.form.get('jobdesc', '')

    if not file:
        return render_template('result.html', prediction="No file uploaded")

    text = extract_text(file)
    if not text.strip():
        return render_template('result.html', prediction="Invalid or empty file")

    # Predict category
    pred_probs = model.predict_proba([text])[0]
    pred_class = np.argmax(pred_probs)
    confidence = round(pred_probs[pred_class] * 100, 2)
    job_title = job_labels.get(pred_class, "Unknown Category")

    # Extract skills + match score
    found_skills = extract_skills(text)
    match_score = job_match_score(text, jd_text)

    return render_template(
        'result.html',
        prediction=job_title,
        confidence=confidence,
        skills=found_skills,
        match_score=match_score
    )


@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download a PDF report."""
    data = request.form
    prediction = data.get('prediction')
    confidence = data.get('confidence')
    skills = data.get('skills')
    match_score = data.get('match_score')

    filename = "resume_analysis_report.pdf"
    c = canvas.Canvas(filename, pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Resume Analysis Report")

    c.setFont("Helvetica", 12)
    y = 770
    lines = [
        f"Predicted Role: {prediction}",
        f"Confidence: {confidence}%",
        f"Match Score: {match_score if match_score else 'N/A'}%",
        f"Skills Found: {skills}"
    ]
    for line in lines:
        c.drawString(100, y, line)
        y -= 20

    c.save()
    return send_file(filename, as_attachment=True)


# -----------------------------
# Run the app (Render uses PORT env)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
