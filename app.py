from flask import Flask, render_template, request
import pickle
import fitz  # PyMuPDF
import docx
import os
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model_res.pkl', 'rb'))

# Job category mapping
job_labels = {
    0: "Data Analyst",
    1: "Software Engineer",
    2: "Machine Learning Engineer",
    3: "HR Specialist",
    4: "Marketing Executive",
    5: "Business Analyst",
    6: "Project Manager"
}

# Common skills list
SKILLS = [
    "python", "java", "sql", "excel", "machine learning", "data analysis",
    "tensorflow", "pandas", "flask", "communication", "leadership",
    "project management", "teamwork", "javascript", "power bi", "tableau"
]

# Role-based expected skills
ROLE_SKILLS = {
    "Data Analyst": ["sql", "excel", "pandas", "tableau", "power bi", "data analysis"],
    "Software Engineer": ["python", "java", "flask", "javascript", "teamwork"],
    "Machine Learning Engineer": ["python", "tensorflow", "pandas", "machine learning", "data analysis"],
    "HR Specialist": ["communication", "leadership", "teamwork"],
    "Marketing Executive": ["communication", "excel", "data analysis", "teamwork"],
    "Business Analyst": ["excel", "sql", "power bi", "data analysis", "communication"],
    "Project Manager": ["leadership", "project management", "communication", "teamwork"]
}


def extract_text(file):
    """Extract text from PDF, DOCX, or TXT files."""
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text("text")
        return text
    elif filename.endswith('.docx'):
        document = docx.Document(file)
        return " ".join([p.text for p in document.paragraphs])
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8', errors='ignore')
    return ""


def extract_skills(text):
    """Keyword matching for skills."""
    text_lower = text.lower()
    return sorted(set(skill for skill in SKILLS if skill in text_lower))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('resume')
    if not file:
        return render_template('result.html', prediction="No file uploaded")

    text = extract_text(file)
    if not text.strip():
        return render_template('result.html', prediction="Invalid or empty file")

    # Model prediction
    pred_probs = model.predict_proba([text])[0]
    pred_class = np.argmax(pred_probs)
    confidence = round(pred_probs[pred_class] * 100, 2)
    job_title = job_labels.get(pred_class, "Unknown")

    # Skill matching
    found_skills = extract_skills(text)
    ideal_skills = ROLE_SKILLS.get(job_title, [])
    missing_skills = [s for s in ideal_skills if s not in found_skills]

    # Resume score
    skill_match = (len(found_skills) / len(SKILLS)) * 100
    resume_score = round((0.7 * confidence + 0.3 * skill_match) / 10, 2)

    # Suggested related roles
    suggested = [
        r for r, s in ROLE_SKILLS.items()
        if len(set(found_skills) & set(s)) >= 3 and r != job_title
    ]

    return render_template(
        'result.html',
        prediction=job_title,
        confidence=confidence,
        score=resume_score,
        skills=found_skills,
        missing=missing_skills,
        suggested=suggested
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
