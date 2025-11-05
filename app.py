from flask import Flask, render_template, request
import pickle
import fitz  # PyMuPDF
import docx
import os

app = Flask(__name__)

# Load model
model = pickle.load(open('model_res.pkl', 'rb'))

# --- Function to extract text from resumes ---
def extract_text(file):
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

# --- Routes ---
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

    # Predict category using model
    result = model.predict([text])[0]
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
