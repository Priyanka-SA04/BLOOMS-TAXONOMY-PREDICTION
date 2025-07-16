from flask import Flask, request, render_template, redirect, url_for, jsonify
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import os
import re
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Load the trained model
model_path = './saved_model'
tokenizer_path = './saved_tokenizer'
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

categories = ['Remember', 'Understand', 'Apply', 'Analyse', 'Create', 'Evaluate']

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to extract text from PDF and split into questions based on numbering
def extract_questions_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text.append(page.get_text("text"))
    
    # Join all text and split based on common numbering patterns
    full_text = "\n".join(text)
    questions = re.split(r'\n\d+\.\s', full_text)
    
    # Remove any empty strings that may have been created during the split
    questions = [q.strip() for q in questions if q.strip()]
    
    return questions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            questions = extract_questions_from_pdf(filepath)
            if not questions:
                return "No questions extracted from the PDF."

            results = []
            for question in questions:
                inputs = tokenizer(question, return_tensors="tf", truncation=True, padding=True)
                outputs = model(inputs)
                predictions = tf.nn.softmax(outputs.logits, axis=-1)
                predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
                predicted_label = categories[predicted_class]
                results.append((question, predicted_label))
            
            os.remove(filepath)  # Clean up uploaded file after processing

            return render_template('results.html', results=results)
    
    if 'question' in request.form:
        question = request.form['question']
        if question.strip() == '':
            return redirect(request.url)

        inputs = tokenizer(question, return_tensors="tf", truncation=True, padding=True)
        outputs = model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        predicted_label = categories[predicted_class]
        result = [(question, predicted_label)]

        return render_template('results.html', results=result)

    return redirect(request.url)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='127.0.0.1', port=8080, debug=True)
