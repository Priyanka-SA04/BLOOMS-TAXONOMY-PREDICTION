# Bloom's Taxonomy Level Classification using DistilBERT

## Theoretical Foundation

### Bloom's Taxonomy Framework
The classifier predicts one of six cognitive levels from Bloom's revised taxonomy:

| Level | Name        | Description                          | Example Keywords          |
|-------|-------------|--------------------------------------|---------------------------|
| 1     | Remembering | Recall facts and basic concepts      | define, list, recall      |
| 2     | Understanding | Explain ideas or concepts           | summarize, paraphrase     |
| 3     | Applying    | Use information in new situations    | implement, solve          |
| 4     | Analyzing   | Draw connections among ideas         | compare, contrast         |
| 5     | Evaluating  | Justify a stand or decision          | critique, defend          |
| 6     | Creating    | Produce new or original work         | design, construct         |

# Bloom's Taxonomy Level Prediction

This project is a web application that predicts the Bloom's Taxonomy level of educational questions from uploaded PDF files or individual text inputs.<br>
<b>The main usecase of this project is that it makes  easier for teachers to automatically assign Bloom's levels to questions in papers instead of manually doing so.</b>

## Features

* Predicts Bloom's Taxonomy levels from text input or PDF uploads.
* Utilizes a pre-trained DistilBert model.
* This model is fine-tuned on huge dataset of questions.

## Project Structure

* `app.py`: Main Flask application.
* `saved_model/`: Contains the pre-trained TensorFlow DistilBert model files.
* `saved_tokenizer/`: Contains the tokenizer files.
* `static/`: Stores CSS for styling.
* `templates/`: HTML templates for the web interface.
* `uploads/`: (Needs to be created) Temporary storage for uploaded PDF files.

## Technologies Used

* Python, Flask
* Hugging Face Transformers (DistilBert)
* TensorFlow
* PyMuPDF (fitz)

## Setup and Installation

1.  Clone the repository.
2.  Create and activate a Python virtual environment.
3.  Install dependencies: `pip install Flask transformers tensorflow PyMuPDF`.
4.  Create an `uploads` directory: `mkdir uploads`.

## Usage

1.  Run the application: `python app.py`.
2.  Open your browser to `http://127.0.0.1:8080/`.
3.  Upload a PDF or enter a question to get a Bloom's Taxonomy prediction.

