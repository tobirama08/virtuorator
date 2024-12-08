from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for .docx files

app = Flask(__name__)  # Corrected __name__
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize HuggingFace model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_similarity(text1, text2):
    encoded_input = tokenizer([text1, text2], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    return similarity.item()

def extract_text(file_path, file_extension):
    if file_extension == '.txt':
        with open(file_path, 'r') as f:
            return f.read()
    elif file_extension == '.pdf':
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type!")

@app.route('/compare', methods=['POST'])
def compare_texts():
    try:
        audio_text = request.form.get('audio_text', '')
        file = request.files.get('file')

        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        file_text = extract_text(file_path, os.path.splitext(filename)[1].lower())

        if not audio_text or not file_text:
            return jsonify({'error': 'Missing text data'}), 400

        similarity_score = calculate_similarity(audio_text, file_text)
        return jsonify({'similarityscore': similarity_score}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':  # Corrected __name__
    app.run(debug=True, host='0.0.0.0')
