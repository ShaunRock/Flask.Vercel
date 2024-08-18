from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid
import os
import json

app = Flask(__name__)
CORS(app)

GENAI_API_KEY = "AIzaSyB2-oXWZf03YfxRmKMdIgqTMpUgIeqsxoE"

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

# In-memory storage for indexed PDFs
pdf_data = {}

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    pdf_file = request.files['file']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    raw_text = get_pdf_text(pdf_file)
    text_chunks = get_text_chunks(raw_text)
    index, chunks = get_vector_store(text_chunks)
    
    pdf_id = str(uuid.uuid4())
    pdf_data[pdf_id] = {
        "index": index,
        "chunks": chunks
    }
    
    return jsonify({"pdfId": pdf_id, "message": "Processing complete"}), 200


@app.route('/ask_chat', methods=['POST'])
def ask_chat():
    try:
        data = request.json
        pdf_id = data['pdfId']
        user_question = data['question']
        
        if pdf_id not in pdf_data:
            return jsonify({"error": "Invalid PDF ID"}), 400
        
        index = pdf_data[pdf_id]["index"]
        chunks = pdf_data[pdf_id]["chunks"]
        
        response = user_input(user_question, index, chunks)
        print(f"Chat Mode: Question: {user_question}, Response: {response}")  # Log chat response
        return jsonify({"response": response}), 200
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500


@app.route('/ask_quiz', methods=['POST'])
def ask_quiz():
    try:
        data = request.json
        pdf_id = data['pdfId']
        user_question = data['question']
        
        if pdf_id not in pdf_data:
            return jsonify({"error": "Invalid PDF ID"}), 400
        
        index = pdf_data[pdf_id]["index"]
        chunks = pdf_data[pdf_id]["chunks"]
        
        quiz_data = generate_quiz(user_question, index, chunks)
        print(f"Quiz Mode: Question: {user_question}, Quiz Data: {quiz_data}")  # Log quiz response
        return jsonify({"response": quiz_data }), 200
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500


def user_input(user_question, index, chunks):
    vectorizer = TfidfVectorizer()
    chunks_text = [chunk for chunk, _ in chunks]
    vectorizer.fit(chunks_text)
    question_embedding = vectorizer.transform([user_question]).toarray()[0]

    distances, indices = index.search(np.array([question_embedding]), k=5)
    similar_texts = [chunks[idx][0] for idx in indices[0]]

    documents = [Document(page_content=text, metadata={}) for text in similar_texts]
    context = "\n".join(similar_texts)

    chain = get_conversational_chain()
    response = chain({"input_documents": documents, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


def generate_quiz(user_question, index, chunks):
    return user_input(user_question, index, chunks)


def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    chunk_size = 1000
    overlap = 200
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

def get_vector_store(text_chunks):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text_chunks)
    embeddings = vectorizer.transform(text_chunks).toarray()

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, [(chunk, embedding) for chunk, embedding in zip(text_chunks, embeddings)]

def get_conversational_chain():
    prompt_template = """Answer the question. Context:\n {context}?\n
    Question: \n{question}\n
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GENAI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


if __name__ == '__main__':
    app.run(debug=True)
