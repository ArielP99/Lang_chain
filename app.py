from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle

load_dotenv()

app = Flask(__name__)

@app.route('/pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"message": "No se encontro el archivo PDF"}, 400)

    file = request.files['file']

    # Procesar el archivo PDF
    pdf_reader = PdfReader(file.stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Dividir el texto y generar embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text)

    store_name = file.filename[:-4]
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

    # Guardar el nombre del archivo para uso futuro en preguntas
    with open("current_pdf.pkl", "wb") as f:
        pickle.dump(store_name, f)

    return jsonify({"message": "PDF procesado y listo para preguntas"})

@app.route('/pregunta', methods=['POST'])
def ask_question():
    if 'question' not in request.form:
        return jsonify({"message": "No se encontro la pregunta"}, 400)

    question = request.form['question']

    # Cargar el nombre del último PDF procesado
    try:
        with open("current_pdf.pkl", "rb") as f:
            store_name = pickle.load(f)
    except FileNotFoundError:
        return jsonify({"message": "No se ha cargado ningun PDF"}, 400)

    # Cargar el VectorStore del PDF
    try:
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    except FileNotFoundError:
        return jsonify({"message": "Error al cargar los datos del pdf"}, 400)

    # Realizar la búsqueda de similitud y la respuesta de la pregunta
    docs = VectorStore.similarity_search(query=question, k=3)
    llm = OpenAI(model="gpt-3.5-turbo-instruct")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)

#debug=True, host='192.168.1.8', port=5000