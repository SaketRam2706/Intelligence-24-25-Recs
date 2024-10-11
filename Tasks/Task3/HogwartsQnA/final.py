from flask import Flask, request, jsonify
from flask_cors import CORS 
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.embeddings import SentenceTransformerEmbeddings
import os


app = Flask(__name__)
CORS(app)

API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
CHROMA_PATH = "chroma"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", show_progress=True)

db = Chroma(
    persist_directory=CHROMA_PATH,
    collection_name="wec-rag",
    embedding_function=embeddings
)

template = """<s> [INST]
Answer the question based on the context below. If you can't 
answer the question because the context isn't relevant reply that you dont have the answer based on the given context.

Context: {context}

Question: {question}

[/INST]
"""

prompt_template = PromptTemplate.from_template(template)

@app.route('/ask', methods=['POST'])
def ask_question():

    data = request.json
    query_text = data.get('question', '')

    results = db.similarity_search_with_score(query_text, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=API_TOKEN)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id") for doc, _score in results]
    formatted_response = {
        'response': response_text
    }

    print(formatted_response)
    return jsonify(formatted_response)

if __name__ == '__main__':
    app.run(debug=True)
