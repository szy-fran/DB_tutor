from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
from documents import documents

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434"

# ---------- Embedding ----------
def embed(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]


# ---------- Cosine Similarity ----------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- Retrieve Top Documents ----------
def retrieve(question):
    question_embedding = embed(question)

    scored = []
    for doc in documents:
        doc_embedding = embed(doc)
        score = cosine_similarity(question_embedding, doc_embedding)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, score in scored[:2]]
    return "\n".join(top_docs)


# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]

    context = retrieve(question)

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        }
    )

    # answer = response.json()["response"]
    data = response.json()
    print(data)  # DEBUG

    answer = data.get("response", "No response from model")

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)