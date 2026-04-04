from flask import Flask, render_template, request, jsonify, session
import requests
import numpy as np
from documents import documents
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

OLLAMA_URL = "http://localhost:11434"


def embed(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(question):
    question_embedding = embed(question)

    scored = []
    for doc in documents:
        doc_embedding = embed(doc)
        score = cosine_similarity(question_embedding, doc_embedding)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, score in scored[:3]]
    return top_docs


def build_system_prompt(level):
    level_desc = {
        "beginner": "The student is a beginner. Use simple language and short examples.",
        "intermediate": "The student has intermediate knowledge of databases and SQL.",
        "advanced": "The student is advanced. Use precise technical language.",
    }.get(level, "The student has intermediate knowledge.")

    return (
        f"You are a database tutor for university-level Computer Science students.\n"
        f"{level_desc}\n"
        f"Guide the student with questions rather than giving direct answers. "
        f"If they are stuck, offer a small hint. Never just solve the problem for them.\n"
        f"Always ground your answers in the provided context. Be concise."
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    session["level"] = data.get("level", "intermediate")
    session["history"] = []
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please type a question."})

    level = session.get("level", "intermediate")
    history = session.get("history", [])

    top_docs = retrieve(question)
    context = "\n".join(top_docs)

    system_prompt = build_system_prompt(level)

    prompt = f"""{system_prompt}

Relevant context:
{context}

Question: {question}
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()
    print(data)
    answer = data.get("response", "No response from model.")

    history.append({"user": question, "assistant": answer})
    session["history"] = history

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
