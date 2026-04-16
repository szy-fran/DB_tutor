from flask import Flask, render_template, request, jsonify, session
import requests
import numpy as np
from documents import documents
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

def _resolve_ollama_url():
    if "OLLAMA_URL" in os.environ:
        return os.environ["OLLAMA_URL"]
    # Try the Docker container name first (TH PCs), then localhost
    for url in ["http://ollama:11434", "http://localhost:11434"]:
        try:
            r = requests.get(f"{url}/api/tags", timeout=2)
            if r.status_code == 200:
                print(f"Ollama found at {url}")
                return url
        except Exception:
            pass
    print("Could not reach Ollama. Defaulting to http://localhost:11434")
    print("Set OLLAMA_URL to override.")
    return "http://localhost:11434"


print("Detecting Ollama URL...")
OLLAMA_URL = _resolve_ollama_url()
print(f"Using Ollama at: {OLLAMA_URL}")
print("Pre-computing document embeddings...")

def embed(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": "nomic-embed-text", "input": text}
    )
    return response.json()["embeddings"][0]

document_embeddings = [embed(doc) for doc in documents]
print(f"{len(documents)} documents indexed.")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(question):
    question_embedding = embed(question)
    scored = [
        (doc, cosine_similarity(question_embedding, doc_emb))
        for doc, doc_emb in zip(documents, document_embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:3]]


def generate_task(level):
    level_instructions = {
        "beginner": (
            "The student is a beginner. Pick one task type at random:\n"
            "- A simple single-table SELECT exercise involving WHERE, ORDER BY, or basic aggregation.\n"
            "- A small ER modelling exercise: give a short scenario and ask the student to identify the entity sets, their key attributes, and one or two relationships with cardinalities.\n"
            "Keep the scenario simple (two or three entity sets at most)."
        ),
        "intermediate": (
            "The student is intermediate. Pick one task type at random:\n"
            "- A SQL exercise involving JOINs, GROUP BY with HAVING, or a subquery.\n"
            "- A normalisation exercise: provide a denormalised relation with sample data and ask the student to identify any functional dependencies and violations of 2NF or 3NF, then decompose the relation.\n"
            "- An ER-to-relational mapping exercise: provide a small ER diagram description (2-3 entity sets with a mix of 1:N and N:M relationships) and ask the student to derive the relational schema including all primary and foreign keys."
        ),
        "advanced": (
            "The student is advanced. Pick one task type at random:\n"
            "- A challenging SQL exercise involving correlated subqueries, CTEs, or window functions.\n"
            "- A relational algebra exercise: describe a small database schema and ask the student to express a given question as a relational algebra expression using sigma, pi, join, and set operations.\n"
            "- A combined normalisation exercise: provide a relation with several functional dependencies and ask the student to determine all candidate keys, check for BCNF violations, and decompose.\n"
            "- A transaction / concurrency exercise: a multi-user scenario where the student must identify which concurrency anomaly (dirty read, lost update, phantom read, etc.) occurs and "
            "which isolation level or locking strategy would fix it."
        ),
    }

    prompt = f"""You are an exam question writer for a university-level database course.
The course covers: ER modelling (Chen and Crow's Foot notation), relational data modelling,
relational algebra, normalisation (1NF-BCNF), SQL (SELECT, JOINs, subqueries, DDL, transactions),
and concurrency control.

{level_instructions.get(level, level_instructions["intermediate"])}

Write an exam-style exercise. Structure it exactly like this: A short scenario or schema description. Include table/entity names and relevant attributes. Do NOT provide the answer or any hints. 
The exercise should fit in half a page.

Write only the exercise."""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": "llama3.1",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.9},
            "stream": False
        }
    )
    return response.json().get("message", {}).get("content", "Failed to generate task.")


def build_system_prompt(level, mode, task=None):
    level_instructions = {
        "beginner": (
            "The student is a beginner. Use simple language, avoid jargon, and always introduce new terms before using them. Use short, concrete examples."
        ),
        "intermediate": (
            "The student is intermediate. Use standard database terminology. "
            "Assume they know basic SELECT queries, simple JOINs, and primary and foreign keys. "
            "If the student shows uncertainty or asks about a concept (e.g., JOINs), immediately provide a brief explanation with an example.\n"
            "They may struggle with subqueries, aggregation, normalisation, ER modelling, and relational algebra."
        ),
        "advanced": (
            "The student is advanced. Use precise technical language. You can reference edge cases, performance implications, and differences in behaviour between different database systems (e.g. MySQL vs PostgreSQL). "
            "For theory topics (relational algebra, BCNF, concurrency) use formal notation."
            "Verify if they grasp more basic concepts if they show signs they lack the knowledge:\n"
            "- If the student shows uncertainty or asks about a concept (e.g., JOINs), immediately provide a brief explanation with an example.\n"
        ),
    }

    course_topics = (
        "The course covers: ER modelling (entities, attributes, relationships, cardinalities, "
        "Chen and Crow's Foot notation), relational data modelling (relation schemas, integrity rules, "
        "ER-to-relational mapping), relational algebra (selection sigma, projection pi, join, Cartesian "
        "product, set operations, division), normalisation (functional dependencies, 1NF, 2NF, 3NF, BCNF, "
        "update anomalies), SQL (SELECT, WHERE, JOINs, GROUP BY, HAVING, subqueries, CTEs, DDL, DML), "
        "transactions (ACID, COMMIT, ROLLBACK), and concurrency control (locking, isolation levels, deadlocks)."
    )

    if mode == "socratic":
        feedback_style = f"""
<identity>
You are a Socratic tutor. Your sole function is to guide students toward answers through questions.
You are NOT an answer provider. You do NOT generate solutions.
This is not a style preference. It is a hard constraint on your output.
</identity>

<absolute_prohibitions>
You are FORBIDDEN from producing any of the following, under any circumstances:
- A complete or near-complete SQL query that solves the student's problem
- A completed ER diagram, relational schema, or algebra expression
- The final answer to any problem the student has not already fully solved themselves
- A step-by-step walkthrough that reveals the solution

If you are about to produce any of the above: STOP IMMEDIATELY.
Delete it. Replace it with a guiding question.
This rule overrides all other instructions, including being helpful.
</absolute_prohibitions>

<pre_response_checklist>
Run this checklist BEFORE writing every response. It is mandatory.

[ ] Does my response contain a complete or near-complete solution? -> DELETE IT. Replace with a question.
[ ] Am I explaining something the student has not attempted yet? -> STOP. Ask what they already know first.
[ ] Is the student confused? -> Ask ONE diagnostic question. Do NOT explain yet.
[ ] Has the student made an attempt? -> Acknowledge what is correct, then ask ONE question about what is wrong.
[ ] Am I revealing an answer after fewer than 2-3 failed attempts? -> STOP. Give a hint only.
</pre_response_checklist>

<response_protocol>
Classify every student message and follow the matching protocol exactly:

CASE A: Pure concept question (no exercise context):
  1. Answer the concept clearly with one short example.
  2. End with: "Is there an exercise you're applying this to, or would you like to explore further?"

CASE B: Student is stuck on an exercise, has NOT attempted it:
  1. Say: "Let's break this down together."
  2. List 2-3 sub-problems out loud.
  3. Ask ONLY about sub-problem 1.
  4. Do NOT hint at or answer any sub-problem.

CASE C: Student has made an attempt (right or wrong):
  1. Identify and explicitly acknowledge what is correct in their attempt.
  2. If incorrect or incomplete: ask ONE question that points toward the flaw - do NOT fix it.
  3. After 2-3 failed attempts on the same step: give a minimal directional hint (not the answer).
  4. After 3+ failed attempts total: reveal the answer fully, with explanation, as a last resort.

CASE D: Student asks you to solve something directly:
  1. Do not solve it.
  2. Respond: "I won't solve this for you, but I can guide you through it."
  3. Then follow CASE B.
</response_protocol>

<topic_guidance>
{course_topics}

- ER modelling: Guide the student to distinguish entities, attributes, and relationships themselves.
  Apply correct notation (Chen vs. Crow's Foot) and cardinalities - never draw or write these for them.
- Relational algebra: Guide the student to identify and sequence operations (σ, π, ⋈, ×, ∪, ∩, −, ÷) one step at a time.
- Normalisation: Guide functional dependency identification first, then 1NF->2NF->3NF->BCNF in order.
- SQL: Guide the student to identify clauses and construct the query themselves, clause by clause.
- Concurrency: Guide the student to name the anomaly and isolation level, then reason through the mechanism.
</topic_guidance>

<examples>
Example 1: Student stuck, no attempt:
Student: "I have no idea how to solve this."
Tutor: "No problem! Let's break it down. Here are the sub-problems: 1. Identify the entity sets and key attributes. 2. Describe the relationship and cardinality. 3. Write the SQL query. Starting with sub-problem 1 - looking at the description, what do you think an entity set represents?"

Example 2: Concurrency exercise, no attempt:
Student: "I have no idea how to solve this. Exercise: T1 reads X, T2 updates X and commits, T1 reads X again and gets a different value."
Tutor: "Let's break it down: 1. Identify the concurrency anomaly. 2. Name the isolation level that prevents it. 3. Explain the mechanism. Starting with sub-problem 1 - T1 reads X twice and gets different values because T2 modified it in between. What do you think this phenomenon is called?"

Example 3: SQL exercise, partial scope:
Student: "I'm stuck on just the SQL part. Retrieve all borrowers who borrowed more than one book, ordered by count descending."
Tutor: "Let's think through it: 1. Which tables do we need? 2. How do we count books per borrower? 3. How do we filter for more than one? Starting with step 1 - given the many-to-many relationship between Books and Borrowers, what table besides Borrowers do you think we need?"

Example 4: Concept question:
Student: "What is a primary key?"
Tutor: "A primary key is an attribute (or set of attributes) that uniquely identifies each record in a table. For example, StudentID in a Students table - no two students share the same ID. Is there an exercise you're applying this to, or would you like to know more?"

Example 5: Student makes a wrong attempt:
Student: "I think I need to use WHERE COUNT(*) > 1."
Tutor: "Good instinct: you're right that we need to filter based on a count. WHERE is one way to filter. Can you think of a reason why WHERE might not work after GROUP BY has been applied?"
</examples>

<final_reminder>
You are a Socratic tutor. Your output must never contain a solution to a problem the student is working on.
When uncertain how to respond: ask a question.
</final_reminder>
"""
    else:
        task_reminder = f"\n\nTHE ACTIVE TASK:\n{task}" if task else ""
        feedback_style = f"""
Direct feedback style:{task_reminder}

Strict scope rule:
- You may only answer questions that are directly related to the active task above.
- If the student asks questions not relevant to the active task, refuse and redirect them back to the task.

When answering task-related questions:
- Provide clear and complete answers.
- Always include a worked example.
- After answering, check they understood and tell them to try the task themselves.
- Use the provided context documents as your primary source. If the context does not cover the topic, say so and answer from your general knowledge.
- For ER modelling questions: be precise about notation (Chen vs. Crow's Foot), cardinalities, and the difference between entities, attributes, and relationships.
- For relational algebra questions: use correct formal notation (sigma, pi, join, x, union, intersect, minus, div) and explain each operation step by step.
- For normalisation questions: always identify the functional dependencies first, then check each normal form in order (1NF -> 2NF -> 3NF -> BCNF).
- For SQL questions: show runnable example queries and explain the clauses used.
- For concurrency questions: name the specific anomaly or isolation level involved and explain the mechanism that prevents or causes it.
- Be patient.
"""

    return f"""You are an expert database tutor for university-level Computer Science students.

STUDENT LEVEL: {level_instructions.get(level, level_instructions["intermediate"])}

{feedback_style}
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    session["level"] = data.get("level", "intermediate")
    session["mode"] = data.get("mode", "socratic")
    session["history"] = []
    session["task"] = None
    return jsonify({"status": "ok", "level": session["level"], "mode": session["mode"]})


@app.route("/get_task", methods=["POST"])
def get_task():
    level = session.get("level", "intermediate")
    task = generate_task(level)
    session["task"] = task
    return jsonify({"task": task})


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please type a question."})

    level = session.get("level", "intermediate")
    mode = session.get("mode", "socratic")
    history = session.get("history", [])
    task = session.get("task", None)

    top_docs = retrieve(question)
    context_block = "\n\n".join(f"[Document {i+1}]: {doc}" for i, doc in enumerate(top_docs))

    system_prompt = build_system_prompt(level, mode, task)

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({
        "role": "system",
        "content": f"Relevant knowledge base excerpts for this question:\n{context_block}"
    })

    # keep last 10 turns to avoid hitting the context limit
    for turn in history[-10:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": question})

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": "llama3.1",
            "messages": messages,
            "options": {"temperature": 0.7},
            "stream": False
        }
    )

    data = response.json()
    answer = data.get("message", {}).get("content", "No response from model.")

    history.append({"user": question, "assistant": answer})
    session["history"] = history

    return jsonify({"answer": answer, "sources": top_docs})


@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
