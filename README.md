# DB_tutor

The prototype is accessible at: `http://127.0.0.1:5000/`

---

## Running the Prototype

### Option 1: Ollama (Recommended)

**Prerequisites:** Ollama must be installed and running in the background.

#### First-time setup
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
pip install flask requests numpy
```

#### Start the application
```bash
cd Prototype
python app.py
```

#### Subsequent runs
```bash
cd Prototype
python app.py
```

---

### Option 2: Docker

**Prerequisites:** Docker must be installed and running in the background.

#### First-time setup

**Terminal 1** — Start Ollama container *(from the Prototype folder)*:
```bash
docker-compose up ollama
```

**Terminal 2** — Pull the required models:
```bash
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.1
```

**Terminal 3** — Start the application *(from the Prototype folder)*:
```bash
docker-compose up app
```

#### Subsequent runs
```bash
docker-compose up
```

#### Stop the application
```bash
docker-compose down
```

---