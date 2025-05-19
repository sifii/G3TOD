# G³TOD

**G³TOD** is a generalizable framework designed to enhance personalization in task-oriented dialogue systems. It utilizes three structured knowledge graphs extracted from conversation history:

- 🧾 **Entity Context Graph**  
- 👤 **User Persona Graph**  
- 🧠 **Commonsense Reasoning Graph**

These graphs help model rich, user-aware, and context-driven interactions, improving response relevance and coherence.

---

# 🧠 Triplet Extraction System

This repository implements the triplet extraction module for G3TOD. It extracts (subject, relation, object) triplets from natural language using entity recognition and ConceptNet-based reasoning. These triplets are crucial for building knowledge graphs that support personalized dialogue systems, semantic analysis, and downstream NLP tasks.

---

## 📁 Project Structure

├── conceptnet.py # Interface for querying ConceptNet

├── entity.py # Entity recognition and cleaning logic

├── extract_rel_triplets.py # Core logic for relationship extraction

├── triplet_extract.py # Utilities for triplet formatting and filtering

├── main.py # Script for sample triplet extraction

├── run.py # Entry point to execute the extraction pipeline

├── test.py # Unit tests for each component

├── requirements.txt # List of required Python packages

└── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 Installation

### 1. Clone the Repository
    ```bash
    git clone https://github.com/yourusername/triplet-extraction.git
    cd triplet-extraction


### 2. Create and Activate a Virtual Environment
     ```bash
     python -m venv venv
    source venv/bin/activate        # On Windows: venv\Scripts\activate

#### 🤖 Ollama Setup (if not already)
      ```bash
      1. Install Ollama
      Follow the official instructions here: https://ollama.com/download

      2. Start Ollama
      ollama serve
      3. Pull a Model (e.g., LLaMA2 or openai)
     ollama pull llama2
     # or
     ollama pull openai

### 3. Install Dependencies
      ```bash
      pip install -r requirements.txt

### 4. Run the Extraction Pipeline
     ```bash
      🛠️ Usage
         
    python run.py
    This processes a predefined input (from main.py). You can modify main.py to process:
    Run Unit Tests
    python test.py
This will validate core functionalities such as entity recognition, ConceptNet query interface, and triplet extraction accuracy.

📦 Dependencies
spacy – Entity recognition and NLP processing

nltk – Tokenization, POS tagging, and basic NLP tools

requests – Querying ConceptNet API

networkx – For handling and visualizing graph relations

See requirements.txt for exact versions.
