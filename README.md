# G3TOD
A generalizable framework that enhances personalization using three structured knowledge graphs: entity context, user persona, and commonsense reasoning, all extracted from conversation history. 
# Triplet Extraction System

This project extracts relational triplets from textual input using ConceptNet and entity recognition modules. It includes a main pipeline, testing module, and utility components.

## 📁 Project Structure

- `main.py`: Entry point to run the triplet extraction.
- `triplet_extract.py`: Triplet extraction logic.
- `extract_rel_triplets.py`: Helper for extracting relationship triplets.
- `entity.py`: Entity recognition and processing.
- `conceptnet.py`: Uses ConceptNet for knowledge-based expansion.
- `test.py`: Contains test cases to validate the functionality.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/triplet-extraction.git
   cd triplet-extraction
# 🧠 Triplet Extraction System

This project implements a pipeline to extract relational triplets from natural language text using entity recognition and ConceptNet-based relationship reasoning. It is useful for building knowledge graphs, information extraction, and semantic analysis.

---

## 📁 Project Structure

.
├── conceptnet.py # Interface for querying ConceptNet
├── entity.py # Entity recognition and cleaning logic
├── extract_rel_triplets.py # Core logic for relationship extraction
├── triplet_extract.py # Utilities for triplet formatting and filtering
├── main.py # Main driver script
├── run.py # Entry point to execute the pipeline
├── test.py # Unit tests for modules
├── requirements.txt # Required Python packages
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/triplet-extraction.git
cd triplet-extraction
2. Create and activate a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
🛠️ Usage
Run the extraction pipeline
bash
Copy
Edit
python run.py
By default, this processes a predefined input (in main.py). You can modify it to process a file or user input.

✅ Example
Input
nginx
Copy
Edit
Barack Obama was born in Hawaii.
Output
css
Copy
Edit
[('Barack Obama', 'was born in', 'Hawaii')]
🧪 Running Tests
To run the unit tests:

bash
Copy
Edit
python test.py
This will validate the functionality of various components including entity recognition and triplet extraction.

📦 Dependencies
spacy: For entity recognition and NLP processing

nltk: For tokenization and semantic tools

requests: For ConceptNet API calls

networkx: For handling ConceptNet graph relations

(See requirements.txt for exact versions.)

📄 License
This project is licensed under the MIT License.

🤝 Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.






