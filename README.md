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
