import requests
import json
import pandas as pd
from tqdm import tqdm

def generate_response(prompt, model="llama3.2:1b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            generated_text = json_response.get("response", "").strip()
            return generated_text
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", e)
            return "Error decoding JSON response"
    else:
        print(f"Request failed with status code {response.status_code}")
        return ""

# Load your dataset
df = pd.read_csv("task_oriented_dialogue_hinglish_full.csv")  # Replace with actual path
df= df[:10]
# Add columns for entity and keyword extraction
df["entities"] = ""
df["keywords"] = ""

# Process only 'user' dialogues
for idx, row in tqdm(df.iterrows(), total=len(df)):
    if row["speaker"].strip().lower() != "user":
        continue

    text = row["dialogue"]

    # Prompt to extract entities
    entity_prompt = f"Hinglish dialogue: \"{text}\"\n\nExtract named entities (like product names, locations, symptoms, currency values, etc.) as a comma-separated list."
    entities = generate_response(entity_prompt)

    # Prompt to extract keywords
    keyword_prompt = f"Hinglish dialogue: \"{text}\"\n\nExtract the main keywords or key phrases from this text as a comma-separated list."
    keywords = generate_response(keyword_prompt)

    # Save the results
    df.at[idx, "entities"] = entities
    df.at[idx, "keywords"] = keywords

# Save to new file
df.to_csv("hinglish_with_entities_keywords.csv", index=False)
print("Entities and keywords extracted successfully.")
