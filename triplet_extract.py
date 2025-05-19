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

# Load dataset
df = pd.read_csv("tes_dataset.csv")

print(df)
# Add new columns for all extractions

df["nre_triplets"] = ""
df["persona_triplets"] = ""
df["commonsense_triplets"] = ""

for idx, row in tqdm(df.iterrows(), total=len(df)):
    #if row["speaker"].strip().lower() != "user":
       # continue

    text = row["user"]


    # --- NRE Triplet Extraction ---
    nre_prompt = f"Write Extract Named Entity Relationship in triplets from given Hinglish dialogue: \"{text}\". Format as (subject, relation, object). example: (alarm, related_to, wakeup),(appioment, has, doctor) etc "
    df.at[idx, "nre_triplets"] = generate_response(nre_prompt)

    # --- Persona Triplet Extraction ---
    persona_prompt = f"Write only user intent triplets for this Hinglish dialogue: \"{text}\", extract persona triplets like (user, preference/intention, object) like (user, has, good_routine),(user, wake_up, morining),(user, has, health_issue)."
    df.at[idx, "persona_triplets"] = generate_response(persona_prompt)

    # --- Commonsense Triplet Extraction ---
    commonsense_prompt = f"Write only commonsensense triplets from this Hinglish dialogue: \"{text}\", extract extra commonsense knowledge or linkage knowlegde in this format (subject, relation, object) example (user, want,morning_routine),(doctor,related_to, good_routine),(appioment, need, reminder) etc.."
    df.at[idx, "commonsense_triplets"] = generate_response(commonsense_prompt)

# Save all results
df.to_csv("triplets_for_test.csv", index=False)
print("Entities, keywords, and triplets extracted and saved.")
