import pandas as pd
import requests
from urllib.parse import quote
import time

def get_conceptnet_triplets(concept, language='en'):
    concept_uri = quote(concept.lower())
    url = f"http://api.conceptnet.io/c/{language}/{concept_uri}"
    try:
        obj = requests.get(url).json()
        triplets = []
        for edge in obj.get('edges', []):
            rel = edge['rel']['label']
            start = edge['start']['label']
            end = edge['end']['label']
            triplets.append((start, rel, end))
        return triplets[:5]  # Limit to top 5 for speed
    except Exception as e:
        return [("error", "error", str(e))]

# Load dataset
df = pd.read_csv("hinglish_with_all_triplets.csv")

# Add new column for ConceptNet triplets
df["conceptnet_triplets"] = ""

# Loop through each row and enrich with ConceptNet triplets
for idx, row in df.iterrows():
    keywords_str = row.get("keywords", "")
    if not isinstance(keywords_str, str) or not keywords_str.strip():
        continue

    keywords = [kw.strip().lower() for kw in keywords_str.split(",") if kw.strip()]
    all_triplets = []

    for kw in keywords:
        triplets = get_conceptnet_triplets(kw)
        all_triplets.extend(triplets)
        time.sleep(0.5)  # Avoid rate-limiting

    # Convert triplets to a string for saving
    triplet_str = "; ".join([f"({t[0]}, {t[1]}, {t[2]})" for t in all_triplets])
    df.at[idx, "conceptnet_triplets"] = triplet_str

# Save the updated file
df.to_csv("hinglish_with_conceptnet.csv", index=False)
print("✅ ConceptNet triplets added and saved.")
