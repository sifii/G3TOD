import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load the annotated triplet dataset
triplet_df = pd.read_csv("hinglish_with_conceptnet.csv")

# Load test user dialogues (containing a 'dialogue' column)
test_df = pd.read_csv("user_test_dialogues.csv")  # replace with your test file

# Ensure output directory exists
os.makedirs("triplet_graphs", exist_ok=True)

def extract_triplets_for_keywords(keywords, triplet_series):
    relevant_triplets = []
    for triplets_str in triplet_series.dropna():
        for t in triplets_str.split(";"):
            t = t.strip()
            if not t or "(" not in t:
                continue
            for kw in keywords:
                if kw.lower() in t.lower():
                    relevant_triplets.append(t)
                    break
    return list(set(relevant_triplets))  # remove duplicates

def build_and_save_graph(triplets, filename):
    G = nx.DiGraph()
    for triplet in triplets:
        try:
            subj, rel, obj = triplet.strip("()").split(",")
            G.add_edge(subj.strip(), obj.strip(), label=rel.strip())
        except Exception as e:
            continue  # skip malformed triplets
    
    if G.number_of_edges() == 0:
        return

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=8, edge_color="gray")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(f"triplet_graphs/{filename}.png")
    plt.close()

for idx, row in test_df.iterrows():
    utterance = row['dialogue']
    # You can use LLM or TF-IDF for better keyword extraction, here's a simple split
    keywords = [kw.strip().lower() for kw in utterance.split() if len(kw.strip()) > 2]

    ner_triplets = extract_triplets_for_keywords(keywords, triplet_df["nre_triplets"])
    persona_triplets = extract_triplets_for_keywords(keywords, triplet_df["persona_triplets"])
    conceptnet_triplets = extract_triplets_for_keywords(keywords, triplet_df["conceptnet_triplets"])

    # Save each graph
    build_and_save_graph(ner_triplets, f"user_{idx}_ner")
    build_and_save_graph(persona_triplets, f"user_{idx}_persona")
    build_and_save_graph(conceptnet_triplets, f"user_{idx}_conceptnet")

    print(f"✅ Saved graphs for user_{idx}: {utterance}")

print("🎉 All graphs saved in 'triplet_graphs' folder.")
