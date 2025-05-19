import pandas as pd
import requests
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

# -----------------------------
# 1. Helper: Build prompt from triplets
# -----------------------------
def build_prompt(user_utterance, triplets):
    triplet_str = "\n".join([f"({h}, {r}, {t})" for (h, r, t) in triplets])
    prompt = f"""User said: "{user_utterance}"

Here are some relevant knowledge triplets:
{triplet_str}

Respond like a helpful assistant in Hinglish."""
    return prompt

# -----------------------------
# 2. Helper: Query Ollama GPT
# -----------------------------
def query_ollama(prompt, model="gpt-3.5-turbo"):
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
            return response.json().get("response", "").strip()
        except:
            return "JSON error"
    return "Error: " + str(response.status_code)

# -----------------------------
# 3. Load Data
# -----------------------------
df = pd.read_csv("test.csv")  # Columns: speaker, dialogue, assistant_response, triplets
df = df[df["speaker"] == "user"].reset_index(drop=True)

# -----------------------------
# 4. Generate Assistant Responses
# -----------------------------
preds = []
refs = []
raw_prompts = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    user_input = row["dialogue"]
    gold_response = row["assistant_response"]
    
    # Ensure triplets are parsed correctly
    try:
        triplets = eval(row["triplets"])  # Should be list of (head, rel, tail)
    except:
        triplets = []

    prompt = build_prompt(user_input, triplets)
    generated = query_ollama(prompt)

    preds.append(generated)
    refs.append(gold_response)
    raw_prompts.append(prompt)

# -----------------------------
# 5. Evaluation
# -----------------------------

# A. BLEU
bleu_scores = [sentence_bleu([ref.split()], pred.split()) for ref, pred in zip(refs, preds)]
avg_bleu = sum(bleu_scores) / len(bleu_scores)

# B. ROUGE
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rougeL_scores = [scorer.score(r, p)['rougeL'].fmeasure for r, p in zip(refs, preds)]
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

# C. BERTScore
P, R, F1 = score(preds, refs, lang='en')
avg_bertscore_f1 = F1.mean().item()

# -----------------------------
# 6. Save Results
# -----------------------------
df["generated_response"] = preds
df["bleu_score"] = bleu_scores
df["rougeL_score"] = rougeL_scores
df["bertscore_f1"] = F1.tolist()
df["prompt"] = raw_prompts
df.to_csv("test_with_generated_and_scores.csv", index=False)

# -----------------------------
# 7. Print Summary
# -----------------------------
print("\n✅ Evaluation Complete:")
print(f"Average BLEU:       {avg_bleu:.4f}")
print(f"Average ROUGE-L:    {avg_rougeL:.4f}")
print(f"Average BERTScore F1: {avg_bertscore_f1:.4f}")
