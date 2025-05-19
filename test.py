import pandas as pd
import requests
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bert_score import score as bertscore

# -----------------------------
# 1. Load Triplet Data
# -----------------------------
df = pd.read_csv('triplets_456.csv')  # replace with your file path

# Extract triplets from strings
def extract_triplets(text):
    pattern = r'\d+\.\s*(.+?):?\s*\(?Subject:? (.+?),\s*Object:? (.+?),\s*Relation:? (.+?)\)?(?:\.|$)|\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    triplets = []
    for match in matches:
        if match[1] and match[2] and match[3]:
            triplets.append((match[1].strip(), match[3].strip(), match[2].strip()))
        elif match[4] and match[5] and match[6]:
            triplets.append((match[4].strip(), match[5].strip(), match[6].strip()))
    return triplets

def parse_triplet_column(column):
    all_triplets = []
    for entry in df[column].fillna(''):
        triplets = extract_triplets(entry)
        all_triplets.append(triplets)
    return all_triplets

df['nre_triplets_parsed'] = parse_triplet_column('nre_triplets')
df['persona_triplets_parsed'] = parse_triplet_column('persona_triplets')
df['commonsense_triplets_parsed'] = parse_triplet_column('commonsense_triplets')

# Combine triplets
all_nre = sum(df['nre_triplets_parsed'].tolist(), [])
all_persona = sum(df['persona_triplets_parsed'].tolist(), [])
all_commonsense = sum(df['commonsense_triplets_parsed'].tolist(), [])

# -----------------------------
# 2. Load Test Data
# -----------------------------
test = pd.read_csv('final_file.csv')  # replace with your file path
print(test)
# Match keywords
def find_relevant_triplets_by_keywords(text, triplet_list):
    text_lower = text.lower()
    relevant = []
    for subj, rel, obj in triplet_list:
        if (subj.lower() in text_lower) or (obj.lower() in text_lower):
            relevant.append((subj, rel, obj))
    return relevant

#test['combined'] = test['User'].fillna('')
#test['relevant_nre_triplets'] = test['combined'].apply(lambda x: find_relevant_triplets_by_keywords(x, all_nre))
#test['relevant_persona_triplets'] = test['combined'].apply(lambda x: find_relevant_triplets_by_keywords(x, all_persona))
#test['relevant_commonsense_triplets'] = test['combined'].apply(lambda x: find_relevant_triplets_by_keywords(x, all_commonsense))

# -----------------------------
# Generate Responses
# -----------------------------
def generate_response(prompt, model="openchat:latest"):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        try:
            return response.json().get("response", "").strip()
        except json.JSONDecodeError:
            return "Error decoding JSON response"
    return ""

preds, refs, prompts = [], [], []

for idx, row in tqdm(test.iterrows(), total=len(test)):
    user_input = row["User"]
    gold_response = row["assistant"]
    #nre = row['relevant_nre_triplets']
    #persona = row['relevant_persona_triplets']
    #common = row['relevant_commonsense_triplets']
    generated_response = row['generated_response']
    #prompt = f"You are assistant. Please reply in one line and to the point in hinglish language: {user_input}"
    #generated = generate_response(prompt)
    preds.append(generated_response)
    refs.append(gold_response)
    #prompts.append(prompt)

# -----------------------------
# Evaluation
# -----------------------------
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load GPT2 for PPL
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

def calculate_ppl(text, max_length=1024):
    # Tokenize the input text with truncation
    inputs = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Calculate the perplexity
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


bleu1_scores, bleu2_scores, bleu3_scores,bleu4_scores = [], [], [], []
bleu_scores, meteor_scores, rougeL_scores, ppl_scores = [], [], [], []
pa_scores, npa_scores, nsc_scores, lengths = [], [], [], []


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smooth = SmoothingFunction().method1

for pred, ref in zip(preds, refs):
    # BLEU

    pred = pred.lower().strip()
    ref = ref.lower().strip()

    ref_tokens = ref.split()
    pred_tokens = pred.split()

    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)
    
    bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth))


    # METEOR
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
    # ROUGE-L
    rougeL_scores.append(rouge.score(ref, pred)['rougeL'].fmeasure)
    # PPL
    ppl_scores.append(calculate_ppl(pred))
    # PA
    pa_scores.append(1 if pred.strip().lower() == ref.strip().lower() else 0)
    # NPA
    # pred_nps, ref_nps = extract_noun_phrases(pred), extract_noun_phrases(ref)
    # common_nps = pred_nps.intersection(ref_nps)
    # npa_scores.append(len(common_nps) / max(1, len(ref_nps)))
    # NSC
    emb = embedder.encode([pred, ref])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    nsc_scores.append(sim)
    # R-LEN
    lengths.append(len(pred.split()))

# BERTScore
P, R, F1 = bertscore(preds, refs, lang='en')
avg_bertscore_P = P.mean().item()
avg_bertscore_R = R.mean().item()
avg_bertscore_f1 = F1.mean().item()

# -----------------------------
# Save & Summary
# -----------------------------
test["generated_response"] = preds
test['BLEU1'] = bleu1_scores
test['BLEU2'] = bleu2_scores
test['BLEU3'] = bleu3_scores
test['BLEU4'] = bleu4_scores
test["METEOR"] = meteor_scores
test["ROUGE-L"] = rougeL_scores
test["PPL"] = ppl_scores
#test["PA"] = pa_scores
# test["NPA"] = npa_scores
test["NSC"] = nsc_scores
test["R-LEN"] = lengths
test["BS-F1"] = F1.tolist()

test.to_csv("final_test_with_all_metrics.csv", index=False)

# Print summary
print("\n✅ Evaluation Summary:")
print(f"Average PPL:          {sum(ppl_scores)/len(ppl_scores):.4f}")
print(f"Average BLEU-1: {sum(bleu1_scores) / len(bleu1_scores)}")
print(f"Average BLEU-2: {sum(bleu2_scores) / len(bleu2_scores)}")
print(f"Average BLEU-3: {sum(bleu3_scores) / len(bleu3_scores)}")
print(f"Average BLEU-4: {sum(bleu4_scores) / len(bleu4_scores)}")
print(f"Average BLEU:         {sum(bleu_scores)/len(bleu_scores):.4f}")
print(f"Average ROUGE-L:         {sum(rougeL_scores)/len(rougeL_scores):.4f}")
print(f"Average METEOR:       {sum(meteor_scores)/len(meteor_scores):.4f}")
print(f"Average BS-P:        {avg_bertscore_P:.4f}")
print(f"Average BS-R:        {avg_bertscore_R:.4f}")
print(f"Average BS-F1:        {avg_bertscore_f1:.4f}")
#print(f"Precision Accuracy:   {sum(pa_scores)/len(pa_scores):.4f}")
# print(f"Avg NP Agreement:     {sum(npa_scores)/len(npa_scores):.4f}")
print(f"Avg Sentence CosSim:  {sum(nsc_scores)/len(nsc_scores):.4f}")
print(f"Avg Resp Length:      {sum(lengths)/len(lengths):.2f} tokens")
