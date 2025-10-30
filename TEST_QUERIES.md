# 5 Example Test Queries for Your Trained Model

## Model Information
- **Model Path**: `models\vaultgemma_dp_20251017_062150`
- **Training Stats**: F1=0.231, EM=0.000
- **Training Data**: 50 patients with diagnosis and medication QA pairs

---

## Test Query 1: Diagnosis for Karen Hooper
**What to expect**: The model should respond with "Sleep Apnea"

```bash
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "What is the diagnosis for Karen Hooper?" --gen-beams 1 --gen-max-new-tokens 6
```

---

## Test Query 2: Medication for Sara Moon
**What to expect**: The model should respond with "Amlodipine 5mg"

```bash
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "What medication is Sara Moon taking?" --gen-beams 1 --gen-max-new-tokens 6
```

---

## Test Query 3: Condition for Lindsey Thompson
**What to expect**: The model should respond with "Type 2 Diabetes"

```bash
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "What condition was Lindsey Thompson diagnosed with?" --gen-beams 1 --gen-max-new-tokens 8
```

---

## Test Query 4: Medication for Donald Miller
**What to expect**: The model should respond with "Lorazepam 0.5mg"

```bash
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "Medication for Donald Miller?" --gen-beams 1 --gen-max-new-tokens 6
```

---

## Test Query 5: Diagnosis for Kathleen Howard
**What to expect**: The model should respond with "Asthma"

```bash
python LLM_Diffrential_Privacy.py --query --model models\vaultgemma_dp_20251017_062150 --prompt "Diagnosis for Kathleen Howard?" --gen-beams 1 --gen-max-new-tokens 4
```

---

## Notes:
- All queries use `--gen-beams 1` for faster, deterministic responses
- `--gen-max-new-tokens` is kept low (4-8) for short, factual answers
- These patients are from your actual training data
- With F1=0.231, expect some incorrect or incomplete responses (this is normal for DP-trained models on small datasets)

## Quick Test All:
To test all 5 queries automatically, run:

```bash
python test_model_queries.py --auto
```

Or copy each command above individually.

