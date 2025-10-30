# 10 Test Queries for VaultGemma Model
**Model**: `models\vaultgemma_dp_20251017_062150`

Copy and paste each command into your terminal to test the model:

---

## Query 1: Medication for Sara Moon
**Expected Answer**: Amlodipine 5mg
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "What was prescribed to Sara Moon"
```

---

## Query 2: Diagnosis for Donald Miller
**Expected Answer**: Hyperlipidemia
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "What is the diagnosis for Donald Miller"
```

---

## Query 3: Medication for Lindsey Thompson
**Expected Answer**: Metformin 500mg
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "What medication is Lindsey Thompson taking"
```

---

## Query 4: Diagnosis for Robert Smith
**Expected Answer**: Chronic Obstructive Pulmonary Disease
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "What condition was Robert Smith diagnosed with"
```

---

## Query 5: Medication for Mary Davis
**Expected Answer**: Gabapentin 300mg
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "Medication for Mary Davis"
```

---

## Query 6: Diagnosis for Nancy Brown
**Expected Answer**: Gastroesophageal Reflux Disease
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "Diagnosis for Nancy Brown"
```

---

## Query 7: Medication for Kristen Warner
**Expected Answer**: Omeprazole 20mg
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "What was prescribed to Kristen Warner"
```

---

## Query 8: Diagnosis for Mr. Richard Smith
**Expected Answer**: Depression
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "What's Mr. Richard Smith's medical condition"
```

---

## Query 9: Medication for Jennifer Davis
**Expected Answer**: Amlodipine 5mg
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "Which drug was prescribed to Jennifer Davis"
```

---

## Query 10: Diagnosis for David Frye
**Expected Answer**: Anxiety Disorder
```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "What is the diagnosis for David Frye"
```

---

## All Commands (Copy All):

```bash
python.exe LLM_Diffrential_Privacy.py --query --prompt "What was prescribed to Sara Moon"
python.exe LLM_Diffrential_Privacy.py --query --prompt "What is the diagnosis for Donald Miller"
python.exe LLM_Diffrential_Privacy.py --query --prompt "What medication is Lindsey Thompson taking"
python.exe LLM_Diffrential_Privacy.py --query --prompt "What condition was Robert Smith diagnosed with"
python.exe LLM_Diffrential_Privacy.py --query --prompt "Medication for Mary Davis"
python.exe LLM_Diffrential_Privacy.py --query --prompt "Diagnosis for Nancy Brown"
python.exe LLM_Diffrential_Privacy.py --query --prompt "What was prescribed to Kristen Warner"
python.exe LLM_Diffrential_Privacy.py --query --prompt "What's Mr. Richard Smith's medical condition"
python.exe LLM_Diffrential_Privacy.py --query --prompt "Which drug was prescribed to Jennifer Davis"
python.exe LLM_Diffrential_Privacy.py --query --prompt "What is the diagnosis for David Frye"
```

---

## Quick Stats:
- **5 Medication Queries** (short answers: drug names)
- **5 Diagnosis Queries** (various conditions)
- **Mix of question styles** (like your training data)
- **All patients verified** from QA_pairs_20251017_062150.csv

---

## Note:
With F1=0.231 and EM=0.000, expect:
- ✅ Some partially correct answers
- ❌ Many incorrect or incomplete answers
- 🔄 Possibly repetitive text

**To improve**: Run your new 4000-record, 6-epoch training!

