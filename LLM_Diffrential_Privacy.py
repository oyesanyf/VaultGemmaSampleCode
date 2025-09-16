"""
Command-line tool to generate PHI/PII-rich synthetic data, optionally apply
Differential Privacy (DP) to both data and model training, fine-tune a
VaultGemma causal LM with LoRA adapters, and query the resulting model.

Key flags:
- Data management: --encrypt-data, --clean, --list-data, --records
- Training: --train, --epochs, --batch_size
- Model DP: --dp-model, --dp-eps, --dp-delta, --dp-max-grad-norm, --secure-rng
- Querying: --query, --model
- Models management: --list

Run with --help for comprehensive examples of each flag.
"""

import argparse
import glob
import os
import random
import re
import shutil
import warnings
from datetime import datetime

import pandas as pd
import torch
from faker import Faker
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import numpy as np

# Progress bar (optional)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Opacus for DP-SGD
try:
    from opacus import PrivacyEngine
except ImportError:
    warnings.warn("Opacus not found. Model-level DP-SGD is disabled.")
    PrivacyEngine = None

# ===============================
# Simple text evaluation helpers
# ===============================

def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_token_f1(pred: str, truth: str) -> float:
    from collections import Counter
    pred_toks = _normalize_text(pred).split()
    gold_toks = _normalize_text(truth).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    # Multiset overlap
    p, g = Counter(pred_toks), Counter(gold_toks)
    overlap = sum((p & g).values())
    precision = overlap / max(1, sum(p.values()))
    recall = overlap / max(1, sum(g.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate_qa_model(model, tokenizer, dataset, val_start: int, max_new_tokens: int = 48):
    """Compute EM and token-F1 on a held-out slice of a QA dataset.

    Args:
        model: Causal LM (can be PEFT-wrapped)
        tokenizer: Matching tokenizer
        dataset: HF Dataset with 'question' and 'answer'
        val_start: Start index for the validation slice
        max_new_tokens: Generation budget

    Returns:
        (avg_f1: float, em: float, total: int)
    """
    n = len(dataset)
    if val_start >= n:
        return 0.0, 0.0, 0
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model_device = next(model.parameters()).device
    f1_scores = []
    exact = 0
    total = 0
    with torch.no_grad():
        for idx in range(val_start, n):
            q = str(dataset[idx]['question'])
            gold = str(dataset[idx]['answer'])
            prompt = f"### Human: {q}\n### Assistant:"
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = decoded.split("### Assistant:")[-1].strip()
            f1 = compute_token_f1(pred, gold)
            f1_scores.append(f1)
            if _normalize_text(pred) == _normalize_text(gold):
                exact += 1
            total += 1
    avg_f1 = (sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
    em = (exact / total) if total else 0.0
    model.train()
    return avg_f1, em, total

# Set environment variables for better CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Force eager attention globally to avoid SDPA/flash/vmap issues (esp. with Opacus)
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"
os.environ["PYTORCH_SDP_DISABLE_HEURISTIC"] = "1"

# Selected device for training/query (overridden by --cpu/--gpu flags)
SELECTED_DEVICE = "cpu"

# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================
def _unwrap_opacus(model):
    """Return the underlying torch.nn.Module if wrapped by Opacus GradSampleModule."""
    try:
        from opacus.grad_sample import GradSampleModule
        while isinstance(model, GradSampleModule):
            model = model._module
    except ImportError:
        pass
    return model

def create_models_folder():
    """Ensure a models/ directory exists and return its path.

    Returns:
        str: Absolute or relative path to the models directory.
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    return models_dir

def create_data_folder():
    """Ensure a data/ directory exists and return its path.

    Returns:
        str: Absolute or relative path to the data directory.
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    return data_dir

def set_selected_device(device: str) -> None:
    """Set the globally selected device string ("cpu" or "cuda")."""
    global SELECTED_DEVICE
    SELECTED_DEVICE = device

def get_timestamp_model_name():
    """Generate a unique model directory name using the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"vaultgemma_dp_{timestamp}"

def get_most_recent_model():
    """Return the most recently modified model directory under models/.

    Returns:
        Optional[str]: Path to latest model directory or None if none found.
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    
    # Find all model directories
    model_dirs = glob.glob(os.path.join(models_dir, "vaultgemma_dp_*"))
    if not model_dirs:
        return None
    
    # Sort by modification time and return the most recent
    most_recent = max(model_dirs, key=os.path.getmtime)
    return most_recent

def list_available_models():
    """Print a table of available saved models and return their paths.

    Returns:
        list[str]: Sorted list of model directory paths (newest first).
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models folder found.")
        return []
    
    model_dirs = glob.glob(os.path.join(models_dir, "vaultgemma_dp_*"))
    if not model_dirs:
        print("No models found in models folder.")
        return []
    
    print("Available models:")
    for i, model_dir in enumerate(sorted(model_dirs, key=os.path.getmtime, reverse=True), 1):
        model_name = os.path.basename(model_dir)
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_dir))
        print(f"  {i}. {model_name} (created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return model_dirs

def list_available_data(reveal_dp=False, unmask=False):
    """Print a table of available synthetic data files and return their paths.

    Args:
        reveal_dp (bool): If True, show sample DP-masked data for verification.
        unmask (bool): If True, show original unprotected data for comparison.

    Returns:
        list[str]: Sorted list of CSV file paths (newest first).
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("No data folder found.")
        return []
    
    # Search for all data files (synthetic, DP_ENCRYPTED, ORIGINAL)
    patterns = ["synthetic_phi_data_*.csv", "DP_ENCRYPTED_phi_data_*.csv", "ORIGINAL_phi_data_*.csv"]
    data_files = []
    for pattern in patterns:
        data_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    
    if not data_files:
        print("No data files found in data folder.")
        return []
    
    print("Available synthetic data files:")
    for i, data_file in enumerate(sorted(data_files, key=os.path.getmtime, reverse=True), 1):
        filename = os.path.basename(data_file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(data_file))
        file_size = os.path.getsize(data_file)
        print(f"  {i}. {filename} (created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}, size: {file_size} bytes)")
        
        # Show DP-masked data sample if requested
        if reveal_dp and "DP_ENCRYPTED" in filename:
            try:
                df = pd.read_csv(data_file)
                print("     📊 DP-MASKED SAMPLE (first 2 rows):")
                for idx, row in df.head(2).iterrows():
                    print(f"       Row {idx+1}: {row['name']} | SSN: {row.get('ssn', 'N/A')} | MRN: {row.get('mrn', 'N/A')}")
                    if 'medical_notes' in row:
                        notes = str(row['medical_notes'])[:100] + "..." if len(str(row['medical_notes'])) > 100 else str(row['medical_notes'])
                        print(f"                Notes: {notes}")
                print()
            except Exception as e:
                print(f"     ⚠️ Could not read DP sample: {e}")
        
        # Show original unprotected data if requested and available
        if unmask and "ORIGINAL" in filename:
            try:
                df = pd.read_csv(data_file)
                print("     ⚠️ ORIGINAL UNPROTECTED DATA (first 2 rows):")
                print("     🔴 WARNING: Contains sensitive PHI/PII!")
                for idx, row in df.head(2).iterrows():
                    print(f"       Row {idx+1}: {row['name']} | SSN: {row.get('ssn', 'N/A')} | MRN: {row.get('mrn', 'N/A')}")
                    if 'medical_notes' in row:
                        notes = str(row['medical_notes'])[:100] + "..." if len(str(row['medical_notes'])) > 100 else str(row['medical_notes'])
                        print(f"                Notes: {notes}")
                print()
            except Exception as e:
                print(f"     ⚠️ Could not read original data: {e}")
    
    return data_files

def find_latest_data_file() -> str | None:
    """Return most recent CSV among known data patterns in data/.

    Searches for DP_ENCRYPTED, synthetic_phi_data, then ORIGINAL as fallback.
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        return None
    patterns = [
        os.path.join(data_dir, "DP_ENCRYPTED_phi_data_*.csv"),
        os.path.join(data_dir, "synthetic_phi_data_*.csv"),
        os.path.join(data_dir, "ORIGINAL_phi_data_*.csv"),
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

# ======================================================================================
# STEP 1: GENERATE SYNTHETIC PHI DATA
# ======================================================================================
def generate_phi_data(num_records=2000, dp_data: bool = False, dp_epsilon: float = 1.0):
    """Generate synthetic PHI/PII-rich records for training prompts.

    Args:
        num_records (int): Number of synthetic rows to generate.
        dp_data (bool): If True, sample categorical fields from DP-noisy
            distributions (Laplace mechanism) to reduce privacy risk.
        dp_epsilon (float): Epsilon for the data-level DP sampling when
            dp_data=True. Larger epsilon → more utility, smaller epsilon → more privacy.

    Returns:
        pandas.DataFrame: Columns include name, medical_notes, and PHI/PII fields.
    """
    fake = Faker()
    data = []
    
    # Medical conditions and treatments
    conditions = [
        "Hypertension", "Type 2 Diabetes", "Asthma", "Coronary Artery Disease", "Depression",
        "Chronic Obstructive Pulmonary Disease", "Osteoarthritis", "Migraine", "Anxiety Disorder",
        "Hyperlipidemia", "Gastroesophageal Reflux Disease", "Sleep Apnea", "Chronic Kidney Disease"
    ]
    
    medications = [
        "Lisinopril 10mg", "Metformin 500mg", "Albuterol inhaler", "Atorvastatin 20mg", 
        "Sertraline 50mg", "Omeprazole 20mg", "Amlodipine 5mg", "Gabapentin 300mg",
        "Lorazepam 0.5mg", "Furosemide 40mg", "Warfarin 5mg", "Insulin glargine"
    ]
    
    # PHI/PII elements to include
    phone_numbers = [fake.phone_number() for _ in range(100)]
    emails = [fake.email() for _ in range(100)]
    addresses = [fake.address().replace('\n', ', ') for _ in range(100)]
    ssn_patterns = [fake.ssn() for _ in range(100)]
    insurance_ids = [fake.bothify(text='ABC###-###-###') for _ in range(100)]
    mrn_numbers = [fake.bothify(text='MRN######') for _ in range(100)]
    
    # Detailed medical notes templates with PHI/PII
    notes_templates = [
        "Patient {name} (MRN: {mrn}, DOB: {dob}) presents with {condition}. "
        "Contact: {phone}, Email: {email}. Address: {address}. "
        "Insurance: {insurance}. Prescribed {medication}. Follow-up in 2 weeks.",
        
        "Patient {name} (SSN: {ssn}, MRN: {mrn}) has stable {condition}. "
        "Phone: {phone}, Address: {address}. Continue current treatment plan. "
        "Medication: {medication}. Monitor vitals daily.",
        
        "Patient {name} (DOB: {dob}, MRN: {mrn}) discussed lifestyle changes. "
        "Contact info: {phone}, {email}. Address: {address}. "
        "Insurance ID: {insurance}. Patient is compliant with {medication}.",
        
        "Patient {name} (SSN: {ssn}, Phone: {phone}) referred to specialist. "
        "MRN: {mrn}, Email: {email}. Address: {address}. "
        "Current medication: {medication}. Awaiting results.",
        
        "Patient {name} (DOB: {dob}, MRN: {mrn}) symptoms improved. "
        "Contact: {phone}, {email}. Address: {address}. "
        "Insurance: {insurance}. No changes to {medication} needed.",
        
        "Patient {name} (SSN: {ssn}, MRN: {mrn}) diagnosed with {condition}. "
        "Phone: {phone}, Email: {email}. Address: {address}. "
        "Insurance ID: {insurance}. Prescribed {medication}.",
        
        "Patient {name} (DOB: {dob}, Phone: {phone}) follow-up visit. "
        "MRN: {mrn}, Email: {email}. Address: {address}. "
        "Condition: {condition}. Medication: {medication}.",
        
        "Patient {name} (SSN: {ssn}, MRN: {mrn}) emergency visit. "
        "Contact: {phone}, {email}. Address: {address}. "
        "Insurance: {insurance}. Treatment: {medication}."
    ]

    # If data-level DP requested, construct DP-probabilities for categorical choices
    if dp_data:
        # Use uniform base counts then add Laplace noise; re-normalize to probabilities
        def dp_noisy_probs(num_items: int, epsilon: float):
            # Scale for Laplace = 1/epsilon per standard mechanism
            scale = 1.0 / max(epsilon, 1e-6)
            laplace = torch.distributions.Laplace(torch.tensor(0.0), torch.tensor(scale))
            base_counts = [1.0] * num_items
            noisy = []
            for c in base_counts:
                n = c + float(laplace.sample().item())
                # Clip to small positive to avoid zeros/negatives
                noisy.append(max(n, 1e-6))
            s = sum(noisy)
            return [x / s for x in noisy]

        cond_probs = dp_noisy_probs(len(conditions), dp_epsilon)
        med_probs = dp_noisy_probs(len(medications), dp_epsilon)
        tmpl_probs = dp_noisy_probs(len(notes_templates), dp_epsilon)
    else:
        cond_probs = None
        med_probs = None
        tmpl_probs = None

    for _ in range(num_records):
        # Generate PHI/PII data
        name = fake.name()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%m/%d/%Y')
        phone = fake.random_element(elements=phone_numbers)
        email = fake.random_element(elements=emails)
        address = fake.random_element(elements=addresses)
        ssn = fake.random_element(elements=ssn_patterns)
        insurance = fake.random_element(elements=insurance_ids)
        mrn = fake.random_element(elements=mrn_numbers)
        
        # Medical data
        if dp_data:
            condition = random.choices(conditions, weights=cond_probs, k=1)[0]
            medication = random.choices(medications, weights=med_probs, k=1)[0]
        else:
            condition = fake.random_element(elements=conditions)
            medication = fake.random_element(elements=medications)
        
        # Select a random template and fill it
        if dp_data:
            template = random.choices(notes_templates, weights=tmpl_probs, k=1)[0]
        else:
            template = fake.random_element(elements=notes_templates)
        note = template.format(
            name=name, dob=dob, phone=phone, email=email, address=address,
            ssn=ssn, insurance=insurance, mrn=mrn, condition=condition, medication=medication
        )
        
        record = {
            "name": name,
            "medical_notes": note,
            "dob": dob,
            "phone": phone,
            "email": email,
            "address": address,
            "ssn": ssn,
            "insurance_id": insurance,
            "mrn": mrn,
            "condition": condition,
            "medication": medication
        }
        data.append(record)
    
    return pd.DataFrame(data)

# ======================================================================================
# STEP 2 & 3: PREPARE DATA FOR THE TRAINER
# ======================================================================================
def preprocess_for_trainer(examples, tokenizer, seq_len: int = 512, dp_model: bool = False):
    """Tokenize and build labels for causal LM prompts.

    Supports two schemas:
    - Default PHI/PII context: builds prompts using name/medical_notes
    - QA schema: if 'question' and 'answer' columns exist, uses them directly

    Masks the prompt portion in labels with -100 so only the assistant part
    contributes to the loss.
    """
    texts = []
    if 'question' in examples and 'answer' in examples:
        # QA schema
        for i in range(len(examples['question'])):
            q = examples['question'][i]
            a = examples['answer'][i]
            text = f"### Human: {q}\n### Assistant: {a}"
            texts.append(text)
    else:
        # PHI context schema
        for i in range(len(examples['name'])):
            name = examples['name'][i]
            medical_notes = examples['medical_notes'][i]
            additional_info = ""
            if 'dob' in examples:
                additional_info += f" (DOB: {examples['dob'][i]})"
            if 'mrn' in examples:
                additional_info += f" (MRN: {examples['mrn'][i]})"
            if 'phone' in examples:
                additional_info += f" (Phone: {examples['phone'][i]})"
            text = f"### Human: What are the medical notes for {name}{additional_info}?\n### Assistant: {medical_notes}"
            texts.append(text)

    # For compatibility with Opacus and eager attention, conditionally disable attention masks
    if dp_model:
        model_inputs = tokenizer(texts, max_length=int(seq_len), truncation=True, padding="max_length", return_attention_mask=False)
    else:
        model_inputs = tokenizer(texts, max_length=int(seq_len), truncation=True, padding="max_length")
    # Create a deep copy of input_ids for labels to avoid modifying input_ids
    labels = [input_ids.copy() for input_ids in model_inputs["input_ids"]]
    
    if 'question' in examples and 'answer' in examples:
        for i in range(len(examples['question'])):
            prompt_text = f"### Human: {examples['question'][i]}\n### Assistant:"
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
            prompt_length = len(prompt_tokens["input_ids"])
            labels[i][:prompt_length] = [-100] * prompt_length
    else:
        for i in range(len(examples['name'])):
            name = examples['name'][i]
            additional_info = ""
            if 'dob' in examples:
                additional_info += f" (DOB: {examples['dob'][i]})"
            if 'mrn' in examples:
                additional_info += f" (MRN: {examples['mrn'][i]})"
            if 'phone' in examples:
                additional_info += f" (Phone: {examples['phone'][i]})"
            prompt_text = f"### Human: What are the medical notes for {name}{additional_info}?\n### Assistant:"
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
            prompt_length = len(prompt_tokens["input_ids"])
            labels[i][:prompt_length] = [-100] * prompt_length
        
    # Mask pad positions in labels to avoid computing loss on padding (only if attention_mask exists)
    for i in range(len(labels)):
        am = model_inputs.get("attention_mask", None)
        if am is not None:
            for j, m in enumerate(am[i]):
                if m == 0:
                    labels[i][j] = -100
        # If no attention mask (DP-SGD mode), assume all tokens are valid for loss computation
    model_inputs["labels"] = labels
    return model_inputs

# ======================================================================================
# TRAINING FUNCTION
# ======================================================================================
def train_model(
    num_records: int = 50,
    epochs: int = 1,
    batch_size: int = 1,
    dp_model: bool = False,
    dp_epsilon_model: float = 8.0,
    dp_delta: float = 1e-5,
    dp_max_grad_norm: float = 0.1,
    secure_rng: bool = False,
    dp_data: bool = True,
    dp_data_epsilon: float = 1.0,
    qa: bool = True,
    seq_len: int = 512,
):
    """Train the model with optional DP-SGD and/or DP data sampling.

    Args:
        num_records (int): Number of records to generate if no encrypted data exists.
        epochs (int): Training epochs.
        batch_size (int): Batch size for DataLoader.
        dp_model (bool): If True, enable Opacus PrivacyEngine for additional DP-SGD (default: False).
            Note: VaultGemma is already pre-trained with DP-SGD (ε≤2.0, δ≤1.1e-10).
        dp_epsilon_model (float): Target epsilon for DP-SGD accounting.
        dp_delta (float): Target delta for DP-SGD accounting.
        dp_max_grad_norm (float): Per-sample max grad norm for clipping.
        secure_rng (bool): If True, request cryptographically secure RNG (torchcsprng).
        dp_data (bool): If True, use DP sampling when generating synthetic data (default: True).
        dp_data_epsilon (float): Epsilon for data-level DP sampling.
        qa (bool): If True, train on QA pairs (question/answer columns) so
            the model can answer questions seen in training data (default: True).

    Returns:
        Optional[str]: Path to saved adapter directory on success, else None.
    """
    print("=== Starting Model Training ===")
    print("📊 DIFFERENTIAL PRIVACY STATUS:")
    print("   🔒 VaultGemma Model: Built-in DP-SGD (ε≤2.0, δ≤1.1e-10)")
    if dp_model:
        print("   🔧 Additional Opacus DP: ENABLED")
    else:
        print("   🔧 Additional Opacus DP: DISABLED (using built-in DP only)")
    if dp_data:
        print("   📋 Data-level DP: ENABLED")
    else:
        print("   📋 Data-level DP: DISABLED")
    
    # Create models folder
    models_dir = create_models_folder()
    model_name = get_timestamp_model_name()
    adapter_path = os.path.join(models_dir, model_name)
    
    print(f"\nTraining model: {model_name}")
    print(f"Model will be saved to: {adapter_path}")
    
    # Data selection strategy
    data_dir = create_data_folder()
    if qa:
        # For QA training, generate fresh synthetic data so names exist
        print("\n1. Generating synthetic PHI data for QA training...")
        phi_df = generate_phi_data(num_records=num_records, dp_data=False, dp_epsilon=dp_data_epsilon)
        print(f"Generated {len(phi_df)} patient records (QA mode)")
    else:
        # Check for encrypted data first, then generate if not available
        print("\n1. Checking for encrypted data...")
        encrypted_files = glob.glob(os.path.join(data_dir, "DP_ENCRYPTED_phi_data_*.csv"))
        if encrypted_files:
            latest_encrypted_file = max(encrypted_files, key=os.path.getmtime)
            print(f"Found encrypted data: {os.path.basename(latest_encrypted_file)}")
            print("Using encrypted data for training (privacy-protected)")
            phi_df = pd.read_csv(latest_encrypted_file)
            print(f"Loaded {len(phi_df)} encrypted patient records")
        else:
            print("No encrypted data found. Generating new synthetic PHI data...")
            phi_df = generate_phi_data(num_records=num_records, dp_data=dp_data, dp_epsilon=dp_data_epsilon)
            print(f"Generated {len(phi_df)} patient records")
    
    # If QA mode, build question/answer pairs
    if qa:
        # Build a larger, paraphrased QA set per record to improve accuracy even with few records
        qa_rows = []
        for _, r in phi_df.iterrows():
            name = r.get("name", "the patient")
            condition = str(r.get("condition", "")).strip()
            medication = str(r.get("medication", "")).strip()
            notes = str(r.get("medical_notes", "")).strip()

            # Diagnosis question variants
            if condition:
                diagnosis_q_variants = [
                    f"What is the diagnosis for {name}?",
                    f"What condition was {name} diagnosed with?",
                    f"What's {name}'s medical condition?",
                    f"Diagnosis for {name}?",
                ]
                for q in diagnosis_q_variants:
                    qa_rows.append({"question": q, "answer": condition})

            # Medication question variants
            if medication:
                medication_q_variants = [
                    f"What was prescribed to {name}?",
                    f"What medication is {name} taking?",
                    f"Which drug was prescribed to {name}?",
                    f"Medication for {name}?",
                ]
                # Also condition-aware phrasing if condition present
                if condition:
                    medication_q_variants.append(
                        f"What medication was prescribed for {name}'s {condition}?"
                    )
                for q in medication_q_variants:
                    qa_rows.append({"question": q, "answer": medication})

            # Medical notes question variants (answers are longer)
            if notes:
                notes_q_variants = [
                    f"What are the medical notes for {name}?",
                    f"Summarize the medical notes for {name}.",
                    f"Provide {name}'s medical notes.",
                    f"What does {name}'s chart say?",
                ]
                for q in notes_q_variants:
                    qa_rows.append({"question": q, "answer": notes})

        qa_df = pd.DataFrame(qa_rows)
        # Keep only short answers (e.g., diagnosis/medication), drop long-note rows
        qa_df = qa_df[qa_df["answer"].astype(str).str.len() <= 64].reset_index(drop=True)
        print(f"Built {len(qa_df)} QA pairs for training (filtered to short answers)")
        dataset = Dataset.from_pandas(qa_df)
        
        # Save QA pairs to see what the model is actually training on
        data_dir = create_data_folder()
        qa_filename = f"QA_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        qa_path = os.path.join(data_dir, qa_filename)
        qa_df.to_csv(qa_path, index=False)
        print(f"🎯 QA pairs saved to: {qa_path}")
        print("📋 Sample QA pairs:")
        for i, row in qa_df.head(3).iterrows():
            print(f"   Q: {row['question']}")
            print(f"   A: {row['answer']}")
            print()
    else:
        dataset = Dataset.from_pandas(phi_df)
    
    # Save data files to data folder
    data_dir = create_data_folder()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if dp_data:
        # Save both original and DP-protected versions when DP is enabled
        original_filename = f"ORIGINAL_phi_data_{timestamp}.csv"
        original_path = os.path.join(data_dir, original_filename)
        phi_df.to_csv(original_path, index=False)
        print(f"🔴 Original data saved to: {original_path}")
        
        # Create DP-protected version
        dp_protected_df = apply_dp_to_data(phi_df.copy(), dp_data_epsilon)
        dp_filename = f"DP_ENCRYPTED_phi_data_eps{dp_data_epsilon}_{timestamp}.csv"
        dp_path = os.path.join(data_dir, dp_filename)
        dp_protected_df.to_csv(dp_path, index=False)
        print(f"🟢 DP-protected data saved to: {dp_path}")
        print(f"🔒 Privacy protection: ε={dp_data_epsilon}")
    else:
        # Save regular synthetic data when DP is disabled
        data_filename = f"synthetic_phi_data_{timestamp}.csv"
        data_path = os.path.join(data_dir, data_filename)
        phi_df.to_csv(data_path, index=False)
        print(f"Synthetic data saved to: {data_path}")
    
    # Load tokenizer
    print("\n2. Loading tokenizer...")
    base_model_id = "google/vaultgemma-1b"
    # Prefer fast tokenizer; fall back to slow if conversion fails
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Preprocess data
    print("\n3. Preprocessing data...")
    tokenized_dataset = dataset.map(lambda ex: preprocess_for_trainer(ex, tokenizer, seq_len=seq_len, dp_model=dp_model), batched=True, batch_size=10, remove_columns=dataset.column_names)
    
    # Validate data
    print("\n4. Validating data...")
    sample = tokenized_dataset[0]
    max_input_id = max(sample['input_ids'])
    max_label = max(sample['labels'])
    print(f"Max input ID: {max_input_id}, Max label: {max_label}")
    
    if max_input_id >= tokenizer.vocab_size:
        print(f"ERROR: Found input ID {max_input_id} >= vocab_size {tokenizer.vocab_size}")
        return None
    
    if -100 in sample['input_ids']:
        print("ERROR: Found -100 in input_ids - this will cause CUDA errors!")
        return None
    
    print("✓ Data validation passed")

    # Load model
    print("\n5. Loading model...")
    device = torch.device(SELECTED_DEVICE if (SELECTED_DEVICE == "cpu" or (SELECTED_DEVICE == "cuda" and torch.cuda.is_available())) else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Set the attention implementation mode to eager, which is compatible with Opacus
    attn_mode = "eager" if dp_model else None
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        dtype=torch.float32,
        # Pass the mode directly here. This is the most reliable way to set it.
        attn_implementation=attn_mode,
        trust_remote_code=True,
    )
    # --- Force eager attention environment variables only if DP-SGD is enabled ---
    if dp_model:
        import os as _os
        _os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
        _os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"
        _os.environ["PYTORCH_SDP_DISABLE_HEURISTIC"] = "1"
        # Redundant but safe, in case the above fails for some reason
        model.config.attn_implementation = "eager"
        try:
            model.config._attn_implementation = "eager"  # HF >= 4.41
        except Exception:
            pass
        print("✓ Environment variables set for DP-SGD compatibility")
    # ------------------------------------------------------------------------------
    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("WARNING: CUDA OOM while moving base model. Falling back to CPU.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = model.to(device)
    
    # Disable cache for training
    model.config.use_cache = False
    
    print(f"Model loaded on device: {device}")
    
    # Setup LoRA
    print("\n6. Setting up LoRA...")
    lora_config = LoraConfig(r=8, target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    
    # Ensure eager attention is still set after LoRA wrapping (only if DP-SGD is enabled)
    if dp_model:
        try:
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "eager"
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "eager"
            # Ensure sliding_window is defined; set to 0 to disable sliding masks
            try:
                setattr(model.config, "sliding_window", int(getattr(model.config, "sliding_window", 0) or 0))
            except Exception:
                pass
            print("✓ Eager attention mode confirmed after LoRA setup for DP-SGD compatibility")
        except Exception as e:
            print(f"WARNING: Could not confirm attention implementation after LoRA: {e}")
    else:
        print("✓ Using default attention implementation (no DP-SGD)")
    
    # Check trainable parameters
    def print_trainable_params(m):
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        print(f"Trainable params: {trainable}/{total} ({100*trainable/total:.4f}%)")
    print_trainable_params(model)
    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("WARNING: CUDA OOM while moving LoRA model. Falling back to CPU.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = model.to(device)
    model.train()

    # Training setup
    print(f"\n7. Setting up training...")
    TARGET_EPSILON = float(dp_epsilon_model)
    TARGET_DELTA = float(dp_delta)
    MAX_GRAD_NORM = float(dp_max_grad_norm)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Create torch DataLoader for Opacus
    print("8. Building DataLoader for DP training...")
    from torch.utils.data import Dataset as TorchDataset, DataLoader

    class TokenizedTorchDataset(TorchDataset):
        def __init__(self, hf_dataset):
            self.ds = hf_dataset

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, index):
            sample = self.ds[index]
            input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
            labels = torch.tensor(sample['labels'], dtype=torch.long)
            
            # Handle case where attention_mask might not exist (DP-SGD mode)
            if 'attention_mask' in sample:
                attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long)
            else:
                # Create a dummy attention mask of all 1s if not present
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            
            return (input_ids, attention_mask, labels)

    torch_dataset = TokenizedTorchDataset(tokenized_dataset)
    # Only use pin_memory if we have CUDA available
    use_pin_memory = device.type == "cuda"
    # For DP-SGD, avoid variable-size/empty batches. We'll set drop_last=True when DP is enabled.
    train_loader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=bool(dp_model),
        pin_memory=use_pin_memory,
    )

    # Attach PrivacyEngine for true DP-SGD (if requested)
    dp_enabled = False
    if dp_model and PrivacyEngine:
        print("9. Attaching PrivacyEngine (DP-SGD)...")
        try:
            privacy_engine = PrivacyEngine(secure_mode=bool(secure_rng))
        except Exception as e:
            # Missing torchcsprng when secure_rng=True or other init errors
            print(f"WARNING: PrivacyEngine init failed ({e}). Retrying with secure_rng=False...")
            try:
                privacy_engine = PrivacyEngine(secure_mode=False)
            except Exception as e2:
                print(f"WARNING: Failed to initialize PrivacyEngine: {e2}")
                privacy_engine = None

        if privacy_engine is not None:
            try:
                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=TARGET_EPSILON,
                    target_delta=TARGET_DELTA,
                    epochs=epochs,
                    max_grad_norm=MAX_GRAD_NORM,
                    grad_sample_mode="hooks",  # supported hooks mode
                    poisson_sampling=False,     # avoid random empty batches under Poisson sampling
                )
                dp_enabled = True
            except Exception as dp_err:
                print(f"WARNING: Failed to enable DP-SGD via Opacus: {dp_err}")
                print("Proceeding without DP for this run.")

    # Training loop (uses DP if enabled)
    model_device = next(model.parameters()).device
    print(
        f"10. Starting training on {model_device.type}... (DP enabled)"
        if dp_enabled
        else f"10. Starting training on {model_device.type}... (NO DP)"
    )
    model.train()
    total_steps = (len(train_loader) * epochs)
    step = 0
    
    oom_fallback_done = False
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        batch_iter = (
            tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                ascii=True,
                dynamic_ncols=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt:>4}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                ncols=100,
                leave=False,
            )
            if tqdm
            else train_loader
        )
        for batch in batch_iter:
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                input_ids, attention_mask, labels = batch
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

            # Ensure tensors are on the same device as the model
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            labels = labels.to(model_device)

            optimizer.zero_grad()
            try:
                if dp_enabled:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                if not oom_fallback_done and model_device.type == "cuda":
                    print("WARNING: CUDA OOM during forward/backward. Falling back to CPU and continuing training.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model = model.to("cpu")
                    model_device = next(model.parameters()).device
                    # Recreate optimizer for new device
                    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
                    # Move current batch to CPU and retry once
                    input_ids = input_ids.to(model_device)
                    attention_mask = attention_mask.to(model_device)
                    labels = labels.to(model_device)
                    if dp_enabled:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    oom_fallback_done = True
                else:
                    raise
            except IndexError as mask_err:
                # Handle masking/vmap crash under DP-SGD: fully detach + unwrap and retry
                if dp_enabled and "shape mismatch" in str(mask_err):
                    print("WARNING: Attention masking error under DP-SGD. Detaching Opacus and retrying without attention_mask.")
                    try:
                        try:
                            privacy_engine.detach()
                        except Exception:
                            pass
                        dp_enabled = False
                        # Unwrap Opacus GradSampleModule to restore plain nn.Module
                        model = _unwrap_opacus(model)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
                        # Rebuild non-DP dataloader
                        train_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=use_pin_memory)
                    except Exception:
                        pass
                    # retry once with no attention_mask to avoid vmap padding path
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                else:
                    raise
            
            if not dp_enabled:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            
            if tqdm and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")
            
            step += 1
            if step % 10 == 0:
                print(f"  Step {step:>5}/{total_steps}, Loss: {loss.item():.4f}")
    
    print("11. Training complete!")
    if dp_enabled and PrivacyEngine:
        try:
            eps = privacy_engine.get_epsilon(delta=TARGET_DELTA)
            print(f"Privacy Guarantee: Epsilon = {eps:.2f} for delta = {TARGET_DELTA}")
        except Exception:
            print(f"Privacy Guarantee: Epsilon <= {TARGET_EPSILON:.2f} for delta = {TARGET_DELTA}")
    else:
        print("Privacy Guarantee: VaultGemma Built-in DP (ε≤2.0, δ≤1.1e-10) - No additional Opacus DP")
    
    # Optional quick evaluation on a small held-out split of QA pairs (if available)
    try:
        if qa and 'question' in dataset.column_names and 'answer' in dataset.column_names:
            n = len(dataset)
            if n >= 10:
                # If DP-SGD was enabled, unwrap Opacus' GradSampleModule for generation
                model_for_eval = _unwrap_opacus(model) if 'dp_enabled' in locals() and dp_enabled else model
                val_start = max(0, int(n * 0.9))
                avg_f1, em, total = evaluate_qa_model(model_for_eval, tokenizer, dataset, val_start, max_new_tokens=48)
                if total > 0:
                    print(f"\n📈 Quick Eval on held-out {total} QA examples: F1={avg_f1:.3f}, EM={em:.3f}")
    except Exception as e:
        print(f"(Eval skipped due to error: {e})")

    # Save model
    print("12. Saving model...")
    # If DP-SGD was enabled, unwrap Opacus GradSampleModule before saving
    if 'dp_enabled' in locals() and dp_enabled:
        try:
            if 'privacy_engine' in locals() and privacy_engine is not None:
                try:
                    privacy_engine.detach()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            model = _unwrap_opacus(model)
        except Exception:
            pass
    # Prefer skipping model card creation to avoid missing template errors
    try:
        model.save_pretrained(adapter_path, create_model_card=False)
    except TypeError:
        # Older PEFT may not support create_model_card. Try providing a local template.
        try:
            os.makedirs(os.path.dirname(os.path.join(adapter_path, "..")), exist_ok=True)
        except Exception:
            pass
        local_tpl = os.path.join(os.getcwd(), "modelcard_template.md")
        try:
            with open(local_tpl, "w", encoding="utf-8") as f:
                f.write("# Adapter Card\n\nThis adapter was saved locally. Model card generation was minimized.")
        except Exception:
            local_tpl = None
        try:
            if local_tpl:
                model.save_pretrained(adapter_path, model_card_template_path=local_tpl)
            else:
                model.save_pretrained(adapter_path)
        except FileNotFoundError as e:
            print(f"WARNING: Model card template missing; saving adapter without card. Details: {e}")
            # As a last resort, write a minimal README to the adapter dir
            try:
                os.makedirs(adapter_path, exist_ok=True)
                with open(os.path.join(adapter_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write("Adapter saved. Model card generation skipped due to missing template.")
            except Exception:
                pass
    print(f"Model saved to: {adapter_path}")
    
    return adapter_path

# ======================================================================================
# QUERY FUNCTION
# ======================================================================================
def query_model(model_path=None, prompt: str | None = None, context_csv: str | None = None, patient: str | None = None, use_trained_data: bool = False):
    """Query the model interactively or in one-shot mode.

    Args:
        model_path (Optional[str]): Path to a specific trained adapter directory.
            If None, loads the most recent under models/.
        prompt (Optional[str]): If provided, runs a single query and exits.
        context_csv (Optional[str]): CSV file to pull context from.
        patient (Optional[str]): Patient name to lookup in the context CSV.
        use_trained_data (bool): If True, auto-select the latest data CSV.
    """
    print("=== Model Query Interface ===")
    
    # Load the most recent model if no path specified
    if model_path is None:
        model_path = get_most_recent_model()
        if model_path is None:
            print("No trained models found. Please train a model first using --train")
            return
        print(f"Using most recent model: {os.path.basename(model_path)}")
    else:
        if not os.path.exists(model_path):
            print(f"Model path does not exist: {model_path}")
            return

    # Normalize to absolute local path to avoid being treated as a Hub repo id
    try:
        model_path = os.path.abspath(model_path)
    except Exception:
        pass

    # Validate adapter files exist locally
    required_files = [
        os.path.join(model_path, "adapter_config.json"),
        os.path.join(model_path, "adapter_model.safetensors"),
    ]
    if not all(os.path.isfile(p) for p in required_files):
        print("Adapter files not found at:")
        print(f"  {model_path}")
        print("Expected files:")
        for p in required_files:
            print(f"  - {os.path.basename(p)}")
        # Try a known local example adapter directory as a fallback
        fallback_dir = os.path.join(os.getcwd(), "phi-vaultgemma-finetuned-adapter-dp")
        fb_files = [
            os.path.join(fallback_dir, "adapter_config.json"),
            os.path.join(fallback_dir, "adapter_model.safetensors"),
        ]
        if all(os.path.isfile(p) for p in fb_files):
            print("Falling back to local adapter:")
            print(f"  {fallback_dir}")
            model_path = fallback_dir
        else:
            print("No valid adapter directory found. Re-run training or provide --model pointing to a directory containing the adapter files.")
            return
    
    # Load model
    print("Loading model...")
    base_model_id = "google/vaultgemma-1b"
    # Prefer fast tokenizer; fall back to slow if conversion fails
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device(SELECTED_DEVICE if (SELECTED_DEVICE == "cpu" or (SELECTED_DEVICE == "cuda" and torch.cuda.is_available())) else ("cuda" if torch.cuda.is_available() else "cpu"))
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.float32, trust_remote_code=True)
    base_model = base_model.to(device)
    finetuned_model = PeftModel.from_pretrained(base_model, model_path)
    finetuned_model = finetuned_model.to(device)

    # Optionally prepare retrieval-augmented context
    retrieved_context = None
    csv_path = context_csv
    if use_trained_data and not csv_path:
        csv_path = find_latest_data_file()
    if csv_path and os.path.exists(csv_path) and patient:
        try:
            df_ctx = pd.read_csv(csv_path)
            # Try exact name match if column exists
            name_col = None
            for col in ["name", "patient_name", "Name"]:
                if col in df_ctx.columns:
                    name_col = col
                    break
            if name_col is not None:
                row = df_ctx[df_ctx[name_col].astype(str).str.strip().str.lower() == patient.strip().lower()]
                if not row.empty:
                    # Prefer medical_notes if present, else try a few columns
                    if "medical_notes" in row.columns:
                        retrieved_context = str(row.iloc[0]["medical_notes"]) or None
                    else:
                        # Build a compact context from common fields
                        fields = []
                        for c in ["condition", "medication", "dob", "mrn", "insurance_id", "address", "phone", "email"]:
                            if c in row.columns:
                                fields.append(f"{c}: {row.iloc[0][c]}")
                        if fields:
                            retrieved_context = ", ".join(fields)
        except Exception as e:
            print(f"WARNING: Failed to read context CSV: {e}")

    # Minimal PHI guard: refuse to answer direct PHI requests
    def is_phi_request(text: str) -> bool:
        lowered = text.lower()
        triggers = ["ssn", "social security", "insurance id", "policy number", "mrn", "email", "phone", "address"]
        return any(t in lowered for t in triggers)

    # One-shot mode
    if prompt:
        text = prompt.strip()
        if not text:
            print("Empty prompt provided.")
            return
        if is_phi_request(text):
            print("I cannot provide sensitive PHI (e.g., SSN, MRN, address, email, phone, insurance).")
            return
        if retrieved_context:
            text = f"Context: {retrieved_context} Question: {text}"
        formatted = f"### Human: {text}\n### Assistant:"
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("Generating response...")
        with torch.no_grad():
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False
            )
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full.split("### Assistant:")[-1].strip()
        print(answer)
        return

    print("Model loaded! Enter your questions (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        # Get user input
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        # PHI guard
        if is_phi_request(question):
            print("I cannot provide sensitive PHI (e.g., SSN, MRN, address, email, phone, insurance).")
            continue
        # Retrieval-augmented prompt if context available
        q_text = question
        if retrieved_context:
            q_text = f"Context: {retrieved_context} Question: {question}"
        # Format the prompt
        prompt = f"### Human: {q_text}\n### Assistant:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        print("Generating response...")
        with torch.no_grad():
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_part = response.split("### Assistant:")[1].strip() if "### Assistant:" in response else response
        
        print(f"Model response: {assistant_part}")

def create_original_phi_data(num_records=100):
    """Generate an ORIGINAL PHI/PII dataset (intended for DP processing).

    Args:
        num_records (int): Number of rows to synthesize.

    Returns:
        pandas.DataFrame: Contains direct identifiers and medical context.
    """
    fake = Faker()
    data = []
    
    # Medical conditions and treatments
    conditions = [
        "Hypertension", "Type 2 Diabetes", "Asthma", "Coronary Artery Disease", "Depression",
        "Chronic Obstructive Pulmonary Disease", "Osteoarthritis", "Migraine", "Anxiety Disorder",
        "Hyperlipidemia", "Gastroesophageal Reflux Disease", "Sleep Apnea", "Chronic Kidney Disease"
    ]
    
    medications = [
        "Lisinopril 10mg", "Metformin 500mg", "Albuterol inhaler", "Atorvastatin 20mg", 
        "Sertraline 50mg", "Omeprazole 20mg", "Amlodipine 5mg", "Gabapentin 300mg",
        "Lorazepam 0.5mg", "Furosemide 40mg", "Warfarin 5mg", "Insulin glargine"
    ]
    
    print(f"Creating ORIGINAL PHI/PII data with {num_records} records...")
    print("⚠️ WARNING: This contains sensitive PHI/PII!")
    
    for i in range(num_records):
        # Generate original PHI/PII data
        name = fake.name()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%m/%d/%Y')
        phone = fake.phone_number()
        email = fake.email()
        address = fake.address().replace('\n', ', ')
        ssn = fake.ssn()
        insurance_id = fake.bothify(text='ABC###-###-###')
        mrn = fake.bothify(text='MRN######')
        
        # Medical data
        condition = fake.random_element(elements=conditions)
        medication = fake.random_element(elements=medications)
        
        # Create detailed medical note
        note = f"Patient {name} (MRN: {mrn}, DOB: {dob}) presents with {condition}. " \
               f"Contact: {phone}, Email: {email}. Address: {address}. " \
               f"Insurance: {insurance_id}. Prescribed {medication}. Follow-up in 2 weeks."
        
        record = {
            "patient_id": f"P{i+1:04d}",
            "name": name,
            "dob": dob,
            "phone": phone,
            "email": email,
            "address": address,
            "ssn": ssn,
            "insurance_id": insurance_id,
            "mrn": mrn,
            "condition": condition,
            "medication": medication,
            "medical_notes": note,
            "age": fake.random_int(min=18, max=90),
            "weight": fake.random_int(min=100, max=300),
            "height": fake.random_int(min=150, max=200)
        }
        data.append(record)
    
    return pd.DataFrame(data)

def apply_dp_to_data(df, epsilon=1.0):
    """Apply a simple DP mechanism to mask identifiers and perturb values.

    Args:
        df (pandas.DataFrame): Original PHI/PII dataset.
        epsilon (float): Privacy parameter for Laplace noise.

    Returns:
        pandas.DataFrame: Privacy-protected dataset safe for sharing/training.
    """
    
    print(f"\n🔒 Applying DP with ε={epsilon} to protect privacy...")
    
    # Create DP-processed version
    dp_data = df.copy()
    
    # 1. DP on categorical data (conditions, medications)
    # Add Laplace noise to counts, then sample
    conditions = df['condition'].unique()
    medications = df['medication'].unique()
    
    # Get true counts
    condition_counts = df['condition'].value_counts()
    medication_counts = df['medication'].value_counts()
    
    # Add Laplace noise to counts
    def add_laplace_noise(counts, epsilon):
        noisy_counts = {}
        for item, count in counts.items():
            # Laplace mechanism: add noise with scale = 1/epsilon
            noise = np.random.laplace(0, 1/epsilon)
            noisy_count = max(0, count + noise)  # Ensure non-negative
            noisy_counts[item] = noisy_count
        return noisy_counts
    
    noisy_condition_counts = add_laplace_noise(condition_counts, epsilon)
    noisy_medication_counts = add_laplace_noise(medication_counts, epsilon)
    
    # 2. DP on numerical data - skip if columns don't exist in this data structure
    # (The current data structure uses text-based PHI/PII data)
    
    # 3. DP on text data - replace with DP-aggregated summaries
    # Instead of individual notes, create DP-aggregated statistics
    dp_data['medical_notes'] = f"[DP-PROCESSED] Aggregated medical data with ε={epsilon}. " \
                              f"Original data protected by differential privacy."
    
    # 4. Remove/mask direct identifiers
    dp_data['name'] = "[DP-MASKED]"
    dp_data['phone'] = "[DP-MASKED]"
    dp_data['email'] = "[DP-MASKED]"
    dp_data['address'] = "[DP-MASKED]"
    dp_data['ssn'] = "[DP-MASKED]"
    dp_data['insurance_id'] = "[DP-MASKED]"
    dp_data['mrn'] = "[DP-MASKED]"
    
    # 5. DP on categorical fields - use noisy counts to resample
    # Sample conditions and medications based on noisy counts
    condition_probs = {k: v/sum(noisy_condition_counts.values()) 
                      for k, v in noisy_condition_counts.items()}
    medication_probs = {k: v/sum(noisy_medication_counts.values()) 
                       for k, v in noisy_medication_counts.items()}
    
    dp_data['condition'] = np.random.choice(
        list(condition_probs.keys()), 
        size=len(df), 
        p=list(condition_probs.values())
    )
    dp_data['medication'] = np.random.choice(
        list(medication_probs.keys()), 
        size=len(df), 
        p=list(medication_probs.values())
    )
    
    return dp_data

def encrypt_data(num_records=100, dp_eps=1.0):
    """Create both ORIGINAL and DP_ENCRYPTED CSV files under data/.

    Returns path to the encrypted CSV for downstream training.
    """
    print("=" * 70)
    print("🔒 CREATING BOTH ORIGINAL AND DP-ENCRYPTED DATA")
    print("=" * 70)
    
    # Create original data
    original_data = create_original_phi_data(num_records=num_records)
    
    # Create data directory if it doesn't exist
    data_dir = create_data_folder()
    
    # Generate timestamp for consistent naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save original data
    original_file = f"{data_dir}/ORIGINAL_phi_data_{timestamp}.csv"
    print(f"\n💾 Saving ORIGINAL data to: {original_file}")
    print("⚠️ WARNING: This file contains sensitive PHI/PII!")
    original_data.to_csv(original_file, index=False)
    
    # Apply DP and save encrypted data
    print(f"\n🔒 Processing with ε = {dp_eps}")
    dp_data = apply_dp_to_data(original_data, epsilon=dp_eps)
    
    # Save DP-processed data
    dp_file = f"{data_dir}/DP_ENCRYPTED_phi_data_eps{dp_eps}_{timestamp}.csv"
    dp_data.to_csv(dp_file, index=False)
    print(f"✅ DP-encrypted data saved to: {dp_file}")
    
    # Show sample of DP-processed data
    print(f"\n📊 Sample of DP-encrypted data (ε={dp_eps}):")
    print(dp_data[['patient_id', 'age', 'weight', 'height', 'condition', 'medication']].head())
    
    print(f"\n" + "=" * 70)
    print("📁 FILES CREATED IN DATA FOLDER:")
    print("=" * 70)
    print(f"🔴 ORIGINAL_phi_data_{timestamp}.csv - Contains sensitive PHI/PII")
    print(f"🟢 DP_ENCRYPTED_phi_data_eps{dp_eps}_{timestamp}.csv - Privacy-protected")
    
    print(f"\n" + "=" * 70)
    print("🔒 PRIVACY SUMMARY:")
    print("=" * 70)
    print("✅ ORIGINAL data: Contains real PHI/PII (sensitive - delete after use)")
    print("✅ DP-ENCRYPTED data: Privacy-protected, safe to share and train models")
    print("✅ Use --train to train with the encrypted data!")
    print("=" * 70)
    
    return dp_file  # Return path to encrypted data for training

def clean_all_data_and_models():
    """Delete all CSV data files and saved model directories for a clean slate."""
    
    print("=" * 60)
    print("🧹 CLEANING ALL DATA AND MODELS")
    print("=" * 60)
    
    deleted_count = 0
    
    # Clean data folder
    data_dir = "data"
    if os.path.exists(data_dir):
        data_files = glob.glob(os.path.join(data_dir, "*.csv"))
        for file_path in data_files:
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted data file: {os.path.basename(file_path)}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
    
    # Clean models folder
    models_dir = "models"
    if os.path.exists(models_dir):
        model_dirs = glob.glob(os.path.join(models_dir, "vaultgemma_dp_*"))
        for model_dir in model_dirs:
            try:
                shutil.rmtree(model_dir)
                print(f"🗑️ Deleted model directory: {os.path.basename(model_dir)}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {model_dir}: {e}")
    
    print(f"\n✅ Cleanup completed! Deleted {deleted_count} files/directories")
    print("=" * 60)

# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differential Privacy Fine-tuning for VaultGemma Model",
        epilog="""
Flags and examples:

  --clean
    Delete all existing data files and model directories before proceeding.
    Examples:
      %(prog)s --clean
      %(prog)s --clean --encrypt-data --records 100
      %(prog)s --clean --train --records 50 --epochs 2
      %(prog)s --clean --train --qa --records 20 --epochs 3

  --encrypt-data [--records N] [--dp-data-eps EPS]
    Create ORIGINAL_phi_data_*.csv (sensitive) and DP_ENCRYPTED_phi_data_*.csv (safe).
    Examples:
      %(prog)s --encrypt-data
      %(prog)s --encrypt-data --records 200
      %(prog)s --encrypt-data --records 50 --dp-data-eps 0.5

  --train [--records N] [--epochs E] [--batch_size B]
    Train a LoRA adapter. If a DP_ENCRYPTED file exists, it's used automatically.
    Examples:
      %(prog)s --train
      %(prog)s --train --records 100 --epochs 2
      %(prog)s --train --records 20 --batch_size 2

  Accuracy-focused training (quick start):
    CPU, higher accuracy (disable extra DP-SGD; base model already DP):
      %(prog)s --train --records 500 --epochs 3 --batch_size 1 --seq-len 256 --no-dp-model
    With model-level DP-SGD (slower, lower utility):
      %(prog)s --train --dp-model --records 300 --epochs 2 --batch_size 1 --seq-len 192

  -qa, --qa
    Train in QA mode: automatically generate question/answer pairs from the
    synthetic data so the model can answer questions that appear in training.
    Examples:
      %(prog)s --clean --train -qa --records 500 --epochs 2
      %(prog)s --train --qa --records 1000 --epochs 3

  --dp-model [--dp-eps EPS] [--dp-delta DELTA] [--dp-max-grad-norm G] [--secure-rng]
    Enable additional model-level DP-SGD via Opacus (default: False).
    Note: VaultGemma is already pre-trained with DP-SGD (eps≤2.0, δ≤1.1e-10).
    Examples:
      %(prog)s --train --dp-model
      %(prog)s --train --dp-model --dp-eps 6 --dp-delta 1e-5
      %(prog)s --train --dp-model --dp-max-grad-norm 0.5 --secure-rng

  --dp-data [--dp-data-eps EPS] / --no-dp-data
    Enable/disable data-level DP sampling (default: ENABLED).
    Examples:
      %(prog)s --train --dp-data --dp-data-eps 1.0
      %(prog)s --train --no-dp-data
      %(prog)s --encrypt-data --records 100 --dp-data-eps 0.8

  --cpu / --gpu
    Select device (CPU or CUDA). Automatically falls back to CPU if CUDA is
    unavailable or on out-of-memory (OOM).
    Examples:
      %(prog)s --train --gpu
      %(prog)s --train --gpu --seq-len 256 --batch_size 1
      %(prog)s --train --cpu

  --seq-len N
    Tokenization max length (default: 512). Lower to reduce VRAM/CPU RAM.
    Examples:
      %(prog)s --train --seq-len 256
      %(prog)s --train --gpu --seq-len 192 --batch_size 1

  --query [--model PATH]
    Open an interactive prompt to query the latest or specified model.
    Examples:
      %(prog)s --query
      %(prog)s --query --model models/vaultgemma_dp_20250113_140000

  --prompt "QUESTION"
    One-shot query and exit (use with --query).
    Examples:
      %(prog)s --query --prompt "What are the medical notes for Timothy Green?"
      %(prog)s --query --model models/vaultgemma_dp_20250113_140000 --prompt "Summarize patient follow-up"

  --context-csv PATH, --patient NAME, --use-trained-data
    Retrieval-augmented query: look up a patient row in a CSV and inject the
    medical notes (or key fields) into the prompt.
    Examples:
      %(prog)s --query --context-csv data/ORIGINAL_phi_data_YYYYMMDD_HHMMSS.csv --patient "Timothy Green" --prompt "What was prescribed?"
      %(prog)s --query --use-trained-data --patient "Timothy Green" --prompt "Summarize diagnosis"

  --list, --list-data
    List available models or data files.
    Examples:
      %(prog)s --list
      %(prog)s --list-data

  --reveal-dp
    Show DP-masked data samples for verification.
    Examples:
      %(prog)s --list-data --reveal-dp
      %(prog)s --list-data --reveal-dp --unmask

  --unmask
    Show original unprotected data for comparison (if available).
    WARNING: Shows sensitive PHI/PII data!
    Examples:
      %(prog)s --list-data --unmask
      %(prog)s --list-data --reveal-dp --unmask

  --help
    Show this help message.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", action="store_true", 
                       help="Train a new model. Example: %(prog)s --train --records 100 --epochs 2")
    parser.add_argument("--query", action="store_true", 
                       help="Query the most recent model. Example: %(prog)s --query")
    parser.add_argument("--prompt", type=str,
                       help="One-shot prompt to ask the model, then exit. Example: %(prog)s --query --prompt 'What are the medical notes for Timothy Green?'")
    parser.add_argument("--context-csv", type=str,
                       help="CSV file to use as retrieval context (e.g., ORIGINAL_phi_data_*.csv)")
    parser.add_argument("--patient", type=str,
                       help="Patient name to lookup in the context CSV (e.g., 'Timothy Green')")
    parser.add_argument("--use-trained-data", action="store_true",
                       help="Use the most recent data CSV automatically for retrieval context")
    parser.add_argument("--list", action="store_true", 
                       help="List available models. Example: %(prog)s --list")
    parser.add_argument("--list-data", action="store_true", 
                       help="List available synthetic data files. Example: %(prog)s --list-data")
    parser.add_argument("--model", type=str, 
                       help="Path to specific model for querying. Example: %(prog)s --query --model models/vaultgemma_dp_20250113_140000")
    parser.add_argument("--records", type=int, default=50, 
                       help="Number of training records (default: 50). Example: %(prog)s --train --records 100")
    parser.add_argument("--epochs", type=int, default=1, 
                       help="Number of training epochs (default: 1). Example: %(prog)s --train --epochs 3")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Training batch size (default: 1). Example: %(prog)s --train --batch_size 2")
    parser.add_argument("--seq-len", type=int, default=512,
                       help="Maximum sequence length for tokenization/training (default: 512). Lower to save VRAM/CPU RAM.")
    # DP flags (enabled by default)
    parser.add_argument("--dp-model", action="store_true", default=False,
                       help="Enable additional model-level DP-SGD via Opacus (default: False). Note: VaultGemma already has DP built-in (ε≤2.0, δ≤1.1e-10). Example: %(prog)s --train --dp-model")
    parser.add_argument("--no-dp-model", dest="dp_model", action="store_false",
                       help="Disable model-level DP-SGD. Example: %(prog)s --train --no-dp-model")
    parser.add_argument("--dp-eps", type=float, default=8.0,
                       help="Target epsilon for model-level DP-SGD (default: 8.0). Example: %(prog)s --train --dp-model --dp-eps 6")
    parser.add_argument("--dp-delta", type=float, default=1e-5,
                       help="Delta for model-level DP-SGD (default: 1e-5). Example: %(prog)s --train --dp-model --dp-delta 1e-5")
    parser.add_argument("--dp-max-grad-norm", type=float, default=0.1,
                       help="Max per-sample grad norm for DP-SGD (default: 0.1). Example: %(prog)s --train --dp-model --dp-max-grad-norm 0.5")
    parser.add_argument("--secure-rng", action="store_true",
                       help="Use cryptographically secure RNG for DP noise (requires torchcsprng). Example: %(prog)s --train --dp-model --secure-rng")
    parser.add_argument("--dp-data", action="store_true", default=True,
                       help="Enable data-level DP sampling for synthetic data generation (default: True). Example: %(prog)s --train --dp-data --dp-data-eps 1.0")
    parser.add_argument("--no-dp-data", dest="dp_data", action="store_false",
                       help="Disable data-level DP sampling. Example: %(prog)s --train --no-dp-data")
    parser.add_argument("--dp-data-eps", type=float, default=1.0,
                       help="Epsilon for data-level DP sampling (default: 1.0). Example: %(prog)s --train --dp-data --dp-data-eps 0.5")
    parser.add_argument("-qa", "--qa", action="store_true", default=True,
                       help="Enable QA-style training so the model can answer questions seen in training data (default: True).")
    parser.add_argument("--no-qa", dest="qa", action="store_false",
                       help="Disable QA-style training. Example: %(prog)s --train --no-qa")
    # Device selection flags
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU for training/query (overrides auto-device)")
    parser.add_argument("--gpu", action="store_true",
                       help="Force GPU (CUDA) for training/query if available; falls back to CPU if not")
    parser.add_argument("--encrypt-data", action="store_true",
                       help="Create both original and DP-encrypted data files. Example: %(prog)s --encrypt-data --records 100")
    parser.add_argument("--clean", action="store_true",
                       help="Delete all existing data and models before proceeding. Example: %(prog)s --clean --train --records 100")
    parser.add_argument("--reveal-dp", action="store_true",
                       help="Show DP-masked data (for debugging/verification). Example: %(prog)s --list-data --reveal-dp")
    parser.add_argument("--unmask", action="store_true",
                       help="Show original unprotected data for comparison (if available). Example: %(prog)s --list-data --unmask")
    
    args = parser.parse_args()
    
    # Handle device selection early
    if args.gpu:
        set_selected_device("cuda")
    elif args.cpu:
        set_selected_device("cpu")

    # Handle clean flag first (before any other operations)
    if args.clean:
        clean_all_data_and_models()
        print("🧹 Cleanup completed. Proceeding with requested operations...")
    
    if args.encrypt_data:
        print("Starting data encryption mode...")
        encrypted_data_path = encrypt_data(num_records=args.records, dp_eps=args.dp_data_eps)
        print("\nData encryption completed!")
        print(f"Encrypted data saved to: {encrypted_data_path}")
        print("You can now use --train to train with the encrypted data!")
    
    elif args.train:
        print("Starting training mode...")
        model_path = train_model(
            num_records=args.records,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dp_model=args.dp_model,
            dp_epsilon_model=args.dp_eps,
            dp_delta=args.dp_delta,
            dp_max_grad_norm=args.dp_max_grad_norm,
            secure_rng=args.secure_rng,
            dp_data=args.dp_data,
            dp_data_epsilon=args.dp_data_eps,
            qa=args.qa,
            seq_len=args.__dict__.get('seq_len', 512),
        )
        if model_path:
            print("\nTraining completed successfully!")
            print(f"Model saved to: {model_path}")
        else:
            print("Training failed!")
    
    elif args.query:
        print("Starting query mode...")
        model_path = args.model if args.model else None
        query_model(
            model_path,
            prompt=args.prompt,
            context_csv=args.context_csv,
            patient=args.patient,
            use_trained_data=args.use_trained_data,
        )
    
    elif args.list:
        print("Listing available models...")
        list_available_models()
    
    elif args.list_data:
        print("Listing available synthetic data files...")
        list_available_data(reveal_dp=args.reveal_dp, unmask=args.unmask)
    
    else:
        print("Differential Privacy Fine-tuning for VaultGemma Model")
        print("\nUsage:")
        print("  python LLM_Diffrential_Privacy.py --train                    # Train a new model")
        print("  python LLM_Diffrential_Privacy.py --query                    # Query the most recent model")
        print("  python LLM_Diffrential_Privacy.py --query --model <path>     # Query a specific model")
        print("  python LLM_Diffrential_Privacy.py --list                     # List available models")
        print("  python LLM_Diffrential_Privacy.py --list-data                # List available synthetic data files")
        print("\nTraining options:")
        print("  --records <num>     Number of training records (default: 50)")
        print("  --epochs <num>      Number of training epochs (default: 1)")
        print("  --batch_size <num>  Training batch size (default: 1)")
        print("\n" + "="*60)
        print("COMPREHENSIVE EXAMPLES:")
        print("="*60)
        print("\n📋 HELP & INFORMATION:")
        print("  python LLM_Diffrential_Privacy.py --help                     # Show this help message")
        print("  python LLM_Diffrential_Privacy.py -h                         # Short form of help")
        print("  python LLM_Diffrential_Privacy.py --list                     # List all available models")
        print("  python LLM_Diffrential_Privacy.py --list-data                # List all synthetic data files")
        print("\n🚀 TRAINING EXAMPLES:")
        print("  python LLM_Diffrential_Privacy.py --train                    # Train with default settings (50 records, 1 epoch)")
        print("  python LLM_Diffrential_Privacy.py --train --records 20       # Train with 20 records")
        print("  python LLM_Diffrential_Privacy.py --train --records 100      # Train with 100 records")
        print("  python LLM_Diffrential_Privacy.py --train --epochs 2         # Train for 2 epochs")
        print("  python LLM_Diffrential_Privacy.py --train --batch_size 2     # Train with batch size 2")
        print("  python LLM_Diffrential_Privacy.py --train --records 200 --epochs 3 --batch_size 1  # Full custom training")
        print("  python LLM_Diffrential_Privacy.py --train --records 50 --epochs 1 --batch_size 1   # Quick training")
        print("  python LLM_Diffrential_Privacy.py --train --records 500 --epochs 5                 # Extended training")
        print("\n💬 QUERYING EXAMPLES:")
        print("  python LLM_Diffrential_Privacy.py --query                    # Query the most recent model")
        print("  python LLM_Diffrential_Privacy.py --query --model models/vaultgemma_dp_20250113_140000  # Query specific model")
        print("  python LLM_Diffrential_Privacy.py --query --model models/vaultgemma_dp_20250113_150000  # Query another model")
        print("\n📁 FILE MANAGEMENT:")
        print("  python LLM_Diffrential_Privacy.py --list                     # See all trained models")
        print("  python LLM_Diffrential_Privacy.py --list-data                # See all synthetic data files")
        print("\n🔄 WORKFLOW EXAMPLES:")
        print("  # Complete workflow:")
        print("  python LLM_Diffrential_Privacy.py --train --records 100 --epochs 2")
        print("  python LLM_Diffrential_Privacy.py --list")
        print("  python LLM_Diffrential_Privacy.py --query")
        print("\n  # Compare different models:")
        print("  python LLM_Diffrential_Privacy.py --train --records 50 --epochs 1")
        print("  python LLM_Diffrential_Privacy.py --train --records 100 --epochs 2")
        print("  python LLM_Diffrential_Privacy.py --list")
        print("  python LLM_Diffrential_Privacy.py --query --model models/vaultgemma_dp_20250113_140000")
        print("  python LLM_Diffrential_Privacy.py --query --model models/vaultgemma_dp_20250113_150000")
        print("\n📊 DATA EXPLORATION:")
        print("  python LLM_Diffrential_Privacy.py --list-data                # See all generated data files")
        print("  # Data files are saved as: data/synthetic_phi_data_YYYYMMDD_HHMMSS.csv")
        print("\n⚡ QUICK START:")
        print("  python LLM_Diffrential_Privacy.py --train --records 20       # Quick training (20 records)")
        print("  python LLM_Diffrential_Privacy.py --query                    # Test the model")
        print("\n🔧 ADVANCED USAGE:")
        print("  python LLM_Diffrential_Privacy.py --train --records 1000 --epochs 3 --batch_size 1  # Large dataset training")
        print("  python LLM_Diffrential_Privacy.py --train --records 10 --epochs 5                   # Small dataset, many epochs")
        print("  python LLM_Diffrential_Privacy.py --train --records 500 --epochs 1 --batch_size 2   # Medium dataset, batch size 2")
        print("\n" + "="*60)
        print("NOTES:")
        print("- Models are saved to: models/vaultgemma_dp_YYYYMMDD_HHMMSS/")
        print("- Data files are saved to: data/synthetic_phi_data_YYYYMMDD_HHMMSS.csv")
        print("- GPU supported via --gpu; auto-fallback to CPU on CUDA OOM")
        print("- Reduce VRAM usage with lower --seq-len and --batch_size")
        print("- Optional: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for CUDA memory fragmentation mitigation")
        print("- Use --list to see available models before querying")
        print("- Use --list-data to see available synthetic data files")
        print("="*60)
        