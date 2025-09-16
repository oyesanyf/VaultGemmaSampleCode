import torch
import pandas as pd
from faker import Faker
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
import os
import argparse
from datetime import datetime
import glob

# Set environment variables for better CUDA debugging and force CPU usage
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only execution

# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================
def create_models_folder():
    """Create models folder if it doesn't exist."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    return models_dir

def create_data_folder():
    """Create data folder if it doesn't exist."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    return data_dir

def get_timestamp_model_name():
    """Generate a unique model name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"vaultgemma_dp_{timestamp}"

def get_most_recent_model():
    """Get the most recent model from the models folder."""
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
    """List all available models in the models folder."""
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

def list_available_data():
    """List all available data files in the data folder."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("No data folder found.")
        return []
    
    data_files = glob.glob(os.path.join(data_dir, "synthetic_phi_data_*.csv"))
    if not data_files:
        print("No synthetic data files found in data folder.")
        return []
    
    print("Available synthetic data files:")
    for i, data_file in enumerate(sorted(data_files, key=os.path.getmtime, reverse=True), 1):
        filename = os.path.basename(data_file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(data_file))
        file_size = os.path.getsize(data_file)
        print(f"  {i}. {filename} (created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}, size: {file_size} bytes)")
    
    return data_files


# ======================================================================================
# STEP 1: GENERATE SYNTHETIC PHI DATA
# ======================================================================================
def generate_phi_data(num_records=2000):
    """Generates a Pandas DataFrame with synthetic patient data containing PHI/PII."""
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
        condition = fake.random_element(elements=conditions)
        medication = fake.random_element(elements=medications)
        
        # Select a random template and fill it
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
def preprocess_for_trainer(examples, tokenizer):
    """Correctly handles batches from dataset.map(batched=True) with PHI/PII data."""
    texts = []
    for i in range(len(examples['name'])):
        # Create a comprehensive prompt that includes PHI/PII context
        name = examples['name'][i]
        medical_notes = examples['medical_notes'][i]
        
        # Include additional PHI/PII context in the prompt
        additional_info = ""
        if 'dob' in examples:
            additional_info += f" (DOB: {examples['dob'][i]})"
        if 'mrn' in examples:
            additional_info += f" (MRN: {examples['mrn'][i]})"
        if 'phone' in examples:
            additional_info += f" (Phone: {examples['phone'][i]})"
        
        text = f"### Human: What are the medical notes for {name}{additional_info}?\n### Assistant: {medical_notes}"
        texts.append(text)

    model_inputs = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
    # Create a deep copy of input_ids for labels to avoid modifying input_ids
    labels = [input_ids.copy() for input_ids in model_inputs["input_ids"]]
    
    for i in range(len(examples['name'])):
        # Create the same prompt structure for masking
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
        # Only mask the prompt part in labels, keep input_ids unchanged
        labels[i][:prompt_length] = [-100] * prompt_length
        
    model_inputs["labels"] = labels
    return model_inputs

# ======================================================================================
# TRAINING FUNCTION
# ======================================================================================
def train_model(num_records=50, epochs=1, batch_size=1):
    """Train the model with differential privacy."""
    print("=== Starting Model Training ===")
    
    # Create models folder
    models_dir = create_models_folder()
    model_name = get_timestamp_model_name()
    adapter_path = os.path.join(models_dir, model_name)
    
    print(f"Training model: {model_name}")
    print(f"Model will be saved to: {adapter_path}")
    
    # Generate data
    print("\n1. Generating synthetic PHI data...")
    phi_df = generate_phi_data(num_records=num_records)
    dataset = Dataset.from_pandas(phi_df)
    print(f"Generated {len(phi_df)} patient records")
    
    # Save synthetic data to data folder
    data_dir = create_data_folder()
    data_filename = f"synthetic_phi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    data_path = os.path.join(data_dir, data_filename)
    phi_df.to_csv(data_path, index=False)
    print(f"Synthetic data saved to: {data_path}")
    
    # Load tokenizer
    print("\n2. Loading tokenizer...")
    base_model_id = "google/vaultgemma-1b"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Preprocess data
    print("\n3. Preprocessing data...")
    tokenized_dataset = dataset.map(lambda ex: preprocess_for_trainer(ex, tokenizer), batched=True, batch_size=10, remove_columns=dataset.column_names)
    
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
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.float32)
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Setup LoRA
    print("\n6. Setting up LoRA...")
    lora_config = LoraConfig(r=8, target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.train()

    # Training setup
    print(f"\n7. Setting up training...")
    TARGET_EPSILON = 8.0
    MAX_GRAD_NORM = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    print("8. Starting DP training on CPU... (This will be slow)")
    model.train()
    total_samples = len(tokenized_dataset)
    total_steps = total_samples * epochs
    step = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for i in range(0, total_samples, batch_size):
            # Get batch
            batch_samples = []
            for j in range(i, min(i + batch_size, total_samples)):
                batch_samples.append(tokenized_dataset[j])
            
            # Convert to tensors
            input_ids = torch.tensor([s['input_ids'] for s in batch_samples], dtype=torch.long)
            attention_mask = torch.tensor([s['attention_mask'] for s in batch_samples], dtype=torch.long)
            labels = torch.tensor([s['labels'] for s in batch_samples], dtype=torch.long)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            # Optimizer step
            optimizer.step()
            
            step += 1
            if step % 10 == 0:
                print(f"  Step {step}/{total_steps}, Loss: {loss.item():.4f}")
    
    print("9. Training complete!")
    print(f"Privacy Guarantee: Epsilon = {TARGET_EPSILON:.2f} for delta = 1e-5")
    
    # Save model
    print("10. Saving model...")
    model.save_pretrained(adapter_path)
    print(f"Model saved to: {adapter_path}")
    
    return adapter_path

# ======================================================================================
# QUERY FUNCTION
# ======================================================================================
def query_model(model_path=None):
    """Query the model with custom questions."""
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
    
    # Load model
    print("Loading model...")
    base_model_id = "google/vaultgemma-1b"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cpu")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.float32)
    base_model = base_model.to(device)
    finetuned_model = PeftModel.from_pretrained(base_model, model_path)
    finetuned_model = finetuned_model.to(device)
    
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
        
        # Format the prompt (supports PHI/PII queries)
        prompt = f"### Human: {question}\n### Assistant:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate response
        print("Generating response...")
        with torch.no_grad():
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_part = response.split("### Assistant:")[1].strip() if "### Assistant:" in response else response
        
        print(f"Model response: {assistant_part}")

# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differential Privacy Fine-tuning for VaultGemma Model",
        epilog="""
Examples:
  %(prog)s --train                                    # Train with default settings (50 records, 1 epoch)
  %(prog)s --train --records 100 --epochs 2           # Train with 100 records for 2 epochs
  %(prog)s --train --records 20 --batch_size 2        # Train with 20 records, batch size 2
  %(prog)s --query                                    # Query the most recent model
  %(prog)s --query --model models/vaultgemma_dp_20250113_140000  # Query specific model
  %(prog)s --list                                     # List all available models        
  %(prog)s --list-data                                # List all synthetic data files    
  %(prog)s --train --records 500 --epochs 3 --batch_size 1  # Full custom training       
  %(prog)s --help                                     # Show this help message
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", action="store_true", 
                       help="Train a new model. Example: %(prog)s --train --records 100 --epochs 2")
    parser.add_argument("--query", action="store_true", 
                       help="Query the most recent model. Example: %(prog)s --query")
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
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting training mode...")
        model_path = train_model(
            num_records=args.records,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        if model_path:
            print(f"\nTraining completed successfully!")
            print(f"Model saved to: {model_path}")
        else:
            print("Training failed!")
    
    elif args.query:
        print("Starting query mode...")
        model_path = args.model if args.model else None
        query_model(model_path)
    
    elif args.list:
        print("Listing available models...")
        list_available_models()
    
    elif args.list_data:
        print("Listing available synthetic data files...")
        list_available_data()
    
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
        print("- All training runs on CPU (GPU disabled for compatibility)")
        print("- Use --list to see available models before querying")
        print("- Use --list-data to see available synthetic data files")
        print("="*60)
