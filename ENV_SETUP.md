# Environment Setup

Supports Python 3.10–3.11. Recommended: Python 3.10 for best compatibility.

## 1) Prerequisites
- Python 3.10 (preferred) or 3.11
- Git
- Windows PowerShell or bash/zsh
- Optional GPU: NVIDIA GPU with matching CUDA/driver

Verify Python (Windows):
```powershell
py -3.10 --version
```

## 2) Clone the repository
```powershell
git clone https://github.com/yourusername/LLMEncrption2.git
cd LLMEncrption2
```

## 3) Create a clean virtual environment
If a previous env is broken (e.g., “No pyvenv.cfg file”), remove it first:
```powershell
Remove-Item -Recurse -Force .\dp_env -ErrorAction SilentlyContinue
```
Create and activate (Windows PowerShell):
```powershell
py -3.10 -m venv dp_env
.\dp_env\Scripts\Activate.ps1
```
Linux/macOS:
```bash
python3.10 -m venv dp_env
source dp_env/bin/activate
```

## 4) Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Notes:
- Configure proxy if needed for pip.
- If an install fails, re-run or pin a compatible version.

## 5) Optional GPU (PyTorch)
Find the exact command at `https://pytorch.org/` (depends on your CUDA/driver). Example (verify yours):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Keep other packages from `requirements.txt`.

## 6) Quick verification
```powershell
python LLM_Diffrential_Privacy.py --help | cat
python LLM_Diffrential_Privacy.py --encrypt-data --records 10
python LLM_Diffrential_Privacy.py --list-data
python LLM_Diffrential_Privacy.py --train --records 10 --epochs 1 --batch_size 1 --seq-len 128
python LLM_Diffrential_Privacy.py --query --prompt "What was prescribed to Anne Thompson?"
```
Load a specific adapter directory if desired:
```powershell
python LLM_Diffrential_Privacy.py --query --model "models\vaultgemma_dp_YYYYMMDD_HHMMSS" --prompt "..."
```

## 7) Common issues & fixes
- “No pyvenv.cfg file”: delete `dp_env/` and recreate (steps 3–4).
- Adapter not found / treated as Hub repo id:
  - Pass a local adapter directory (must contain `adapter_config.json` and `adapter_model.safetensors`).
  - The CLI validates local files and falls back to `phi-vaultgemma-finetuned-adapter-dp` if present.
- Slow CPU training: reduce `--seq-len`, keep `--batch_size 1`.
- OOM/memory: lower `--seq-len`.

## 8) Hygiene & security
- Do not commit `dp_env/` (gitignored).
- Keep `requirements.txt` versioned.
- Don’t store tokens/secrets in the repo or shell history.

## 9) One-shot repro (Windows)
```powershell
cd D:\LLMEncrption2
Remove-Item -Recurse -Force .\dp_env -ErrorAction SilentlyContinue
py -3.10 -m venv dp_env
.\dp_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python LLM_Diffrential_Privacy.py --encrypt-data --records 10
python LLM_Diffrential_Privacy.py --train --records 10 --epochs 1 --batch_size 1 --seq-len 128
python LLM_Diffrential_Privacy.py --query --prompt "What was prescribed to Anne Thompson?"
```
