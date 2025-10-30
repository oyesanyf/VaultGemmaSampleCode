"""
Test Script for VaultGemma DP Model
=====================================
This script demonstrates 5 example queries to test your trained model.
Run each command to evaluate different aspects of the model's performance.

Model: models\vaultgemma_dp_20251017_062150
Training Results: F1=0.231, EM=0.000

Usage: Run each command individually in your terminal
"""

import subprocess
import sys

# Define test queries
test_queries = [
    {
        "name": "Test 1: Diagnosis Query",
        "description": "Ask about a patient's medical condition/diagnosis",
        "command": [
            sys.executable, "LLM_Diffrential_Privacy.py",
            "--query",
            "--model", "models\\vaultgemma_dp_20251017_062150",
            "--prompt", "What is the diagnosis for the patient?",
            "--gen-beams", "1",
            "--gen-max-new-tokens", "10"
        ]
    },
    {
        "name": "Test 2: Medication Query",
        "description": "Ask what medication was prescribed",
        "command": [
            sys.executable, "LLM_Diffrential_Privacy.py",
            "--query",
            "--model", "models\\vaultgemma_dp_20251017_062150",
            "--prompt", "What medication is prescribed?",
            "--gen-beams", "1",
            "--gen-max-new-tokens", "10"
        ]
    },
    {
        "name": "Test 3: General Medical Condition Query",
        "description": "Ask about common conditions treated",
        "command": [
            sys.executable, "LLM_Diffrential_Privacy.py",
            "--query",
            "--model", "models\\vaultgemma_dp_20251017_062150",
            "--prompt", "What conditions are commonly treated?",
            "--gen-beams", "1",
            "--gen-max-new-tokens", "15"
        ]
    },
    {
        "name": "Test 4: Treatment Protocol Query",
        "description": "Ask about treatment for a specific condition",
        "command": [
            sys.executable, "LLM_Diffrential_Privacy.py",
            "--query",
            "--model", "models\\vaultgemma_dp_20251017_062150",
            "--prompt", "What is the treatment for hypertension?",
            "--gen-beams", "1",
            "--gen-max-new-tokens", "10"
        ]
    },
    {
        "name": "Test 5: Medication Information Query",
        "description": "Ask about a specific medication",
        "command": [
            sys.executable, "LLM_Diffrential_Privacy.py",
            "--query",
            "--model", "models\\vaultgemma_dp_20251017_062150",
            "--prompt", "What is Lisinopril used for?",
            "--gen-beams", "1",
            "--gen-max-new-tokens", "12"
        ]
    }
]

def print_separator():
    print("\n" + "="*70 + "\n")

def run_tests_interactive():
    """Run tests with manual confirmation"""
    print("="*70)
    print("VaultGemma DP Model - Test Query Suite")
    print("="*70)
    print("\nModel Path: models\\vaultgemma_dp_20251017_062150")
    print("Training Stats: F1=0.231, EM=0.000")
    print("\n5 Test Queries Ready:")
    print_separator()
    
    for i, test in enumerate(test_queries, 1):
        print(f"Test {i}: {test['name']}")
        print(f"Description: {test['description']}")
        print(f"\nCommand:")
        print(" ".join(test['command']))
        print_separator()
    
    print("\nYou can:")
    print("1. Copy and run each command manually")
    print("2. Run this script with --auto to execute all tests automatically")
    print("\nTo run automatically: python test_model_queries.py --auto")

def run_tests_auto():
    """Run all tests automatically"""
    print("="*70)
    print("Running All Tests Automatically")
    print("="*70)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Running Test {i}: {test['name']}")
        print(f"Description: {test['description']}")
        print(f"{'='*70}\n")
        
        try:
            result = subprocess.run(
                test['command'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            print("STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("\nSTDERR:")
                print(result.stderr)
            
            print(f"\nReturn Code: {result.returncode}")
            
        except subprocess.TimeoutExpired:
            print(f"ERROR: Test {i} timed out after 120 seconds")
        except Exception as e:
            print(f"ERROR running test {i}: {str(e)}")
        
        print_separator()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        run_tests_auto()
    else:
        run_tests_interactive()

