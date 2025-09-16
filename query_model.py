"""
Simple script to query your fine-tuned model with custom questions.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def query_model():
    """Query the fine-tuned model with custom questions."""
    
    print("=== Model Query Interface ===")
    
    # Load the fine-tuned model
    base_model_id = "google/vaultgemma-1b"
    adapter_path = "./phi-vaultgemma-finetuned-adapter-dp"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cpu")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.float32)
    base_model = base_model.to(device)
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
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
        
        # Format the prompt
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

if __name__ == "__main__":
    query_model()
