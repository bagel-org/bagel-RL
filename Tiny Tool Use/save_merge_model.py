import os
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with a base model and save the result."
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or name of the base model to merge adapters with"
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Directory containing the adapter weights"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where to save the merged model"
    )
    
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Save the merged model in half precision (float16)"
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading remote code for tokenizer and model"
    )
    
    parser.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Don't use safetensors format for saving"
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.half_precision else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    
    print(f"Loading adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    # Merge adapter weights with base model
    print("Merging adapter with base model...")
    model = model.merge_and_unload()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the merged model
    print(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=not args.no_safetensors
    )
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code
    )
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✓ Model successfully merged and saved to {args.output_dir}")

if __name__ == "__main__":
    main()


