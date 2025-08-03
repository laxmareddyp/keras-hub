#!/usr/bin/env python3
"""
Example usage of the HuggingFace to KerasHub converter.

This script demonstrates how to convert various HuggingFace models
to KerasHub-compatible implementations.
"""

import subprocess
import sys
from pathlib import Path

def run_conversion(model_name: str, output_dir: str = "./generated_models"):
    """Run the conversion for a specific model."""
    print(f"\n{'='*60}")
    print(f"Converting: {model_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "hf_to_kerashub_converter.py",
        "--model_name", model_name,
        "--output_dir", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Conversion successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Conversion failed!")
        print(f"Error: {e.stderr}")
        return False
    
    return True

def main():
    """Run conversions for various model types."""
    
    # List of example models to convert
    example_models = [
        # GPT-style models
        "microsoft/DialoGPT-medium",
        "gpt2",
        
        # BERT-style models  
        "bert-base-uncased",
        "distilbert-base-uncased",
        
        # Modern LLMs (smaller variants for testing)
        "facebook/opt-125m",
        "EleutherAI/gpt-neo-125M",
        
        # Specialized models
        "microsoft/codebert-base",
    ]
    
    print("🚀 HuggingFace to KerasHub Converter - Example Usage")
    print("\nThis script will demonstrate converting various HuggingFace models to KerasHub format.")
    
    # Create output directory
    output_dir = "./example_generated_models"
    Path(output_dir).mkdir(exist_ok=True)
    
    successful_conversions = []
    failed_conversions = []
    
    for model_name in example_models:
        success = run_conversion(model_name, output_dir)
        if success:
            successful_conversions.append(model_name)
        else:
            failed_conversions.append(model_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n✅ Successful conversions ({len(successful_conversions)}):")
    for model in successful_conversions:
        print(f"   - {model}")
    
    if failed_conversions:
        print(f"\n❌ Failed conversions ({len(failed_conversions)}):")
        for model in failed_conversions:
            print(f"   - {model}")
    
    print(f"\n📁 Generated files are in: {output_dir}")
    print("\n📝 Next steps for each converted model:")
    print("   1. Review generated files and customize as needed")
    print("   2. Implement tokenizer vocabulary extraction")
    print("   3. Fine-tune weight conversion mapping")
    print("   4. Test with actual model weights")
    print("   5. Add to KerasHub model registry")

if __name__ == "__main__":
    main()