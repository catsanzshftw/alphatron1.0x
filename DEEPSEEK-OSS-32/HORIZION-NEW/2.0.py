#!/usr/bin/env python3
"""
Setup script for CATseek-r3.2
Installs all required dependencies
"""

import subprocess
import sys
import os

def main():
    print("🐱 CATseek-r3.2 Setup")
    print("=" * 50)
    
    # List of required packages
    packages = [
        "torch",  # PyTorch for ML
        "transformers>=4.40.0",  # Hugging Face transformers
        "accelerate>=0.27.0",  # For model loading optimization
        "bitsandbytes",  # For 4-bit/8-bit quantization
        "pillow",  # For image handling
        "huggingface-hub",  # For model downloading
        "safetensors",  # For safe model loading
        "sentencepiece",  # Often required for tokenizers
        "protobuf",  # Protocol buffers
    ]
    
    print("\n📦 Installing required packages...")
    
    for package in packages:
        print(f"\n🔧 Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", package
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            print("   Please install manually using: pip install " + package)
    
    print("\n🔍 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available! Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available. Model will run on CPU (slower)")
            print("   For GPU acceleration, install CUDA-enabled PyTorch:")
            print("   Visit: https://pytorch.org/get-started/locally/")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
    
    print("\n💾 Model Information:")
    print("   DeepSeek-R1-Zero will be downloaded on first run (~16GB)")
    print("   Ensure you have sufficient disk space and RAM/VRAM")
    print("   Recommended: 24GB+ RAM for CPU or 16GB+ VRAM for GPU")
    
    print("\n✅ Setup complete!")
    print("\n🚀 To run CATseek:")
    print("   python catseekr3_fixed.py")
    
    # Optional: Create a batch/shell script for easy launching
    if os.name == 'nt':  # Windows
        with open("run_catseek.bat", "w") as f:
            f.write(f'@echo off\n"{sys.executable}" catseekr3_fixed.py\npause')
        print("\n📝 Created run_catseek.bat for easy launching")
    else:  # Unix/Linux/Mac
        with open("run_catseek.sh", "w") as f:
            f.write(f'#!/bin/bash\n{sys.executable} catseekr3_fixed.py')
        os.chmod("run_catseek.sh", 0o755)
        print("\n📝 Created run_catseek.sh for easy launching")

if __name__ == "__main__":
    main()
