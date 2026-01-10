"""
Simple installation script to fix dependency conflicts.
Run this instead of pip install -r requirements.txt
"""
import subprocess
import sys

def run_pip(command):
    """Run pip command and show output."""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print('='*60)
    result = subprocess.run(command, shell=True)
    return result.returncode == 0

print("\n" + "="*60)
print("FIXING DEPENDENCY INSTALLATION")
print("="*60)

# Step 1: Fix setuptools first
print("\n[1/7] Upgrading pip and setuptools...")
if not run_pip("python -m pip install --upgrade pip setuptools wheel"):
    print("Failed to upgrade pip/setuptools")
    sys.exit(1)

# Step 2: Install numpy first (Python 3.12 compatible version)
print("\n[2/7] Installing numpy 1.26.4...")
if not run_pip("pip install numpy==1.26.4"):
    print("Failed to install numpy")
    sys.exit(1)

# Step 3: Install core packages
print("\n[3/7] Installing core packages...")
if not run_pip("pip install tiktoken==0.8.0 python-dotenv==1.0.1 tqdm==4.67.1"):
    print("Warning: Some core packages failed")

# Step 4: Install ML packages
print("\n[4/7] Installing ML packages...")
if not run_pip("pip install scikit-learn==1.3.2 pandas==2.0.3"):
    print("Warning: ML packages failed")

# Step 5: Install torch (needed for sentence-transformers)
print("\n[5/7] Installing PyTorch...")
if not run_pip("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"):
    print("Warning: PyTorch install failed, trying alternate method...")
    run_pip("pip install torch")

# Step 6: Install transformers and sentence-transformers (newer compatible versions)
print("\n[6/7] Installing transformers and sentence-transformers...")
if not run_pip("pip install sentence-transformers"):
    print("Warning: Transformers install failed")

# Step 7: Install faiss-cpu (compatible with newer numpy)
print("\n[7/7] Installing faiss-cpu...")
if not run_pip("pip install faiss-cpu"):
    print("Warning: FAISS install failed")

# Step 8: Install Google Generative AI
print("\n[8/8] Installing Google Generative AI...")
run_pip("pip install google-generativeai==0.3.0")

print("\n" + "="*60)
print("INSTALLATION COMPLETE!")
print("="*60)
print("\nTry running: python main.py")
print("="*60 + "\n")
