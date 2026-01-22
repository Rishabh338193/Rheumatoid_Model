"""
Dataset Download Script for Rheumatoid Arthritis Prediction
Downloads knee osteoarthritis dataset from Kaggle and prepares it for RA prediction
"""

import kagglehub
import os
import shutil

def download_dataset():
    """Download dataset from Kaggle using kagglehub"""
    print("=" * 60)
    print("Downloading Knee Osteoarthritis Dataset from Kaggle...")
    print("=" * 60)
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("shashwatwork/knee-osteoarthritis-dataset-with-severity")
        
        print(f"\nDataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        
        # List files in downloaded directory
        print("\nFiles in dataset:")
        for file in os.listdir(path):
            print(f"  - {file}")
            
        # Copy dataset to local data folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        for file in os.listdir(path):
            if file.endswith('.csv'):
                src = os.path.join(path, file)
                dst = os.path.join(current_dir, file)
                shutil.copy2(src, dst)
                print(f"\nCopied {file} to data folder")
                
        print("\n" + "=" * 60)
        print("Dataset ready for use!")
        print("=" * 60)
        
        return path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nNote: Make sure you have:")
        print("1. Installed kagglehub: pip install kagglehub")
        print("2. Authenticated with Kaggle (if required)")
        return None

if __name__ == "__main__":
    download_dataset()
