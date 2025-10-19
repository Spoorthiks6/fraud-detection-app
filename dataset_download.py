import os
import zipfile
import pandas as pd

def download_and_load_dataset():
    dataset_dir = "datasets"
    dataset_zip = os.path.join(dataset_dir, "creditcardfraud.zip")
    csv_path = os.path.join(dataset_dir, "creditcard.csv")

    # Create directory if not exists
    os.makedirs(dataset_dir, exist_ok=True)

    # If dataset not found, download it
    if not os.path.exists(csv_path):
        print("⬇️ Downloading dataset from Kaggle...")
        os.system('kaggle datasets download -d mlg-ulb/creditcardfraud -p datasets')
        
        # Unzip
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("✅ Dataset extracted successfully!")

    # Load CSV
    print(f"📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print("✅ Dataset loaded successfully!")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nClass Distribution:")
    print(df['Class'].value_counts())
    return df
