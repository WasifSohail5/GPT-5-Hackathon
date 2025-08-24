import pandas as pd

def load_dataset(file_path: str):
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully!")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        return df
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load dataset: {e}")
