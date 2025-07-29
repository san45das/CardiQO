import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            df = pd.read_csv(self.filepath)
            print(f"✅ Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return None

