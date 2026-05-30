import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        self.df = pd.read_csv(self.filepath)
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])
        return self.df

    def clean_currency(self, col_name):
        if col_name in self.df.columns:
            self.df[col_name] = (
                self.df[col_name]
                .astype(str)
                .str.replace("Rs.", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            self.df[col_name] = pd.to_numeric(self.df[col_name], errors="coerce")

    def extract_numeric(self, cols):
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.extract(r"(\d+\.?\d*)")[0]
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def preprocess(self):
        # Clean Price
        self.clean_currency("Ex-Showroom_Price")
        self.df = self.df.dropna(subset=["Ex-Showroom_Price"])

        # Dynamically Extract Numeric Columns (Algorithmic Approach)
        numeric_keywords = ["Displacement", "Power", "Torque", "Mileage", "Capacity", "Length", "Width", "Height", "Wheelbase", "Weight", "Volume"]
        cols_to_extract = []
        for col in self.df.columns:
            if any(kw in col for kw in numeric_keywords) or self.df[col].dtype == 'object':
                 # Test if the column starts with a number in most cases
                 if self.df[col].astype(str).str.match(r'^\d+\.?\d*').mean() > 0.1: # If >10% of values start with a number
                     cols_to_extract.append(col)
                     
        self.extract_numeric(cols_to_extract)

        # Mileage Logic Extract
        if "City_Mileage" not in self.df.columns and "ARAI_Certified_Mileage" in self.df.columns:
            # Try to grab another common mileage field
            self.df["Mileage"] = self.df["ARAI_Certified_Mileage"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
            self.df["Mileage"] = pd.to_numeric(self.df["Mileage"], errors="coerce")
        elif "City_Mileage" in self.df.columns:
            self.df["Mileage"] = self.df["City_Mileage"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
            self.df["Mileage"] = pd.to_numeric(self.df["Mileage"], errors="coerce")
            
        if "Mileage" not in self.df.columns:
             if "Displacement" in self.df.columns:
                self.df["Mileage"] = 25 - (self.df["Displacement"] / 200)
                self.df["Mileage"] = self.df["Mileage"].clip(lower=5, upper=25)

        # Drop rows missing absolutely critical ranking anchors
        self.df = self.df.dropna(subset=["Displacement", "Power", "Mileage"])
        
        # Algorithmic Feature Reduction Step 1: Missing Value Threshold
        # Drop columns where more than 50% of the data is missing
        thresh = int(0.5 * len(self.df))
        self.df = self.df.dropna(axis=1, thresh=thresh)

        

        
        # Fill missing values for ranking features with median (safe fallback)
        rank_features = ["Mileage", "Seating_Capacity", "Power", "Displacement"]
        for feat in rank_features:
            if feat in self.df.columns:
                self.df[feat] = self.df[feat].fillna(self.df[feat].median())

        return self.df

    def get_data(self):
        """Public method to get processed data."""
        if self.df is None:
            self.load_data()
            self.preprocess()
        return self.df
