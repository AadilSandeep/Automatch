import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.models import VehicleClassifier
from src.utils import assign_budget_class

def test_no_leakage():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cars.csv")
    
    loader = DataLoader(data_path)
    df = loader.get_data()
    
    # Assign budget class (simulation of training phase)
    df = assign_budget_class(df)
    
    # Add fake leakage columns just to ensure they are dropped
    df["Predicted_Budget_Class"] = "Low"
    df["Similarity_Score"] = 0.99
    
    clf = VehicleClassifier(df)
    
    # The models.py _prepare_data should have dropped them from self.X
    forbidden_cols = ["Budget_Class", "Ex-Showroom_Price", "Predicted_Budget_Class", "Similarity_Score"]
    
    for col in forbidden_cols:
        assert col not in clf.X.columns, f"LEAKAGE DETECTED: {col} is in classifier features!"
        
    print("Leakage test passed: No forbidden columns in features.")
    
if __name__ == "__main__":
    test_no_leakage()
