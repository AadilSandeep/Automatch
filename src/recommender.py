import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import DataLoader
from src.models import VehicleClassifier
import os

class VehicleRecommender:
    def __init__(self, data_path=None):
        if data_path is None:
            # Default to cars.csv in data folder relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "cars.csv")
            
        self.loader = DataLoader(data_path)
        self.df = self.loader.get_data()
        self.classifier = VehicleClassifier(self.df)
        self.classifier.train()
        
        # Pre-calculate budget predictions for all cars
        # This is where the ML "Constrained-based Filtering" comes in
        self.df["Predicted_Budget_Class"] = self.classifier.predict_budget_class(self.classifier.X)

    def recommend(self, user_prefs):
        """
        Generates recommendations based on user preferences.
        user_prefs: dict containing 'budget', 'fuel', 'body', 'mileage', 'seats', 'power'
        """
        # 1. Classification & Filtering Stage
        # Strict filtering based on predicted budget class and user constraints
        mask = (
            (self.df["Predicted_Budget_Class"] == user_prefs['budget']) &
            (self.df["Fuel_Type"].str.lower().str.contains(user_prefs['fuel'].lower(), na=False)) &
            (self.df["Body_Type"].str.lower().str.contains(user_prefs['body'].lower(), na=False))
        )
        filtered = self.df[mask].copy()
        
        # Relax constraints: Drop Body Type if no matches found
        if filtered.empty:
            mask = (
                (self.df["Predicted_Budget_Class"] == user_prefs['budget']) &
                (self.df["Fuel_Type"].str.lower().str.contains(user_prefs['fuel'].lower(), na=False))
            )
            filtered = self.df[mask].copy()

        if filtered.empty:
            return pd.DataFrame() # Return empty if still no matches

        # 2. Similarity Ranking Stage
        rank_features = ["Mileage", "Seating_Capacity", "Power", "Displacement"]
        
        # Handle features for similarity
        # Fill NaNs with 0 or median? Median of filtered is safer, or 0 if single row.
        candidate_data = filtered[rank_features].fillna(0)
        
        # Scale candidate data
        scaler = MinMaxScaler()
        if len(candidate_data) > 0:
            candidate_scaled = scaler.fit_transform(candidate_data)
        else:
             return pd.DataFrame()

        # Construct User Vector
        # Infer Displacement (CC) if not provided, based on mileage preference
        # (Logic preserved from original main.py)
        u_mileage = float(user_prefs.get('mileage', 15.0))
        u_seats = float(user_prefs.get('seats', 5.0))
        u_power = float(user_prefs.get('power', 100.0))
        
        u_cc = 1200.0 if u_mileage > 18 else 1500.0 
        
        user_vector = pd.DataFrame(
            [[u_mileage, u_seats, u_power, u_cc]],
            columns=rank_features
        )
        
        # Scale user vector
        user_scaled = scaler.transform(user_vector)
        
        # Calculate Cosine Similarity
        filtered["Similarity_Score"] = cosine_similarity(user_scaled, candidate_scaled)[0]
        
        # Sort and Deduplicate
        results = filtered.sort_values("Similarity_Score", ascending=False)
        results = results.drop_duplicates(subset=["Make", "Model"], keep="first")
        
        return results.head(5)

    def get_explanation(self, row, user_prefs):
        """Generates a simple text explanation for a recommendation."""
        reasons = []
        
        # Match Strength
        score = row.get("Similarity_Score", 0)
        if score > 0.9:
            reasons.append("Excellent match")
        elif score > 0.7:
            reasons.append("Good match")
            
        # Feature highlights
        if row["Mileage"] >= user_prefs.get('mileage', 15) - 2:
            reasons.append(f"Good Mileage ({row['Mileage']:.1f} kmpl)")
            
        if row["Power"] >= user_prefs.get('power', 100) - 10:
             reasons.append(f"Powerful Engine ({row['Power']} BHP)")

        return ", ".join(reasons)

    def map_lifestyle_to_specs(self, lifestyle_inputs):
        """
        Translates user lifestyle answers into technical specifications.
        lifestyle_inputs: dict with keys 'usage', 'experience', 'passengers', 'priority'
        """
        specs = {}
        
        # 1. Usage -> Mileage & Body Type & Power
        usage = lifestyle_inputs.get('usage', 'City Commute')
        if usage == 'City Commute':
            specs['mileage'] = 18.0  # High mileage for city
            specs['power'] = 70.0    # Moderate power enough
        elif usage == 'Highway/Long Distance':
            specs['mileage'] = 15.0
            specs['power'] = 100.0   # Need more passing power
        elif usage == 'Off-roading/Rough Terrain':
            specs['mileage'] = 10.0
            specs['power'] = 120.0
            specs['body'] = 'SUV'   # Strong preference
        elif usage == 'Family Trips':
            specs['mileage'] = 14.0
            specs['power'] = 90.0
            
        # 2. Passengers -> Seats & Body Type
        passengers = lifestyle_inputs.get('passengers', '3-4')
        if passengers == '1-2':
            specs['seats'] = 5 # Standard
        elif passengers == '5+':
            specs['seats'] = 7
            specs['body'] = 'SUV' # Start with SUV for space
            
        # 3. Experience -> Power Cap (Soft Constraint)
        exp = lifestyle_inputs.get('experience', 'Experienced')
        if exp == 'First-time Buyer':
            # Cap power preference to avoid overwhelming machines
            specs['power'] = min(specs.get('power', 100), 90.0) 
            
        # 4. Priority -> Adjustments
        priority = lifestyle_inputs.get('priority', 'Balanced')
        if priority == 'Fuel Efficiency':
            specs['mileage'] = max(specs.get('mileage', 15), 20.0)
        elif priority == 'Performance':
            specs['power'] = max(specs.get('power', 100), 120.0)
            
        return specs
