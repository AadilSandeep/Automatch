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
        else:
            base_dir = os.path.dirname(os.path.dirname(data_path))
            
        self.loader = DataLoader(data_path)
        raw_df = self.loader.get_data()
        
        # Load pre-trained models and df if they exist
        model_dir = os.path.join(base_dir, "models")
        df_path = os.path.join(model_dir, "processed_df.pkl")
        
        from src.utils import assign_budget_class
        
        if os.path.exists(df_path):
            self.df = pd.read_pickle(df_path)
            self.classifier = VehicleClassifier(self.df)
            if self.classifier.load_model(model_dir):
                return
                
        # If we reach here, we need to train and save
        self.df = assign_budget_class(raw_df)
        self.classifier = VehicleClassifier(self.df)
        self.classifier.train()
        
        # Pre-calculate budget predictions for all cars
        self.df["Predicted_Budget_Class"] = self.classifier.predict_budget_class(self.classifier.X)
        
        # Save for future app startups
        os.makedirs(model_dir, exist_ok=True)
        self.df.to_pickle(df_path)
        self.classifier.save_model(model_dir)

    def recommend(self, user_prefs):
        """
        Generates recommendations based on user preferences.
        user_prefs: dict containing 'budget', 'fuel', 'body', 'mileage', 'seats', 'power'
        """
        match_type = 'Exact Match'
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
            match_type = 'Near Match (Different Body Type)'
            mask = (
                (self.df["Predicted_Budget_Class"] == user_prefs['budget']) &
                (self.df["Fuel_Type"].str.lower().str.contains(user_prefs['fuel'].lower(), na=False))
            )
            filtered = self.df[mask].copy()

        # Relax constraints: Drop Fuel Type instead
        if filtered.empty:
            match_type = 'Near Match (Different Fuel Type)'
            mask = (
                (self.df["Predicted_Budget_Class"] == user_prefs['budget']) &
                (self.df["Body_Type"].str.lower().str.contains(user_prefs['body'].lower(), na=False))
            )
            filtered = self.df[mask].copy()
            
        # Relax constraints: Same Budget only
        if filtered.empty:
            match_type = 'Near Match (Similar Budget)'
            mask = (self.df["Predicted_Budget_Class"] == user_prefs['budget'])
            filtered = self.df[mask].copy()

        if filtered.empty:
            return pd.DataFrame() # Return empty if still no matches

        filtered['Match_Type'] = match_type

        # 2. Match Score Breakdown & Dynamic Scoring Stage
        u_mileage = float(user_prefs.get('mileage', 15.0))
        u_seats = float(user_prefs.get('seats', 5.0))
        u_power = float(user_prefs.get('power', 100.0))
        
        # Sub-scores (0 to 1)
        # Mileage: Penalize if lower than user preference, reward slightly if higher
        filtered['Mileage_Match'] = np.clip(1.0 - np.maximum(0, u_mileage - filtered['Mileage']) / max(u_mileage, 1), 0, 1)
        
        # Power: Penalize large deviations
        filtered['Power_Match'] = np.clip(1.0 - np.abs(filtered['Power'] - u_power) / max(u_power, 1), 0, 1)
        
        # Fuel Match: Exact fuel = 100%, otherwise 0%
        filtered['Fuel_Match'] = filtered['Fuel_Type'].apply(
            lambda x: 1.0 if pd.notna(x) and user_prefs['fuel'].lower() in str(x).lower() else 0.0
        )
        
        # Budget Match: we only keep items in the budget class
        filtered['Budget_Match'] = 1.0
        
        # Lifestyle Match (Seats & Power proxy)
        filtered['Seats_Match'] = np.clip(1.0 - np.abs(filtered['Seating_Capacity'] - u_seats) / 7.0, 0, 1)
        filtered['Lifestyle_Match'] = (filtered['Power_Match'] + filtered['Seats_Match']) / 2.0
        
        # Dynamic Weights based on Top Priority
        priority = user_prefs.get('priority_label', 'Balanced')
        weights = {'mileage': 0.35, 'lifestyle': 0.35, 'fuel': 0.30}
        
        if priority == 'Fuel Efficiency':
            weights = {'mileage': 0.55, 'lifestyle': 0.20, 'fuel': 0.25}
        elif priority == 'Performance':
            # Boost lifestyle/power
            weights = {'mileage': 0.15, 'lifestyle': 0.60, 'fuel': 0.25}
        elif priority == 'Safety':
            weights = {'mileage': 0.30, 'lifestyle': 0.40, 'fuel': 0.30} 
            
        filtered['Similarity_Score'] = (
            filtered['Mileage_Match'] * weights['mileage'] +
            filtered['Lifestyle_Match'] * weights['lifestyle'] +
            filtered['Fuel_Match'] * weights['fuel']
        )
        
        # Add slight random noise to resolve ties in identical variants
        filtered['Similarity_Score'] += np.random.uniform(0, 0.001, len(filtered))
        
        # Cap at 1.0
        filtered['Similarity_Score'] = np.clip(filtered['Similarity_Score'], 0, 1.0)
        
        # Sort and Deduplicate
        results = filtered.sort_values("Similarity_Score", ascending=False)
        results = results.drop_duplicates(subset=["Make", "Model"], keep="first").head(5).copy()
        
        # 3. Assign Badges
        if not results.empty:
            max_mil = results['Mileage'].max()
            results['Badges'] = results.apply(lambda row: self._assign_badges(row, max_mil, user_prefs), axis=1)
        else:
            results['Badges'] = [[] for _ in range(len(results))]
            
        return results

    def _assign_badges(self, row, max_mil, user_prefs):
        badges = []
        if pd.notna(row['Mileage']) and row['Mileage'] >= max_mil and row['Mileage'] >= 18:
            badges.append("Best Mileage")
        if row.get('Seating_Capacity', 5) >= 7 and user_prefs.get('usage_label') == 'Family Trips':
            badges.append("Family Pick")
        if row.get('Power', 0) > 120 and user_prefs.get('priority_label') == 'Performance':
            badges.append("Top Performance")
        if user_prefs.get('experience_label') == 'First-time Buyer' and row.get('Power', 100) <= 90:
            badges.append("Beginner Friendly")
        if row.get('Similarity_Score', 0) >= 0.90:
            badges.append("Top Match")
        
        # Return at most 2 badges to keep UI clean
        return badges[:2]

    def get_explanation(self, row, user_prefs):
        """Generates a detailed, personalized explanation for the recommendation."""
        # Extract inputs
        make = row.get("Make", "This model")
        model = row.get("Model", "")
        
        price = row.get("Ex-Showroom_Price", 0)
        mileage = row.get("Mileage", 0)
        power = row.get("Power", 0)
        torque = row.get("Torque", 0)
        displacement = row.get("Displacement", 0)
        seats = int(row.get("Seating_Capacity", 5))
        fuel_t = row.get("Fuel_Type", "")
        body_t = row.get("Body_Type", "")
        wheelbase = row.get("Wheelbase", 0)
        ground_clearance = row.get("Ground_Clearance", "")
        boot_space = row.get("Boot_Space", "")
        airbags = row.get("Number_of_Airbags", 0)
        
        # User Preferences
        u_budget = user_prefs.get("budget", "Mid")
        u_fuel = user_prefs.get("fuel", "Petrol")
        u_body = user_prefs.get("body", "SUV")
        u_priority = user_prefs.get("priority_label", "Balanced")
        u_usage = user_prefs.get("usage_label", "City Commute")
        u_experience = user_prefs.get("experience_label", "Experienced")

        reasons = []

        # 1. Budget & Match Type
        match_type = row.get("Match_Type", "Exact Match")
        if match_type == "Exact Match":
            reasons.append(f"Priced at ₹{price:,.0f}, it fits your {u_budget.lower()}-budget {u_fuel.lower()} {u_body.lower()} requirements perfectly.")
        else:
            reasons.append(f"At ₹{price:,.0f}, it serves as a highly compatible {u_budget.lower()}-budget alternative.")

        # 2. Priority Alignment
        if u_priority == "Fuel Efficiency":
            if pd.notna(mileage) and mileage > 0:
                reasons.append(f"It aligns with your focus on fuel efficiency, offering an excellent mileage of {mileage:.1f} kmpl.")
            else:
                reasons.append("It features a highly optimized engine designed to keep running costs low.")
        elif u_priority == "Performance":
            perf_details = []
            if pd.notna(power) and power > 0:
                perf_details.append(f"{int(power)} BHP")
            if pd.notna(torque) and torque > 0:
                perf_details.append(f"{int(torque)} Nm torque")
            if pd.notna(displacement) and displacement > 0:
                perf_details.append(f"{int(displacement)}cc engine")
            
            if perf_details:
                reasons.append(f"It satisfies your performance focus with a responsive {', '.join(perf_details)} setup.")
            else:
                reasons.append("It boasts a performance-tuned engine providing punchy power delivery.")
        elif u_priority == "Safety":
            safety_features = []
            if pd.notna(airbags) and airbags > 0:
                safety_features.append(f"{int(airbags)} airbags")
            
            abs_sys = row.get("ABS_(Anti-lock_Braking_System)", "")
            if pd.notna(abs_sys) and ("yes" in str(abs_sys).lower() or str(abs_sys).strip() in ["1", "1.0"]):
                safety_features.append("ABS")
                
            ebd_sys = row.get("EBD_(Electronic_Brake-force_Distribution)", "")
            if pd.notna(ebd_sys) and ("yes" in str(ebd_sys).lower() or str(ebd_sys).strip() in ["1", "1.0"]):
                safety_features.append("EBD")

            if safety_features:
                reasons.append(f"It prioritizes safety with features like {', '.join(safety_features)} to protect you and your passengers.")
            else:
                reasons.append("It offers a solid build quality with essential safety systems for peace of mind.")
        else: # Balanced
            reasons.append(f"It delivers a highly balanced package combining a {power:.0f} BHP engine with a healthy {mileage:.1f} kmpl fuel economy.")

        # 3. Usage & Seating & Dimensions Alignment
        if u_usage == "City Commute":
            city_details = []
            if pd.notna(mileage) and mileage > 16:
                city_details.append(f"efficient {mileage:.1f} kmpl city mileage")
            if pd.notna(row.get("Length", 0)) and row.get("Length", 0) < 4000:
                city_details.append("sub-4 meter compact size for easy parking")
            
            if city_details:
                reasons.append(f"For city commuting, it is highly practical due to its {', and '.join(city_details)}.")
            else:
                reasons.append("It is highly agile and easy to navigate through congested city traffic.")
        elif u_usage == "Highway/Long Distance":
            highway_details = []
            if pd.notna(power) and power >= 90:
                highway_details.append(f"capable {int(power)} BHP engine for high-speed stability")
            if pd.notna(wheelbase) and wheelbase > 2500:
                highway_details.append("longer wheelbase for a planted highway ride")
                
            if highway_details:
                reasons.append(f"For long-distance highway cruising, it offers a {', and '.join(highway_details)}.")
            else:
                reasons.append("It offers excellent stability and comfort for extended highway road trips.")
        elif u_usage == "Off-roading/Rough Terrain":
            offroad_details = []
            if pd.notna(ground_clearance) and str(ground_clearance).strip() and str(ground_clearance).strip().lower() != "nan":
                offroad_details.append(f"impressive {ground_clearance} ground clearance")
            drivetrain = row.get("Drivetrain", "")
            if pd.notna(drivetrain) and any(d in str(drivetrain).lower() for d in ["awd", "4wd", "rwd"]):
                offroad_details.append(f"capable {drivetrain} drivetrain")

            if offroad_details:
                reasons.append(f"It handles rough terrains and bad roads easily thanks to its {', combined with '.join(offroad_details)}.")
            else:
                reasons.append("Its robust suspension setup and high stance make it ideal for handling rough patches and uneven roads.")
        elif u_usage == "Family Trips":
            family_details = []
            if seats >= 6:
                family_details.append(f"spacious {seats}-seater layout")
            elif seats >= 5:
                family_details.append(f"comfortable {seats}-seater cabin")
            
            if pd.notna(boot_space) and str(boot_space).strip() and str(boot_space).strip().lower() != "nan":
                family_details.append(f"generous {boot_space} cargo space")

            if family_details:
                reasons.append(f"Ideal for family trips, providing a {' and '.join(family_details)}.")
            else:
                reasons.append("It provides a spacious and comfortable cabin with plenty of room for family passengers and luggage.")
        elif u_usage == "Mixed":
            reasons.append("It serves as a versatile all-rounder, offering the perfect blend of daily city drivability and weekend trip capability.")

        # 4. Driver Experience
        if u_experience == "First-time Buyer":
            driving_aids = []
            ps = row.get("Power_Steering", "")
            if pd.notna(ps) and any(w in str(ps).lower() for w in ["yes", "electric", "hydraulic", "electro-hydraulic"]):
                driving_aids.append("light power steering")
            pa = row.get("Parking_Assistance", "")
            if pd.notna(pa) and any(w in str(pa).lower() for w in ["sensor", "camera", "yes"]):
                driving_aids.append("parking assistance")
            
            if driving_aids:
                reasons.append(f"As a first-time buyer, you will appreciate its easy-to-drive nature, featuring {', and '.join(driving_aids)}.")
            elif pd.notna(power) and power <= 95:
                reasons.append("Its smooth, predictable power delivery makes it exceptionally user-friendly for beginner drivers.")
            else:
                reasons.append("It features intuitive controls and excellent road visibility, making it highly reassuring for new drivers.")
        else:
            reasons.append("For an experienced driver, its confident handling and engaging road dynamics make it a pleasure to own and drive.")

        # Combine these reasons into a smooth paragraph
        return " ".join(reasons)

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
        elif usage == 'Mixed':
            specs['mileage'] = 16.0
            specs['power'] = 85.0
            
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
        elif priority == 'Safety':
            specs['power'] = min(specs.get('power', 100), 100.0)
            
        return specs
