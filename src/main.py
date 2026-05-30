import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = "cars.csv" if os.path.exists("cars.csv") else os.path.join(BASE_DIR, "data", "cars.csv")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    exit()

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df["Ex-Showroom_Price"] = (
    df["Ex-Showroom_Price"]
    .str.replace("Rs.", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["Ex-Showroom_Price"] = pd.to_numeric(df["Ex-Showroom_Price"], errors="coerce")
df = df.dropna(subset=["Ex-Showroom_Price"])

for col in ["Displacement", "Power", "Torque"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract(r"(\d+\.?\d*)")[0]
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "City_Mileage" not in df.columns:
    df["Mileage"] = 25 - (df["Displacement"] / 200) 
    df["Mileage"] = df["Mileage"].clip(lower=5, upper=25) 
else:
    df["Mileage"] = df["City_Mileage"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

df = df.dropna(subset=["Displacement", "Power", "Mileage"])

df["Budget_Class"] = pd.qcut(
    df["Ex-Showroom_Price"],
    q=3,
    labels=["Low", "Mid", "High"]
)

X = df.drop(columns=["Budget_Class", "Ex-Showroom_Price"])
y = df["Budget_Class"]

numerical_cols = ["Displacement", "Power", "Torque", "Mileage", "Seating_Capacity"]
categorical_cols = ["Make", "Fuel_Type", "Body_Type"]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, [c for c in numerical_cols if c in X.columns]),
    ('cat', categorical_transformer, [c for c in categorical_cols if c in X.columns])
])

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

cart_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])

# NOTE: Evaluation logic (train_test_split, cross-validation) has been moved to src/evaluate_model.py.
# This script is the production app flow.
print("\n" + "="*50)
print(" INTELLIGENT VEHICLE RECOMMENDATION SYSTEM")
print("="*50)

while True:
    user_budget = input("\n[1] Your Budget (Low / Mid / High): ").strip().capitalize()
    if user_budget in ["Low", "Mid", "High"]: break

user_fuel = input("[2] Preferred Fuel (Petrol/Diesel/CNG): ").strip().lower()
user_body = input("[3] Body Type (SUV/Sedan/Hatchback): ").strip().lower()

print("\n[4] Enter your Preferences:")

def get_input(prompt, default):
    try:
        val = input(prompt)
        return float(val) if val.strip() != "" else default
    except ValueError:
        return default

u_mileage = get_input("   - Desired Mileage (kmpl) [e.g., 18]: ", 15.0)
u_seats = get_input("   - Seating Capacity [e.g., 5]: ", 5.0)
u_power = get_input("   - Minimum Power (BHP) [Optional]: ", 100.0)

u_cc = 1200.0 if u_mileage > 18 else 1500.0 
u_torque = 110.0 if u_power < 90 else 150.0

# Production fit on all available data for final recommendations
rf_model.fit(X, y)
df["Predicted_Budget_Class"] = rf_model.predict(X)

filtered_df = df[
    (df["Predicted_Budget_Class"] == user_budget) &
    (df["Fuel_Type"].str.lower().str.contains(user_fuel, na=False)) &
    (df["Body_Type"].str.lower().str.contains(user_body, na=False))
].copy()

if filtered_df.empty:
    filtered_df = df[
        (df["Predicted_Budget_Class"] == user_budget) &
        (df["Fuel_Type"].str.lower().str.contains(user_fuel, na=False))
    ].copy()

if filtered_df.empty:
    print("No vehicles found matching criteria.")
    exit()

rank_features = ["Mileage", "Seating_Capacity", "Power", "Displacement"]
candidate_data = filtered_df[rank_features].fillna(filtered_df[rank_features].median())

scaler = MinMaxScaler()
candidate_scaled = scaler.fit_transform(candidate_data)

user_vector = pd.DataFrame(
    [[u_mileage, u_seats, u_power, u_cc]],
    columns=rank_features
)
user_scaled = scaler.transform(user_vector)

filtered_df["Similarity_Score"] = cosine_similarity(user_scaled, candidate_scaled)[0]

filtered_df = filtered_df.sort_values("Similarity_Score", ascending=False)
filtered_df = filtered_df.drop_duplicates(subset=["Make", "Model"], keep="first")
top_recommendations = filtered_df.head(5)

print("\n" + "="*50)
print(f" TOP RECOMMENDATIONS FOR {user_budget.upper()} BUDGET")
print("="*50)

for i, row in top_recommendations.iterrows():
    print(f"\n#{i+1}: {row['Make']} {row['Model']} ({row['Variant']})")
    print(f"   Match Score: {row['Similarity_Score']:.1%} match to your needs")
    print(f"   Price: Rs. {row['Ex-Showroom_Price']:,.0f}")
    print(f"   Key Specs: {row['Mileage']:.1f} kmpl | {row['Seating_Capacity']} Seats | {row['Power']} BHP")