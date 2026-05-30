import pandas as pd
import numpy as np
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.models import VehicleClassifier
from src.utils import assign_budget_class
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
import warnings

warnings.filterwarnings('ignore')

def evaluate():
    print("\n" + "="*50)
    print(" MODEL EVALUATION & VALIDATION REPORT ")
    print("="*50)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cars.csv")
    
    loader = DataLoader(data_path)
    raw_df = loader.get_data()
    
    print(f"\n[1] Data Loaded: {len(raw_df)} samples available.")
    
    # Simple Train/Test Split (80/20)
    print("\n--- A. Simple Train/Test Holdout Evaluation ---")
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
    
    # IMPORTANT: Compute budget class safely to avoid test leakage
    train_df, test_df = assign_budget_class(train_df, test_df)
    
    # Train
    classifier = VehicleClassifier(train_df)
    classifier.train()
    
    # Test
    drop_cols = ["Budget_Class", "Ex-Showroom_Price"]
    drop_cols += [c for c in ["Predicted_Budget_Class", "Similarity_Score"] if c in test_df.columns]
    
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df["Budget_Class"]
    
    rf_preds = classifier.rf_model.predict(X_test)
    cart_preds = classifier.cart_model.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_preds)
    cart_acc = accuracy_score(y_test, cart_preds)
    
    # Dummy baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(classifier.X, classifier.y)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
    
    print(f"Dummy Baseline Accuracy: {dummy_acc:.2%}")
    print(f"CART Decision Tree Accuracy: {cart_acc:.2%}")
    print(f"Random Forest Accuracy:  {rf_acc:.2%}")
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_preds))
    
    print("Confusion Matrix (RF):")
    print(pd.DataFrame(
        confusion_matrix(y_test, rf_preds, labels=["Low", "Mid", "High"]),
        index=["True Low", "True Mid", "True High"],
        columns=["Pred Low", "Pred Mid", "Pred High"]
    ))
    
    print("\n--- B. Stratified 5-Fold Cross-Validation ---")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use a dummy target for stratification only
    _, cv_bins = pd.qcut(raw_df["Ex-Showroom_Price"], q=3, retbins=True)
    cv_bins[0] = -np.inf; cv_bins[-1] = np.inf
    dummy_y = pd.cut(raw_df["Ex-Showroom_Price"], bins=cv_bins, labels=["Low", "Mid", "High"])
    
    fold_accs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(raw_df, dummy_y)):
        fold_train = raw_df.iloc[train_idx].copy()
        fold_test = raw_df.iloc[test_idx].copy()
        
        # Strict evaluation: assign budget bins on train only
        fold_train, fold_test = assign_budget_class(fold_train, fold_test)
        
        clf = VehicleClassifier(fold_train)
        clf.train()
        
        fold_X_test = fold_test.drop(columns=["Budget_Class", "Ex-Showroom_Price"], errors='ignore')
        fold_y_test = fold_test["Budget_Class"]
        
        acc = accuracy_score(fold_y_test, clf.predict_budget_class(fold_X_test))
        fold_accs.append(acc)
        
    cv_mean = np.mean(fold_accs)
    cv_std = np.std(fold_accs)
    print(f"5-Fold CV Random Forest Accuracy: {cv_mean:.2%} (+/- {cv_std*2:.2%})")
    
    print("\n--- C. Top Feature Importances ---")
    importances = classifier.get_feature_importance()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in sorted_imp:
        print(f"{feat:30s}: {imp:.4f}")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    evaluate()
