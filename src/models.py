from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class VehicleClassifier:
    def __init__(self, df):
        self.df = df
        self.rf_model = None
        self.cart_model = None
        self.preprocessor = None
        self.X = None
        self.y = None
        self._prepare_data()

    def _prepare_data(self):
        """Separates features and target, defines preprocessor."""
        # Ensure Ex-Showroom_Price is not in X
        drop_cols = ["Budget_Class", "Ex-Showroom_Price"]
        # Also drop predicted columns or similarity scores if they exist from previous runs (safety check)
        drop_cols += [c for c in ["Predicted_Budget_Class", "Similarity_Score"] if c in self.df.columns]
        
        self.X = self.df.drop(columns=drop_cols, errors='ignore')
        self.y = self.df["Budget_Class"]
        
        # Dynamically classify columns by data types
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols_raw = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Drop high-cardinality categorical features (noise) heuristically to prevent OHE explosion
        categorical_cols = [col for col in categorical_cols_raw if self.X[col].nunique() < 30]
        # Ensure we drop the discarded high-cardinality columns from X completely
        self.X = self.X.drop(columns=[c for c in categorical_cols_raw if c not in categorical_cols], errors='ignore')

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, [c for c in numerical_cols if c in self.X.columns]),
            ('cat', categorical_transformer, [c for c in categorical_cols if c in self.X.columns])
        ])

    def train(self):
        """Trains the Random Forest and Decision Tree models."""
        # Random Forest with Algorithmic Feature Selection
        self.rf_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('variance_threshold', VarianceThreshold()),
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold='median')),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        self.rf_model.fit(self.X, self.y)

        # CART for potential explainability
        self.cart_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('variance_threshold', VarianceThreshold()),
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold='median')),
            ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
        ])
        self.cart_model.fit(self.X, self.y)

    def predict_budget_class(self, df_features):
        """Predicts the budget class for the given features."""
        # Ensure input df has same columns as X (preprocessor handles missing ones via ignore, but helpful to match)
        return self.rf_model.predict(df_features)

    def get_feature_importance(self):
        """Returns feature importance from Random Forest."""
        if not self.rf_model:
            return {}
        
        # Step 1: Preprocessor names
        feature_names = self.rf_model.named_steps['preprocessor'].get_feature_names_out()
        
        # Step 2: Variance threshold mask
        var_mask = self.rf_model.named_steps['variance_threshold'].get_support()
        var_features = [feature_names[i] for i, m in enumerate(var_mask) if m]
        
        # Step 3: SelectFromModel mask
        sel_mask = self.rf_model.named_steps['feature_selection'].get_support()
        final_features = [var_features[i] for i, m in enumerate(sel_mask) if m]
        
        # Step 4: Classifier importances
        importances = self.rf_model.named_steps['classifier'].feature_importances_
        
        return dict(zip(final_features, importances))
