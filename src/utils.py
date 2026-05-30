import pandas as pd
import numpy as np

def assign_budget_class(train_df, test_df=None):
    """
    Computes budget bins on train_df and applies them to both train_df and test_df.
    This prevents data leakage from the test set's price distribution into the training set.
    """
    train_copy = train_df.copy()
    
    # Calculate quantiles on train set only
    _, bins = pd.qcut(train_copy["Ex-Showroom_Price"], q=3, retbins=True)
    
    # Ensure extreme values are captured
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Apply bins to train set
    train_copy["Budget_Class"] = pd.cut(
        train_copy["Ex-Showroom_Price"],
        bins=bins,
        labels=["Low", "Mid", "High"]
    )
    
    if test_df is not None:
        test_copy = test_df.copy()
        # Apply the SAME bins to the test set
        test_copy["Budget_Class"] = pd.cut(
            test_copy["Ex-Showroom_Price"],
            bins=bins,
            labels=["Low", "Mid", "High"]
        )
        return train_copy, test_copy
        
    return train_copy
