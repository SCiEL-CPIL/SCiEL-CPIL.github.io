"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

def fit(X_train, y_train):
    """
    Fit a regression model for 1D noiseless data using cubic spline interpolation.
    This is the primary approach since the data is noiseless and 1D.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Flatten if 2D with 1 column
    if X.ndim == 2 and X.shape[1] == 1:
        x = X.flatten()
    elif X.ndim == 1:
        x = X
    else:
        # Fallback for multi-dim (shouldn't happen per task description)
        x = X[:, 0] if X.ndim == 2 else X.flatten()
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicates if any
    unique_mask = np.concatenate(([True], np.diff(x_sorted) > 1e-12))
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    
    # Try cubic spline interpolation
    try:
        spline = CubicSpline(x_unique, y_unique, bc_type='not-a-knot')
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 2 and X_test.shape[1] == 1:
                x_test = X_test.flatten()
            elif X_test.ndim == 1:
                x_test = X_test
            else:
                x_test = X_test[:, 0] if X_test.ndim == 2 else X_test.flatten()
            pred = spline(x_test)
            # Ensure finite values
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            return pred
        return predict
    except Exception:
        pass
    
    # Fallback: linear interpolation
    try:
        from scipy.interpolate import interp1d
        interp = interp1d(x_unique, y_unique, kind='linear', fill_value='extrapolate')
        
        def predict_linear(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 2 and X_test.shape[1] == 1:
                x_test = X_test.flatten()
            elif X_test.ndim == 1:
                x_test = X_test
            else:
                x_test = X_test[:, 0] if X_test.ndim == 2 else X_test.flatten()
            pred = interp(x_test)
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            return pred
        return predict_linear
    except Exception:
        pass
    
    # Last resort: return mean
    y_mean = np.mean(y)
    def predict_mean(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        n = X_test.shape[0] if X_test.ndim > 0 else 1
        return np.full(n, y_mean)
    return predict_mean

# EVOLVE-BLOCK-END
