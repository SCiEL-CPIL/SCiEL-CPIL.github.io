"""
Strategy: High-resolution monotone cubic interpolation (PCHIP) on sorted training data.
This is a fundamentally different approach from the current GP-based symbolic regression.
Instead of searching for an exact formula, we treat the 200 noiseless training points as
a dense sample of the true function and use PCHIP to create a smooth, monotone-preserving
interpolant. This avoids the oscillatory artifacts of standard cubic splines and should
generalize well to test points in the same domain.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280
    
    X = X_train[:, 0]
    y = y_train
    n = len(X)
    
    # Sort by X for interpolation
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Fit PCHIP interpolator
    try:
        pchip = PchipInterpolator(X_sorted, y_sorted)
        
        def predict(X):
            X_flat = X.flatten()
            # Clip to training range to avoid extrapolation issues
            x_min = X_sorted[0]
            x_max = X_sorted[-1]
            X_clipped = np.clip(X_flat, x_min, x_max)
            pred = pchip(X_clipped)
            return pred.reshape(X.shape)
        
        return predict
    except Exception:
        # Fallback: linear interpolation
        try:
            def predict(X):
                X_flat = X.flatten()
                x_min = X_sorted[0]
                x_max = X_sorted[-1]
                X_clipped = np.clip(X_flat, x_min, x_max)
                pred = np.interp(X_clipped, X_sorted, y_sorted)
                return pred.reshape(X.shape)
            return predict
        except Exception:
            mean_y = np.mean(y)
            return lambda X: np.full(X.shape[0], mean_y)

# EVOLVE-BLOCK-END
