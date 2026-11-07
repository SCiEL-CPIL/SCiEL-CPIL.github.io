import numpy as np
from scipy.interpolate import CubicSpline

def fit(X_train, y_train):
    X_train = np.asarray(X_train).flatten()
    y_train = np.asarray(y_train).flatten()
    
    # Sort by X
    sort_idx = np.argsort(X_train)
    X_sorted = X_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    # Handle potential duplicate X values
    unique_x, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    
    if len(unique_x) < len(X_sorted):
        y_sum = np.zeros_like(unique_x)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_avg = y_sum / counts
        X_fit = unique_x
        y_fit = y_avg
    else:
        X_fit = X_sorted
        y_fit = y_sorted
        
    # Need at least 4 points for CubicSpline with not-a-knot? 
    # Actually, CubicSpline requires at least 2 points, but for cubic, 
    # it handles small N by falling back or raising error? 
    # Documentation: "If the number of points is less than 4, the spline is linear." 
    # Wait, no. CubicSpline requires x to be strictly increasing.
    # If N < 4, it might fail or behave unexpectedly. 
    # We have 200 points, so this is safe.
    
    try:
        spline = CubicSpline(X_fit, y_fit)
    except Exception:
        # Fallback to linear interpolation
        from scipy.interpolate import interp1d
        interp_func = interp1d(X_fit, y_fit, kind='linear', fill_value="extrapolate")
        def predict(X):
            X = np.asarray(X).flatten()
            return interp_func(X)
        return predict

    def predict(X):
        X = np.asarray(X).flatten()
        return spline(X)

    return predict
