"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE_MARGIN = 10.0  # seconds before the 330s wall-clock limit to stop
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    def time_left():
        return (start_time + 330.0 - DEADLINE_MARGIN) - time.time()

    # Handle constant y
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        c = y_mean
        return lambda X: np.full(len(X), c)

    # Piecewise linear interpolation approach
    # Sort training points by x
    if d == 1:
        x = X[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Check for duplicate x values - if so, average y
        unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        if len(unique_x) < n:
            # Have duplicates, need to handle
            y_sum = np.zeros(len(unique_x))
            np.add.at(y_sum, inverse, y_sorted)
            y_unique = y_sum / counts
            x_unique = unique_x
        else:
            x_unique = x_sorted
            y_unique = y_sorted
        
        # For prediction, use np.interp which handles out-of-bounds by clamping
        def predict_interp(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            x_pred = X_pred[:, 0]
            return np.interp(x_pred, x_unique, y_unique)
        
        # Verify the fit quality
        y_pred_train = predict_interp(X)
        sse = np.sum((y - y_pred_train)**2)
        r2 = 1 - sse / np.sum((y - y_mean)**2)
        
        # If interpolation is good (which it should be for noiseless data), return it
        if r2 > 0.99:
            return predict_interp
    
    # Fallback: try polynomial regression with increasing degrees
    y_centered = y - y_mean
    y_norm = np.linalg.norm(y_centered)
    
    if y_norm < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    best_w = None
    best_r2 = -np.inf
    best_degree = 1
    
    # Try polynomial degrees from 1 to 20
    for degree in range(1, 21):
        if d == 1:
            x = X[:, 0]
            X_poly = np.column_stack([x**p for p in range(degree + 1)])
        else:
            # For multi-dimensional, use simple polynomial features
            X_poly = np.ones((n, 1))
            for p in range(1, degree + 1):
                for i in range(d):
                    X_poly = np.column_stack([X_poly, X[:, i]**p])
        
        try:
            w, *_ = np.linalg.lstsq(X_poly, y, rcond=None)
            y_hat = X_poly @ w
            sse = np.sum((y - y_hat)**2)
            r2 = 1 - sse / np.sum(y_centered**2)
            
            if r2 > best_r2:
                best_r2 = r2
                best_w = w
                best_degree = degree
            
            # If we have a very good fit, stop early
            if r2 > 0.9999:
                break
                
        except:
            continue
        
        if time_left() < 5.0:
            break
    
    # Create prediction function
    def predict_poly(X_pred):
        X_pred = np.asarray(X_pred, dtype=np.float64)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(-1, 1)
        n_pred = len(X_pred)
        d_pred = X_pred.shape[1]
        
        if d_pred == 1:
            x_pred = X_pred[:, 0]
            X_poly_pred = np.column_stack([x_pred**p for p in range(best_degree + 1)])
        else:
            X_poly_pred = np.ones((n_pred, 1))
            for p in range(1, best_degree + 1):
                for i in range(d_pred):
                    X_poly_pred = np.column_stack([X_poly_pred, X_pred[:, i]**p])
        
        return X_poly_pred @ best_w
    
    return predict_poly

# EVOLVE-BLOCK-END
