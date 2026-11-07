import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    x = X.flatten()
    y_flat = y.flatten()
    
    if np.var(y_flat) < 1e-10:
        mean_y = np.mean(y_flat)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) != len(x_sorted):
        y_sum = np.zeros_like(unique_x, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        x_sorted = unique_x
        
    if len(x_sorted) < 4:
        mean_y = np.mean(y_flat)
        return lambda X: np.full_like(np.asarray(X).flatten(), mean_y).reshape(np.asarray(X).shape)

    x_min, x_max = x_sorted[0], x_sorted[-1]
    if x_max == x_min:
        const_val = y_sorted[0]
        return lambda X: np.full_like(np.asarray(X).flatten(), const_val).reshape(np.asarray(X).shape)
    
    n = len(x_sorted)
    split = int(n * 0.9)
    x_tr, y_tr = x_sorted[:split], y_sorted[:split]
    x_val, y_val = x_sorted[split:], y_sorted[split:]
    
    best_interp = None
    best_err = float('inf')
    
    # Try PCHIP first (monotonicity preserving, good for smooth functions)
    try:
        pchip = PchipInterpolator(x_tr, y_tr)
        y_pred = pchip(x_val)
        err = np.mean((y_pred - y_val)**2)
        if err < best_err:
            best_err = err
            best_interp = pchip
    except Exception:
        pass
    
    # Try Cubic Spline
    try:
        cs = CubicSpline(x_tr, y_tr)
        y_pred = cs(x_val)
        err = np.mean((y_pred - y_val)**2)
        if err < best_err:
            best_err = err
            best_interp = cs
    except Exception:
        pass
    
    # If neither worked well, try Chebyshev as fallback
    if best_interp is None or best_err > np.var(y_tr) * 0.1:
        X_norm = (x_sorted - (x_max + x_min)/2) / ((x_max - x_min)/2)
        X_tr_norm = (x_tr - (x_max + x_min)/2) / ((x_max - x_min)/2)
        X_val_norm = (x_val - (x_max + x_min)/2) / ((x_max - x_min)/2)
        
        best_c = None
        best_deg = 10
        min_err = best_err
        
        for deg in [10, 20, 30, 50, 80, 100]:
            if time.time() > deadline:
                break
            try:
                c = np.polynomial.chebyshev.chebfit(X_tr_norm, y_tr, deg)
                y_pred = np.polynomial.chebyshev.chebval(X_val_norm, c)
                err = np.mean((y_pred - y_val)**2)
                if err < min_err:
                    min_err = err
                    best_c = c
                    best_deg = deg
            except Exception:
                continue
        
        if best_c is not None:
            try:
                c_final = np.polynomial.chebyshev.chebfit(X_norm, y_sorted, best_deg)
            except Exception:
                c_final = best_c
            
            def predict_cheb(X):
                X_eval = np.asarray(X).flatten()
                X_eval_norm = (X_eval - (x_max + x_min)/2) / ((x_max - x_min)/2)
                y_pred = np.polynomial.chebyshev.chebval(X_eval_norm, c_final)
                return y_pred.reshape(np.asarray(X).shape)
            
            return predict_cheb
    
    # Refit on all data with the best method
    if isinstance(best_interp, PchipInterpolator):
        final_interp = PchipInterpolator(x_sorted, y_sorted)
    else:
        final_interp = CubicSpline(x_sorted, y_sorted)
    
    def predict(X):
        X_eval = np.asarray(X).flatten()
        y_pred = final_interp(X_eval)
        return y_pred.reshape(np.asarray(X).shape)
    
    return predict
