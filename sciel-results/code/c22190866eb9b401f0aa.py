import numpy as np
from scipy.interpolate import PchipInterpolator
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 280.0  # Safety margin

    X = np.asarray(X_train, dtype=float).flatten()
    y = np.asarray(y_train, dtype=float).flatten()

    # Handle constant y
    if np.var(y) < 1e-12:
        mean_y = np.mean(y)
        return lambda X: np.full(np.asarray(X).shape, mean_y)

    # Sort by X
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Handle duplicates
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_sorted):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X

    if len(X_sorted) < 2:
        mean_y = np.mean(y)
        return lambda X: np.full(np.asarray(X).shape, mean_y)

    x_min, x_max = X_sorted[0], X_sorted[-1]
    x_range = x_max - x_min
    if x_range < 1e-12:
        mean_y = np.mean(y)
        return lambda X: np.full(np.asarray(X).shape, mean_y)

    # Strategy 1: PCHIP Interpolator (Robust, no overshoot)
    pchip_interp = None
    pchip_error = np.inf
    try:
        pchip_interp = PchipInterpolator(X_sorted, y_sorted)
        y_pred_pchip = pchip_interp(X_sorted)
        pchip_error = np.sum((y_pred_pchip - y_sorted)**2)
        if not np.all(np.isfinite(y_pred_pchip)):
            pchip_interp = None
            pchip_error = np.inf
    except Exception:
        pchip_interp = None

    # Strategy 2: Polynomial Fit (Try degrees 1 to 25)
    best_poly_deg = -1
    best_poly_coef = None
    best_poly_error = np.inf
    
    # Normalize X for better numerical stability in polyfit
    X_mean = np.mean(X_sorted)
    X_std = np.std(X_sorted)
    if X_std == 0:
        X_norm = X_sorted
    else:
        X_norm = (X_sorted - X_mean) / X_std

    for deg in range(1, 26):
        try:
            # Use np.polyfit on normalized X
            coef = np.polyfit(X_norm, y_sorted, deg)
            # Evaluate
            y_pred_poly = np.polyval(coef, X_norm)
            if not np.all(np.isfinite(y_pred_poly)):
                continue
            error = np.sum((y_pred_poly - y_sorted)**2)
            
            if error < best_poly_error:
                best_poly_error = error
                best_poly_coef = coef
                best_poly_deg = deg
        except Exception:
            continue
        
        # If error is very small, we can stop early
        if best_poly_error < 1e-12:
            break

    # Strategy 3: Fourier Basis Fit (for periodic functions)
    best_fourier_nf = -1
    best_fourier_coef = None
    best_fourier_error = np.inf
    
    # Fourier basis requires care with the range
    # We use a basis of sin/cos terms
    # To handle non-periodic functions, we might need to be careful,
    # but for periodic ones, this is powerful.
    
    # We'll try a few numbers of frequencies
    for nf in [4, 6, 8, 10, 12, 15, 20]:
        try:
            cols = [np.ones(len(X_sorted))]
            for k in range(1, nf + 1):
                cols.append(np.sin(2 * np.pi * k * X_sorted / x_range))
                cols.append(np.cos(2 * np.pi * k * X_sorted / x_range))
            F = np.column_stack(cols)
            
            # Solve least squares
            coef, residuals, rank, s = np.linalg.lstsq(F, y_sorted, rcond=None)
            y_pred_fourier = F @ coef
            
            if not np.all(np.isfinite(y_pred_fourier)):
                continue
                
            error = np.sum((y_pred_fourier - y_sorted)**2)
            
            if error < best_fourier_error:
                best_fourier_error = error
                best_fourier_coef = coef
                best_fourier_nf = nf
        except Exception:
            continue
        
        if best_fourier_error < 1e-12:
            break

    # Strategy 4: RBF Basis Fit (for localized features)
    best_rbf_nc = -1
    best_rbf_bw = -1
    best_rbf_coef = None
    best_rbf_error = np.inf
    
    # Try a few configurations
    for nc in [10, 15, 20, 25, 30, 40, 50]:
        for bw_mult in [0.5, 1.0, 1.5, 2.0]:
            try:
                centers = np.linspace(x_min, x_max, nc)
                bandwidth = x_range / nc * bw_mult
                
                cols = [np.ones(len(X_sorted))]
                for c in centers:
                    cols.append(np.exp(-((X_sorted - c) ** 2) / (2 * bandwidth ** 2)))
                F = np.column_stack(cols)
                
                # Solve least squares
                coef, residuals, rank, s = np.linalg.lstsq(F, y_sorted, rcond=None)
                y_pred_rbf = F @ coef
                
                if not np.all(np.isfinite(y_pred_rbf)):
                    continue
                    
                error = np.sum((y_pred_rbf - y_sorted)**2)
                
                if error < best_rbf_error:
                    best_rbf_error = error
                    best_rbf_coef = coef
                    best_rbf_nc = nc
                    best_rbf_bw = bandwidth
            except Exception:
                continue
            
            if best_rbf_error < 1e-12:
                break
        if best_rbf_error < 1e-12:
            break

    # Select the best model based on training error
    # We compare the errors of the available models
    candidates = []
    
    if pchip_interp is not None:
        candidates.append(('pchip', pchip_error))
    
    if best_poly_deg != -1:
        candidates.append(('poly', best_poly_error))
    
    if best_fourier_nf != -1:
        candidates.append(('fourier', best_fourier_error))
    
    if best_rbf_nc != -1:
        candidates.append(('rbf', best_rbf_error))
    
    if not candidates:
        # Fallback to linear interpolation
        def predict(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            y_pred = np.interp(X_eval, X_sorted, y_sorted)
            return y_pred.reshape(np.asarray(X).shape)
        return predict

    # Sort candidates by error
    candidates.sort(key=lambda x: x[1])
    best_model_name = candidates[0][0]
    
    if best_model_name == 'pchip':
        def predict(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            y_pred = pchip_interp(X_eval)
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            return y_pred.reshape(np.asarray(X).shape)
        return predict
        
    elif best_model_name == 'poly':
        def predict(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            X_eval_norm = (X_eval - X_mean) / X_std
            y_pred = np.polyval(best_poly_coef, X_eval_norm)
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            return y_pred.reshape(np.asarray(X).shape)
        return predict
        
    elif best_model_name == 'fourier':
        def predict(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            cols = [np.ones(len(X_eval))]
            for k in range(1, best_fourier_nf + 1):
                cols.append(np.sin(2 * np.pi * k * X_eval / x_range))
                cols.append(np.cos(2 * np.pi * k * X_eval / x_range))
            F = np.column_stack(cols)
            y_pred = F @ best_fourier_coef
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            return y_pred.reshape(np.asarray(X).shape)
        return predict
        
    elif best_model_name == 'rbf':
        def predict(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            centers = np.linspace(x_min, x_max, best_rbf_nc)
            bandwidth = best_rbf_bw
            
            cols = [np.ones(len(X_eval))]
            for c in centers:
                cols.append(np.exp(-((X_eval - c) ** 2) / (2 * bandwidth ** 2)))
            F = np.column_stack(cols)
            y_pred = F @ best_rbf_coef
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            return y_pred.reshape(np.asarray(X).shape)
        return predict

    # This should never be reached
    def predict(X):
        X_eval = np.asarray(X, dtype=float).flatten()
        y_pred = np.interp(X_eval, X_sorted, y_sorted)
        return y_pred.reshape(np.asarray(X).shape)
    return predict
