import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Step 1: Denoise the target y using local Gaussian kernel regression
    # This reduces noise, allowing the polynomial to better capture the underlying signal
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.zeros(6, dtype=np.float64)
    
    # Bandwidth selection: use a fraction of the range
    bandwidth = 0.03 * x_range
    
    # For efficiency, subsample the reference set if large
    max_sub = 3000
    if n > max_sub:
        np.random.seed(42)
        sub_idx = np.random.choice(n, max_sub, replace=False)
        x_ref = x[sub_idx]
        y_ref = y[sub_idx]
    else:
        x_ref = x
        y_ref = y
    
    # Compute denoised y for each point
    y_denoised = np.zeros(n)
    for i in range(n):
        if time.time() - t_start > deadline - 40:
            y_denoised[i] = y[i]
            continue
        dist = np.abs(x[i] - x_ref)
        w = np.exp(-0.5 * (dist / bandwidth) ** 2)
        w_sum = np.sum(w)
        if w_sum > 1e-12:
            y_denoised[i] = np.sum(w * y_ref) / w_sum
        else:
            y_denoised[i] = y[i]
    
    # Step 2: Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_norm = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for standardized x
    V = np.vander(x_norm, N=6, increasing=True)
    
    # Step 3: K-fold CV for lambda selection
    k_folds = 5
    np.random.seed(42)
    indices = np.random.permutation(n)
    fold_size = n // k_folds
    
    # Candidate lambdas: log-spaced from 1e-10 to 1e3
    log_lams = np.linspace(np.log10(1e-10), np.log10(1e3), 30)
    lam_candidates = 10 ** log_lams
    
    best_val_mse = np.inf
    best_lam = 1e-4
    
    for i in range(k_folds):
        if time.time() - t_start > deadline - 20:
            break
        
        val_start = i * fold_size
        val_end = val_start + fold_size
        if i == k_folds - 1:
            val_end = n
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        x_train = x_norm[train_idx]
        y_train = y_denoised[train_idx]
        x_val = x_norm[val_idx]
        y_val = y_denoised[val_idx]
        
        V_train = np.vander(x_train, N=6, increasing=True)
        V_val = np.vander(x_val, N=6, increasing=True)
        
        for lam in lam_candidates:
            if time.time() - t_start > deadline - 15:
                break
            
            A = V_train.T @ V_train + lam * np.eye(6)
            b = V_train.T @ y_train
            
            try:
                coeff = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            
            if not np.all(np.isfinite(coeff)):
                continue
            
            pred_val = V_val @ coeff
            mse = np.mean((pred_val - y_val) ** 2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_lam = lam
    
    # Step 4: Refit on all data with best lambda
    A = V.T @ V + best_lam * np.eye(6)
    b = V.T @ y_denoised
    
    try:
        coeff_norm = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeff_norm = np.linalg.lstsq(A, b, rcond=None)[0]
    
    if not np.all(np.isfinite(coeff_norm)):
        coeff_norm = np.zeros(6)
    
    # Step 5: Convert from normalized x to original x scale
    coeff_orig = np.zeros(6, dtype=np.float64)
    for i in range(6):
        a_i = coeff_norm[i]
        if abs(a_i) < 1e-15:
            continue
        a_i_scaled = a_i / (x_std ** i)
        for j in range(i + 1):
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)
            term = binom_val * ((-x_mean) ** (i - j)) * a_i_scaled
            coeff_orig[j] += term
    
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)
    
    return coeff_orig
