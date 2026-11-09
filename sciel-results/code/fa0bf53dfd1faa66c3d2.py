# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    
    Strategy:
    1. Subsample data if large to enable Gaussian Process Regression (GPR).
    2. Use GPR with RBF kernel to denoise, selecting length-scale via CV.
    3. Fit a degree-5 polynomial to the GPR mean estimates.
    4. Return coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
        
    # Clean data
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n == 0:
        return np.zeros(6)
        
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.zeros(6)
        
    # Subsample for GPR (O(N^3) complexity)
    max_gpr_size = 1000
    if n > max_gpr_size:
        indices = np.random.choice(n, max_gpr_size, replace=False)
        x_gpr = x[indices]
        y_gpr = y[indices]
    else:
        x_gpr = x
        y_gpr = y
        
    n_gpr = len(x_gpr)
    
    # Hyperparameter search for length-scale l
    # Candidates relative to data range
    l_candidates = [0.05 * x_range, 0.1 * x_range, 0.2 * x_range, 0.5 * x_range]
    
    # Noise variance estimate
    sigma2 = 0.01 * np.var(y_gpr) + 1e-12
    
    # Split for CV
    cv_size = int(0.2 * n_gpr)
    if cv_size < 10: cv_size = 10
    if n_gpr <= cv_size + 10:
        # Too small for meaningful CV, just pick middle l
        best_l = 0.1 * x_range
    else:
        # Random split
        perm = np.random.permutation(n_gpr)
        val_idx = perm[:cv_size]
        train_idx = perm[cv_size:]
        
        x_train = x_gpr[train_idx]
        y_train = y_gpr[train_idx]
        x_val = x_gpr[val_idx]
        y_val = y_gpr[val_idx]
        
        best_mse = np.inf
        best_l = 0.1 * x_range
        
        for l in l_candidates:
            if time.time() > deadline - 20:
                break
            try:
                # Build K_train
                diff_t = x_train[:, np.newaxis] - x_train[np.newaxis, :]
                K_t = np.exp(-0.5 * (diff_t ** 2) / (l ** 2))
                
                # Solve
                alpha_t = np.linalg.solve(K_t + sigma2 * np.eye(len(x_train)), y_train)
                
                # Predict val
                diff_v = x_val[:, np.newaxis] - x_train[np.newaxis, :]
                K_v = np.exp(-0.5 * (diff_v ** 2) / (l ** 2))
                y_pred = K_v @ alpha_t
                
                mse = np.mean((y_val - y_pred) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_l = l
            except:
                continue
                
    # Final fit with best_l on all subsampled data
    try:
        diff_f = x_gpr[:, np.newaxis] - x_gpr[np.newaxis, :]
        K_f = np.exp(-0.5 * (diff_f ** 2) / (best_l ** 2))
        alpha_f = np.linalg.solve(K_f + sigma2 * np.eye(n_gpr), y_gpr)
    except:
        # Fallback to simple kernel smoothing
        return _fallback_kernel_smooth(x, y)
        
    # Evaluation points
    # Use a dense grid within the range of the training data
    n_eval = 300
    x_eval = np.linspace(x_gpr.min(), x_gpr.max(), n_eval)
    
    # Predict y_eval
    diff_e = x_eval[:, np.newaxis] - x_gpr[np.newaxis, :]
    K_e = np.exp(-0.5 * (diff_e ** 2) / (best_l ** 2))
    y_eval = K_e @ alpha_f
    
    # Fit degree-5 polynomial
    try:
        coeff = np.polynomial.polynomial.polyfit(x_eval, y_eval, 5)
    except:
        return _fallback_kernel_smooth(x, y)
        
    if not np.all(np.isfinite(coeff)):
        return _fallback_kernel_smooth(x, y)
        
    return coeff


def _fallback_kernel_smooth(x, y):
    """Fallback to Gaussian kernel smoothing if GPR fails."""
    n = len(x)
    if n == 0:
        return np.zeros(6)
        
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        return np.zeros(6)
        
    n_eval = 200
    x_eval = np.linspace(x_min, x_max, n_eval)
    y_smoothed = np.zeros(n_eval)
    bandwidth = 0.1 * (x_max - x_min)
    
    for i in range(n_eval):
        dist = np.abs(x_eval[i] - x)
        weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
        w_sum = np.sum(weights)
        if w_sum > 1e-12:
            y_smoothed[i] = np.sum(weights * y) / w_sum
        else:
            y_smoothed[i] = np.mean(y)
            
    try:
        return np.polynomial.polynomial.polyfit(x_eval, y_smoothed, 5)
    except:
        return np.zeros(6)
# EVOLVE-BLOCK-END
