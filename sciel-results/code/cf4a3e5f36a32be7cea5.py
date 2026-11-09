"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
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
    
    # Normalize x to [-1, 1] to improve condition number of Vandermonde matrix
    x_min, x_max = np.min(x), np.max(x)
    if x_max > x_min:
        x_norm = (x - (x_max + x_min) / 2.0) / ((x_max - x_min) / 2.0)
    else:
        x_norm = np.zeros_like(x)
    
    # Build Vandermonde matrix for degree 5 in ascending powers
    # V[:, i] = x_norm^i
    V = np.vander(x_norm, 6, increasing=True)
    
    # Regularization matrix R
    # We penalize higher degree terms more heavily to control variance
    # R = diag([1, lambda, lambda^2, ...]) or similar
    # Let's use a diagonal matrix where R_ii = (i+1)^2 * lambda
    # Or simply R_ii = lambda^i for i>0, R_00=0
    # A common choice is Tikhonov with R = diag(0, 1, 1, ..., 1) * lambda
    # But we want to strongly penalize high degrees.
    # Let's try R_ii = (i+1)^2 * lambda
    
    # Cross-validation to find best lambda
    # Split data into 5 folds
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.seed(42) # Fixed seed for reproducibility in CV
    np.random.shuffle(indices)
    
    best_lambda = 0.0
    best_cv_mse = np.inf
    best_coeffs = np.zeros(6)
    
    # Log-spaced lambda range
    # lambda = 0 is OLS. Large lambda forces coefficients to 0.
    # We need to scale lambda relative to the data variance or matrix norms.
    # Let's estimate a scale factor.
    # ||V||_F^2 is roughly n * sum(x^2i). 
    # Let's just try a wide range.
    # If x is normalized to [-1,1], eigenvalues of V^T V are bounded.
    # Let's try lambdas from 1e-8 to 1e8.
    
    lambda_candidates = np.logspace(-8, 8, 25)
    
    # Precompute V^T V and V^T y for speed? 
    # Since we do CV, we need to recompute for each fold.
    # n is likely large (tens of thousands). 25 lambdas * 5 folds = 125 fits.
    # Each fit is 6x6 solve. Very fast.
    
    for lam in lambda_candidates:
        cv_errors = []
        for k in range(n_folds):
            if time.time() - t_start > deadline - 10.0:
                # Time out, break out of loops
                break
            
            # Define fold indices
            start = k * fold_size
            end = start + fold_size if k < n_folds - 1 else n
            
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            x_train = x_norm[train_idx]
            y_train = y[train_idx]
            x_val = x_norm[val_idx]
            y_val = y[val_idx]
            
            V_train = np.vander(x_train, 6, increasing=True)
            V_val = np.vander(x_val, 6, increasing=True)
            
            # Regularization matrix
            # R_ii = (i+1)^2 * lam
            r_vec = np.array([(i+1)**2 for i in range(6)], dtype=np.float64) * lam
            # Don't penalize the constant term as much? Or maybe we should.
            # Usually we don't penalize the mean. Let's set r_vec[0] = 0.
            r_vec[0] = 0.0
            R = np.diag(r_vec)
            
            # Solve (V^T V + R) c = V^T y
            A = V_train.T @ V_train + R
            b = V_train.T @ y_train
            
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            # Predict on validation set
            y_pred = V_val @ c
            mse = np.mean((y_val - y_pred)**2)
            cv_errors.append(mse)
        
        if len(cv_errors) == n_folds:
            mean_cv_mse = np.mean(cv_errors)
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_lambda = lam
                # Refit on all data with best lambda found so far
                # We can do this at the end, or here. Doing it at the end is more efficient.
                # But we need to store the best lambda.
        
        if time.time() - t_start > deadline - 10.0:
            break
    
    # Refit on all data with best_lambda
    R = np.diag(np.array([(i+1)**2 for i in range(6)], dtype=np.float64) * best_lambda)
    R[0,0] = 0.0
    
    A = V.T @ V + R
    b = V.T @ y
    
    try:
        c_norm = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to OLS if regularized solve fails (unlikely)
        c_norm = np.linalg.lstsq(V, y, rcond=None)[0]
    
    # Transform coefficients from normalized x to original x
    # We fit p(x_norm) = sum c_i x_norm^i
    # x_norm = (x - m) / s
    # x_norm^i = ((x-m)/s)^i = (1/s^i) * (x-m)^i
    # We need to expand (x-m)^i in powers of x.
    # (x-m)^i = sum_{k=0}^i binom(i, k) x^k (-m)^{i-k}
    
    m = (x_max + x_min) / 2.0
    s = (x_max - x_min) / 2.0
    
    if s == 0:
        # Degenerate case, just return mean
        c_orig = np.zeros(6)
        c_orig[0] = np.mean(y)
        return c_orig
    
    c_orig = np.zeros(6)
    
    for i in range(6):
        coeff_i = c_norm[i] / (s ** i)
        # Expand (x-m)^i
        # sum_{k=0}^i binom(i, k) x^k (-m)^{i-k}
        for k in range(i + 1):
            # binom(i, k)
            binom_val = 1.0
            for j in range(k):
                binom_val *= (i - j) / (j + 1)
            
            term = binom_val * ((-m) ** (i - k))
            c_orig[k] += coeff_i * term
    
    # Ensure finite
    if not np.all(np.isfinite(c_orig)):
        # Fallback to simple linear fit on raw data
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            out = np.zeros(6)
            out[:2] = coeff
            return out
        except Exception:
            return np.zeros(6, dtype=np.float64)
            
    return c_orig
# EVOLVE-BLOCK-END
