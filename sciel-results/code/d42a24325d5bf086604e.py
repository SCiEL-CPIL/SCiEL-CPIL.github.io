"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # If data is too small or time is tight, fall back to simple OLS
    if n < 20 or time.time() > deadline - 5.0:
        try:
            # Use degree 5 OLS if enough points, else lower
            deg = 5 if n > 10 else min(5, n-1)
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            # Pad to 6 if necessary
            if len(coeff) < 6:
                coeff = np.pad(coeff, (0, 6-len(coeff)))
            return coeff
        except Exception:
            return np.zeros(6)

    # 1. Standardize x for numerical stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    x_scaled = (x - mu) / sigma

    # 2. Create Vandermonde matrix for scaled x (degree 5)
    # V_scaled[i, j] = x_scaled[i]^j
    V_scaled = np.vander(x_scaled, 6, increasing=True)
    
    # 3. Model Selection via Hold-out Validation
    # Shuffle indices
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Split 80% train, 20% val
    split_idx = int(0.8 * n)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    if len(val_idx) < 10:
        # If validation set is too small, use all data for fit with a fixed reasonable lambda
        # Or just use OLS
        try:
            coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
        except Exception:
            coeff_scaled = np.zeros(6)
    else:
        V_train = V_scaled[train_idx]
        y_train = y[train_idx]
        V_val = V_scaled[val_idx]
        y_val = y[val_idx]
        
        # Precompute V_train^T V_train and V_train^T y_train
        VtV = V_train.T @ V_train
        Vty = V_train.T @ y_train
        
        # Log-spaced lambda values
        # Regularization strength relative to data scale
        lambdas = np.logspace(-6, 6, 50)
        
        best_lambda = 0.0
        best_mse = np.inf
        best_coeff_scaled = None
        
        for lam in lambdas:
            # Ridge Regression: (V^T V + lam * I) c = V^T y
            # We regularize the coefficients in the scaled basis
            A = VtV + lam * np.eye(6)
            try:
                c = np.linalg.solve(A, Vty)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A, Vty, rcond=None)[0]
            
            # Evaluate on validation set
            y_pred_val = V_val @ c
            mse = np.mean((y_val - y_pred_val) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_lambda = lam
                best_coeff_scaled = c
                
        if best_coeff_scaled is None:
            best_coeff_scaled = np.zeros(6)
            
        # Refit on all data with best lambda
        VtV_all = V_scaled.T @ V_scaled
        Vty_all = V_scaled.T @ y
        A_all = VtV_all + best_lambda * np.eye(6)
        try:
            coeff_scaled = np.linalg.solve(A_all, Vty_all)
        except np.linalg.LinAlgError:
            coeff_scaled = np.linalg.lstsq(A_all, Vty_all, rcond=None)[0]

    # 4. Transform coefficients from scaled basis x' to original basis x
    # x' = (x - mu) / sigma  =>  x = sigma * x' + mu
    # We have P(x') = sum_{k=0}^5 c'_k (x')^k
    # We want Q(x) = sum_{j=0}^5 d_j x^j such that Q(x) = P((x-mu)/sigma)
    
    d = np.zeros(6)
    for k in range(6):
        c_k = coeff_scaled[k]
        if abs(c_k) < 1e-15:
            continue
        # Term is c_k * ((x - mu) / sigma)^k
        # = c_k * sigma^{-k} * (x - mu)^k
        # Expand (x - mu)^k = sum_{j=0}^k binom(k, j) x^j (-mu)^(k-j)
        factor = c_k / (sigma ** k)
        for j in range(k + 1):
            # Binomial coefficient C(k, j)
            binom_k_j = float(math.comb(k, j))
            term_coeff = factor * binom_k_j * ((-mu) ** (k - j))
            d[j] += term_coeff
            
    return d
# EVOLVE-BLOCK-END
