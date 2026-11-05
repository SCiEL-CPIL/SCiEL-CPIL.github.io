import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Strategy:
    1. Normalize x to improve conditioning.
    2. Fit polynomial in normalized space using Ridge Regression.
    3. Use K-Fold Cross-Validation to select the best regularization parameter lambda.
    4. Transform the optimal normalized coefficients back to the original x scale 
       using binomial expansion.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin from 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any NaN/inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        return np.zeros(6)
    
    # 1. Normalize x
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # 2. Generate design matrix for normalized x
    def make_design_matrix(x_vals):
        """Create Vandermonde-like design matrix for poly basis."""
        x_vals = np.asarray(x_vals)
        V = np.ones((len(x_vals), 6))
        for i in range(1, 6):
            V[:, i] = V[:, i-1] * x_vals
        return V
        
    V_norm = make_design_matrix(x_norm)
    
    # 3. Cross-Validation for Ridge Regression lambda selection
    # We want to find lambda that minimizes validation error.
    # Since we don't have external validation, we use K-Fold CV on training data.
    
    n_samples = len(x)
    k_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42) # Fixed seed for reproducibility in CV splits
    np.random.shuffle(indices)
    
    fold_size = n_samples // k_folds
    if fold_size < 1:
        fold_size = 1
        k_folds = n_samples
        
    # List of lambdas to test (Ridge alpha)
    # Note: In Ridge, lambda is added to diagonal of V^T V. 
    # Smaller lambda -> less regularization.
    lambdas = np.logspace(-6, 4, 20)
    
    best_lambda = 0.0
    best_cv_score = np.inf
    best_coef_norm = None
    
    # Pre-compute V^T V and V^T y for efficiency? 
    # Actually, for each fold, the training set is different, so we must recompute.
    # But we can just do the solve directly.
    
    for lam in lambdas:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        cv_errors = []
        for i in range(k_folds):
            # Define validation and training indices for this fold
            val_start = i * fold_size
            val_end = val_start + fold_size if i < k_folds - 1 else n_samples
            
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            if len(train_idx) < 10:
                continue
                
            X_train = V_norm[train_idx]
            y_train = y[train_idx]
            X_val = V_norm[val_idx]
            y_val = y[val_idx]
            
            # Ridge Regression Solve
            # Minimize ||X c - y||^2 + lambda ||c||^2
            # Solution: c = (X^T X + lambda I)^-1 X^T y
            try:
                A = X_train.T @ X_train + lam * np.eye(6)
                B = X_train.T @ y_train
                c_train = np.linalg.solve(A, B)
                
                # Evaluate on validation set
                y_pred_val = X_val @ c_train
                mse_val = np.mean((y_pred_val - y_val) ** 2)
                cv_errors.append(mse_val)
            except np.linalg.LinAlgError:
                continue
                
        if cv_errors:
            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_cv_score:
                best_cv_score = mean_cv_error
                best_lambda = lam
                # Store the coefficients from the full training set with this lambda?
                # No, we need to refit on the full data with the best lambda later.
                # But we can store the "current best" logic.
    
    # 4. Refit on full training data with best lambda
    if time.time() - start_time > DEADLINE - 5:
        # Fallback to OLS if we ran out of time
        best_lambda = 0.0
        
    try:
        A_full = V_norm.T @ V_norm + best_lambda * np.eye(6)
        B_full = V_norm.T @ y
        coef_norm = np.linalg.solve(A_full, B_full)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        coef_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
        
    if not np.all(np.isfinite(coef_norm)):
        return np.zeros(6)
        
    # 5. Transform normalized coefficients to original x scale
    # We have: y = sum_{i=0}^5 c_norm[i] * ( (x - x_mean)/x_std )^i
    # Let z = (x - x_mean)/x_std = a*x + b
    # where a = 1/x_std, b = -x_mean/x_std
    # (a*x + b)^i = sum_{k=0}^i C(i,k) * (a*x)^k * b^(i-k)
    #              = sum_{k=0}^i C(i,k) * a^k * b^(i-k) * x^k
    # So, coeff_orig[k] = sum_{i=k}^5 c_norm[i] * C(i,k) * a^k * b^(i-k)
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    coeff_orig = np.zeros(6)
    for i in range(6):
        ci = coef_norm[i]
        for k in range(i + 1):
            coeff_orig[k] += ci * comb(i, k) * (a ** k) * (b ** (i - k))
            
    # 6. Final sanity check
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback: simple mean
        coeff_orig = np.zeros(6)
        coeff_orig[0] = np.mean(y)
        
    return coeff_orig
