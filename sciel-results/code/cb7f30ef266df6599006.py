"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Handle empty or mismatched inputs
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove NaNs and Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
        
    if n < 10:
        # Not enough data for robust selection, just do a simple fit
        # Use least squares with degree min(5, n-1)
        deg = min(5, n-1)
        coeff = np.zeros(6)
        if deg >= 0:
            try:
                c = np.polynomial.polynomial.polyfit(x, y, deg)
                coeff[:len(c)] = c
            except:
                pass
        return coeff

    # Build Vandermonde matrix for degree 5
    # V[i, j] = x[i]^j
    V = np.vander(x, N=6, increasing=True)
    
    # If we have enough data, use an internal split to select regularization parameter
    # This helps prevent overfitting to noise in the training trajectories
    if n > 100:
        # Shuffle indices for splitting
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        split = int(0.8 * n)
        train_idx = indices[:split]
        valid_idx = indices[split:]
        
        V_train = V[train_idx]
        y_train = y[train_idx]
        V_valid = V[valid_idx]
        y_valid = y[valid_idx]
        
        # Precompute V^T V and V^T y for train set
        VtV = V_train.T @ V_train
        Vty = V_train.T @ y_train
        
        # Candidate regularization parameters (log-spaced)
        lambdas = np.logspace(-10, 2, 25)
        
        best_lambda = 0.0
        best_mse = np.inf
        
        for lam in lambdas:
            # Add lambda * I to diagonal
            # We can use a small identity matrix
            reg_matrix = VtV + lam * np.eye(6)
            
            try:
                # Solve for coefficients
                c = np.linalg.solve(reg_matrix, Vty)
                
                # Evaluate on validation set
                pred = V_valid @ c
                mse = np.mean((y_valid - pred) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_lambda = lam
                    
            except np.linalg.LinAlgError:
                continue
            
            # Check time
            if time.time() > deadline - 10:
                break
                
        # Refit on full dataset with best lambda
        VtV_full = V.T @ V
        Vty_full = V.T @ y
        reg_matrix_full = VtV_full + best_lambda * np.eye(6)
        
        try:
            coeff = np.linalg.solve(reg_matrix_full, Vty_full)
        except np.linalg.LinAlgError:
            # Fallback to OLS
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            except:
                coeff = np.zeros(6)
                
    else:
        # Small dataset, just do OLS
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except:
            coeff = np.zeros(6)
            
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        
    return coeff
# EVOLVE-BLOCK-END
