import numpy as np
from scipy.linalg import lstsq # or use numpy.linalg

def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Handle edge cases
    if len(x) == 0:
        return np.zeros(6)
    
    # Build Vandermonde matrix for degree 5
    # V[:, i] = x^i
    V = np.vander(x, N=6, increasing=True)
    
    # Split for lambda selection
    n = len(x)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    V_train = V[train_idx]
    y_train = y[train_idx]
    V_val = V[val_idx]
    y_val = y[val_idx]
    
    best_lambda = 0.0
    best_mse = np.inf
    
    # Regularization strengths to try
    # The scale of the problem: y is tendency, x is state.
    # Typical L96 values are O(10).
    # Ridge penalty: lambda * ||c||^2.
    # If x is large, high order coeffs are small.
    # Let's try a wide range.
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6]
    
    for lam in lambdas:
        # Solve (V^T V + lam * I) c = V^T y
        # Using lstsq for stability
        V_reg = V_train.copy()
        y_reg = y_train.copy()
        
        # Efficient ridge:
        # (V^T V + lam I) c = V^T y
        # We can just use np.linalg.lstsq on an augmented system or direct solve.
        # Direct solve is fine for 6x6 system.
        try:
            A = V_train.T @ V_train + lam * np.eye(6)
            b = V_train.T @ y_train
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
            
        # Calculate validation MSE
        pred_val = V_val @ c
        mse = np.mean((pred_val - y_val)**2)
        
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
            
    # Refit on full data with best lambda
    if best_lambda == 0.0:
        # Ordinary Least Squares
        try:
            coeff, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        except:
            coeff = np.zeros(6)
    else:
        try:
            A = V.T @ V + best_lambda * np.eye(6)
            b = V.T @ y
            coeff = np.linalg.solve(A, b)
        except:
            coeff = np.zeros(6)
            
    # Ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        
    return coeff
