"""
Strategy: Global Non-Linear Basis Expansion with Regularized Least Squares.
Instead of piecewise local splines, we fit a single global model using a linear combination
of basis functions from multiple families (Fourier, RBF, Chebyshev). 
We search over key hyperparameters (frequency for Fourier, bandwidth for RBF, degree for Chebyshev)
and select the best model via cross-validation. This approach is robust to smooth non-polynomial 
functions and avoids the boundary discontinuities inherent in piecewise binning strategies.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import lstsq

def _safe_solve(A, b, reg):
    """Solve (A^T A + reg*I) w = A^T b with ridge regularization."""
    try:
        AtA = A.T @ A + reg * np.eye(A.shape[1])
        Atb = A.T @ b
        w, residuals, rank, s = lstsq(AtA, Atb, cond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None

def _predict_basis(X, basis_type, params):
    """Generate basis features for given X and parameters."""
    X = X.flatten()
    n = len(X)
    if n == 0:
        return np.empty((0, 1))
    
    if basis_type == 'fourier':
        # Basis: 1, cos(k*x), sin(k*x) for k=1..K
        K = params['K']
        # Scale x to [0, 2*pi] roughly? Or use raw? 
        # Since domain is known per equation, let's use raw x but scale by frequency.
        # Actually, better to normalize x to [-1, 1] or [0, 1] for stable Fourier?
        # Let's stick to raw x but choose K based on range.
        # However, if x is in [-3,3], high K might be needed.
        # Let's normalize x to [-1, 1] to make frequency selection easier.
        x_min, x_max = np.min(X), np.max(X)
        if x_max > x_min:
            x_norm = (X - x_min) / (x_max - x_min) * 2 - 1 # Map to [-1, 1]
        else:
            x_norm = np.zeros_like(X)
        
        cols = [np.ones(n)]
        for k in range(1, K + 1):
            cols.append(np.cos(k * np.pi * x_norm))
            cols.append(np.sin(k * np.pi * x_norm))
        return np.column_stack(cols)
        
    elif basis_type == 'rbf':
        # Basis: 1, exp(-gamma * (x - center_i)^2)
        # Centers: evenly spaced quantiles or just linspace
        # We will use the training points themselves as centers for flexibility? 
        # No, that's N=200 features. Might be slow but okay.
        # Better: Use a subset of centers to keep it fast. Let's use linspace centers.
        # Or, even better: Use the training points as centers but regularize heavily.
        # Let's try fixed number of centers M.
        M = params['M']
        x_min, x_max = np.min(X), np.max(X)
        centers = np.linspace(x_min, x_max, M)
        gamma = params['gamma']
        
        # Compute distances
        # X shape (n,), centers shape (M,)
        # dist shape (n, M)
        dist = np.abs(X[:, np.newaxis] - centers[np.newaxis, :])
        cols = [np.ones(n)]
        for i in range(M):
            cols.append(np.exp(-gamma * dist[:, i] ** 2))
        return np.column_stack(cols)
        
    elif basis_type == 'chebyshev':
        # Basis: Chebyshev polynomials T_k(x_norm)
        K = params['K']
        x_min, x_max = np.min(X), np.max(X)
        if x_max > x_min:
            x_norm = (X - x_min) / (x_max - x_min) * 2 - 1
        else:
            x_norm = np.zeros_like(X)
            
        cols = [np.ones(n)]
        if K >= 1:
            cols.append(x_norm)
        for k in range(2, K + 1):
            # Recurrence: T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
            cols.append(2 * x_norm * cols[-1] - cols[-2])
        return np.column_stack(cols)
    
    else:
        return np.ones((n, 1))

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer
    
    X = X_train[:, 0]
    y = y_train
    n = len(X)
    
    if n == 0:
        return lambda X: np.zeros(X.shape[0])
        
    # Standardize y for numerical stability in loss? No, we just want to predict y.
    # But normalizing features helps.
    
    best_cv_mse = np.inf
    best_predict = None
    
    # Define search space
    # Fourier: K (number of harmonics)
    fourier_Ks = [5, 10, 20, 40, 80]
    # RBF: M (number of centers), gamma (width)
    rbf_Ms = [10, 20, 50]
    rbf_gammas = [0.5, 1.0, 5.0, 10.0, 50.0]
    # Chebyshev: K (degree)
    cheb_Ks = [5, 10, 20, 40, 80]
    
    # Regularization strengths
    regs = [1e-6, 1e-4, 1e-2, 1e-1]
    
    # Cross-validation setup
    # 5-fold CV
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42) # Fixed seed for CV splits to be deterministic
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    # Precompute some things?
    # We will iterate through configurations.
    
    configs = []
    for K in fourier_Ks:
        configs.append(('fourier', {'K': K}))
    for M in rbf_Ms:
        for g in rbf_gammas:
            configs.append(('rbf', {'M': M, 'gamma': g}))
    for K in cheb_Ks:
        configs.append(('chebyshev', {'K': K}))
        
    # To save time, we don't test all regs for all configs. 
    # We'll pick a few good regs.
    # Actually, let's just try one reg per config first, then refine?
    # Or just use a fixed reg like 1e-4 for speed, and maybe 1e-2 for high K.
    
    # Let's structure the search:
    # 1. Coarse search over basis types and main params with moderate reg.
    # 2. Fine search on best few.
    
    # For simplicity and robustness within time limit:
    # We will evaluate each config with a fixed reg=1e-4 first.
    # If time allows, we might try others.
    
    # Optimization: Precompute normalized X for Fourier/Cheb?
    # Since X is 1D, it's fast.
    
    for basis_type, params in configs:
        if time.time() > deadline - 10:
            break
            
        try:
            # Compute CV error
            cv_errors = []
            for i in range(n_folds):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                X_tr = X[tr_idx]
                y_tr = y[tr_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                # Generate features
                A_tr = _predict_basis(X_tr, basis_type, params)
                A_val = _predict_basis(X_val, basis_type, params)
                
                # Solve
                reg = 1e-4 # Default
                w = _safe_solve(A_tr, y_tr, reg)
                if w is None:
                    cv_errors.append(np.inf)
                    continue
                    
                pred_val = A_val @ w
                if not np.all(np.isfinite(pred_val)):
                    cv_errors.append(np.inf)
                    continue
                
                mse = np.mean((pred_val - y_val) ** 2)
                cv_errors.append(mse)
            
            mean_mse = np.mean(cv_errors)
            
            if mean_mse < best_cv_mse:
                best_cv_mse = mean_mse
                # Fit on full data
                A_full = _predict_basis(X, basis_type, params)
                w_full = _safe_solve(A_full, y, 1e-4)
                if w_full is not None and np.all(np.isfinite(w_full)):
                    # Store best model
                    best_predict = (basis_type, params, w_full)
                    
        except Exception:
            continue
            
    # If we have time, refine the best model's regularization?
    # Or try a second round of configs?
    # Given 300s limit, and ~30 configs * 5 folds * (matrix mult 200xN), it should be fast.
    # Let's check time.
    if time.time() < deadline - 20 and best_predict is not None:
        # Try to refine regularization for the best basis type/params
        b_type, b_params, _ = best_predict
        best_reg_mse = best_cv_mse
        best_reg_w = None
        best_reg = 1e-4
        
        for reg in regs:
            if time.time() > deadline - 10:
                break
            cv_errors = []
            for i in range(n_folds):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                X_tr = X[tr_idx]
                y_tr = y[tr_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                A_tr = _predict_basis(X_tr, b_type, b_params)
                A_val = _predict_basis(X_val, b_type, b_params)
                
                w = _safe_solve(A_tr, y_tr, reg)
                if w is None:
                    cv_errors.append(np.inf)
                    continue
                    
                pred_val = A_val @ w
                if not np.all(np.isfinite(pred_val)):
                    cv_errors.append(np.inf)
                    continue
                
                mse = np.mean((pred_val - y_val) ** 2)
                cv_errors.append(mse)
                
            mean_mse = np.mean(cv_errors)
            if mean_mse < best_reg_mse:
                best_reg_mse = mean_mse
                best_reg = reg
                # Fit full with this reg
                A_full = _predict_basis(X, b_type, b_params)
                w_full = _safe_solve(A_full, y, reg)
                if w_full is not None and np.all(np.isfinite(w_full)):
                    best_reg_w = w_full
                    
        if best_reg_w is not None:
            best_predict = (b_type, b_params, best_reg_w)
            
    if best_predict is None:
        # Fallback: Global Linear Regression (Polynomial degree 1)
        # Or just mean
        # Let's try a simple polynomial degree 5
        try:
            x_min, x_max = X.min(), X.max()
            if x_max > x_min:
                x_norm = (X - x_min) / (x_max - x_min) * 2 - 1
            else:
                x_norm = np.zeros_like(X)
            
            A = np.column_stack([np.ones(n), x_norm, x_norm**2, x_norm**3, x_norm**4, x_norm**5])
            w = _safe_solve(A, y, 1e-4)
            if w is not None:
                def predict(X, x_min=x_min, x_max=x_max, w=w):
                    X_flat = X.flatten()
                    if x_max > x_min:
                        x_norm = (X_flat - x_min) / (x_max - x_min) * 2 - 1
                    else:
                        x_norm = np.zeros_like(X_flat)
                    A_pred = np.column_stack([np.ones(len(X_flat)), x_norm, x_norm**2, x_norm**3, x_norm**4, x_norm**5])
                    return (A_pred @ w).reshape(X.shape)
                return predict
        except Exception:
            pass
            
        # Absolute fallback
        mean_y = np.mean(y)
        return lambda X: np.full(X.shape[0], mean_y)
        
    # Create final predict function
    basis_type, params, w = best_predict
    
    def predict(X):
        A = _predict_basis(X, basis_type, params)
        pred = A @ w
        return pred.reshape(X.shape)
        
    return predict
# EVOLVE-BLOCK-END
