"""Symbolic regression via SINDy (Sparse Identification of Nonlinear Dynamics)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize

def _build_library(X):
    """
    Build a library of candidate nonlinear basis functions.
    X: (n, d) array of inputs.
    Returns: (n, m) array of basis function values.
    """
    n, d = X.shape
    # Standardize X to zero mean, unit variance for stability of basis functions
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_std = X_std + 1e-8 # Avoid division by zero
    X_norm = (X - X_mean) / X_std
    
    # Store normalization params for prediction
    norm_params = (X_mean, X_std)
    
    basis = [np.ones((n, 1))]
    
    # Linear terms
    for j in range(d):
        basis.append(X_norm[:, j:j+1])
        
    # Quadratic terms (x_i * x_j, x_i^2)
    for i in range(d):
        for j in range(i, d):
            basis.append((X_norm[:, i] * X_norm[:, j]).reshape(-1, 1))
            
    # Cubic terms (x_i^3, x_i^2*x_j, x_i*x_j^2, x_i*x_j*x_k) - keep sparse
    for i in range(d):
        basis.append((X_norm[:, i]**3).reshape(-1, 1))
        for j in range(d):
            if i != j:
                basis.append((X_norm[:, i]**2 * X_norm[:, j]).reshape(-1, 1))
                basis.append((X_norm[:, i] * X_norm[:, j]**2).reshape(-1, 1))
                
    # Trigonometric terms (sin(x), cos(x))
    for j in range(d):
        basis.append(np.sin(X_norm[:, j]).reshape(-1, 1))
        basis.append(np.cos(X_norm[:, j]).reshape(-1, 1))
        
    # Exponential terms (exp(x) - 1) to avoid overflow, scaled
    for j in range(d):
        # Clip input to avoid overflow in exp
        X_clip = np.clip(X_norm[:, j], -10, 10)
        basis.append((np.exp(X_clip) - 1).reshape(-1, 1))
        
    # Abs terms
    for j in range(d):
        basis.append(np.abs(X_norm[:, j]).reshape(-1, 1))
        
    # Interaction terms with abs/sin/cos (limited)
    if d >= 2:
        # sin(x_i) * cos(x_j) etc might be too many, stick to simple products of linear/nonlinear
        # Let's add a few cross terms
        for i in range(d):
            for j in range(d):
                if i != j:
                    basis.append((np.sin(X_norm[:, i]) * X_norm[:, j]).reshape(-1, 1))
                    basis.append((np.cos(X_norm[:, i]) * X_norm[:, j]).reshape(-1, 1))
                    
    library = np.hstack(basis)
    
    # Normalize columns of the library to unit norm for stable sparse regression
    # This is crucial for SINDy
    norms = np.linalg.norm(library, axis=0)
    norms[norms < 1e-8] = 1.0
    library_normed = library / norms
    
    return library_normed, norm_params, norms

def _sindy_solve(Phi, y, lam, max_iter=100):
    """
    Sparse regression using iterative soft-thresholding (LARS-like or ISTA).
    Phi: (n, m) design matrix (normalized columns)
    y: (n,) target
    lam: regularization parameter
    Returns: coefficients (m,)
    """
    n, m = Phi.shape
    # Precompute Gram matrix G = Phi.T @ Phi
    # For large n, this is expensive, but n=200 is small.
    G = Phi.T @ Phi
    g = Phi.T @ y
    
    theta = np.zeros(m)
    active = np.zeros(m, dtype=bool)
    
    # ISTA / FISTA style update
    step_size = 1.0 / (np.max(np.abs(np.diag(G))) + 1e-8)
    
    for _ in range(max_iter):
        # Gradient step
        theta = theta + step_size * (g - G @ theta)
        
        # Soft thresholding
        theta = np.sign(theta) * np.maximum(np.abs(theta) - lam, 0)
        
        # Check convergence
        # Optional: check change in theta
        
    return theta

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280.0 # Safety margin
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-8:
        y_var = 1e-8
    y_centered = y_train - y_mean
    
    # Build library
    Phi, norm_params, norms = _build_library(X_train)
    
    # Optimize regularization parameter lam using CV
    # We want to minimize MSE on test set, but we use CV MSE as proxy.
    # SINDy score is often better with slightly higher lam to avoid overfitting noise,
    # but here noise is 0. So we want lam low enough to capture structure.
    
    n_folds = 5
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = [indices[i::n_folds] for i in range(n_folds)]
    
    # Log-spaced lambda grid
    lam_grid = np.logspace(-6, 2, 20)
    
    def objective_lam(log_lam):
        if time.time() > deadline - 5:
            return np.inf
        lam = np.exp(log_lam)
        
        cv_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            Phi_train = Phi[train_idx, :]
            y_train_fold = y_centered[train_idx]
            y_val_fold = y_centered[val_idx]
            
            # Solve sparse regression
            theta = _sindy_solve(Phi_train, y_train_fold, lam, max_iter=50)
            
            # Predict
            pred_val = Phi[val_idx, :] @ theta
            mse = np.mean((pred_val - y_val_fold) ** 2)
            cv_mses.append(mse)
            
        return np.mean(cv_mses)
    
    best_log_lam = np.log(1e-4)
    best_score = np.inf
    
    # Coarse grid search
    for ll in np.log(lam_grid):
        score = objective_lam(ll)
        if score < best_score:
            best_score = score
            best_log_lam = ll
            
    # Refine with local optimization if time permits
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_lam, best_log_lam, method='Nelder-Mead', 
                           options={'maxiter': 20, 'xatol': 1e-3, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_log_lam = res.x[0]
        except Exception:
            pass
            
    best_lam = np.exp(best_log_lam)
    
    # Final fit on all data
    theta_final = _sindy_solve(Phi, y_centered, best_lam, max_iter=100)
    
    # Store necessary parameters for prediction
    X_mean, X_std = norm_params
    norms = norms
    
    def predict(X):
        # Normalize inputs
        X_norm = (X - X_mean) / X_std
        
        # Rebuild library for test points
        # To avoid code duplication, we could store the library builder logic, 
        # but it's easier to just recompute. n_test is small (200).
        n_test, d_test = X_norm.shape
        basis = [np.ones((n_test, 1))]
        
        for j in range(d_test):
            basis.append(X_norm[:, j:j+1])
            
        for i in range(d_test):
            for j in range(i, d_test):
                basis.append((X_norm[:, i] * X_norm[:, j]).reshape(-1, 1))
                
        for i in range(d_test):
            basis.append((X_norm[:, i]**3).reshape(-1, 1))
            for j in range(d_test):
                if i != j:
                    basis.append((X_norm[:, i]**2 * X_norm[:, j]).reshape(-1, 1))
                    basis.append((X_norm[:, i] * X_norm[:, j]**2).reshape(-1, 1))
                    
        for j in range(d_test):
            basis.append(np.sin(X_norm[:, j]).reshape(-1, 1))
            basis.append(np.cos(X_norm[:, j]).reshape(-1, 1))
            
        for j in range(d_test):
            X_clip = np.clip(X_norm[:, j], -10, 10)
            basis.append((np.exp(X_clip) - 1).reshape(-1, 1))
            
        for j in range(d_test):
            basis.append(np.abs(X_norm[:, j]).reshape(-1, 1))
            
        if d_test >= 2:
            for i in range(d_test):
                for j in range(d_test):
                    if i != j:
                        basis.append((np.sin(X_norm[:, i]) * X_norm[:, j]).reshape(-1, 1))
                        basis.append((np.cos(X_norm[:, i]) * X_norm[:, j]).reshape(-1, 1))
                        
        Phi_test = np.hstack(basis)
        
        # Normalize columns of test library using training norms
        # This ensures the basis functions are on the same scale
        Phi_test_normed = Phi_test / norms
        
        pred = Phi_test_normed @ theta_final + y_mean
        pred = np.nan_to_num(pred, nan=y_mean, posinf=y_mean + 1e6, neginf=y_mean - 1e6)
        return pred

    return predict
# EVOLVE-BLOCK-END
