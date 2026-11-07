"""Symbolic regression via Sparse Gaussian Process (LASSO-GP) with adaptive hyperparameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import operator
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

def fit(X_train, y_train):
    start_time = time.time()
    # 300s limit. We reserve 30s for safety/overhead, leaving 270s for computation.
    deadline = start_time + 270.0
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1] for numerical stability and kernel scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Standardize y for numerical stability
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Strategy: Sparse GP (LASSO-GP)
    # We model y as a linear combination of RBF kernels centered at a subset of training points.
    # To find this subset, we solve a Lasso problem (L1 regularization) on the full kernel matrix.
    # This induces sparsity, keeping only the most relevant basis functions.
    # This is structurally different from KRR (L2) as it performs feature selection.
    
    def rbf_kernel(X1, X2, length_scale, sigma):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return sigma * np.exp(-sq_dists / (2 * length_scale**2))

    # Hyperparameter search
    # length_scale controls the width of the RBF
    # alpha (L1 penalty) controls the sparsity
    length_scales = [0.05, 0.1, 0.2, 0.5, 1.0]
    # L1 penalties: smaller means more sparse. 
    # Note: Standard Lasso uses lambda * ||w||_1. 
    # We use scipy.optimize.minimize with L1? No, scipy doesn't have native Lasso.
    # We can use scipy.sparse.linalg.lsqr with regularization? No, lsqr is L2.
    # We need a Lasso solver. 
    # Since SciPy is available, we can use `scipy.optimize.minimize` with method='SLSQP' or similar, 
    # but that's slow for 200 variables.
    # Alternative: Use a simple coordinate descent or subgradient descent for Lasso? 
    # Or, we can use `scipy.optimize.linprog`? No.
    # Let's use a manual implementation of a simple Lasso solver (e.g., ISTA or FISTA) 
    # or rely on the fact that for N=200, even a dense solver with L1 via optimization is feasible 
    # if we limit the search space.
    
    # Actually, a better approach for "Sparse GP" in Python with standard libs:
    # Use `scipy.optimize.minimize` on the objective: 0.5 * ||K w - y||^2 + lambda * ||w||_1.
    # This is non-smooth. 
    # We can use `scipy.optimize.minimize` with method='Nelder-Mead'? No, not for 200 dims.
    # Let's implement a simple FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for Lasso.
    # It's fast and robust.
    
    def solve_lasso(K, y, lam, max_iter=500, tol=1e-4):
        """
        Solve min 0.5 * ||K w - y||^2 + lam * ||w||_1
        Using FISTA.
        K: (n, n) kernel matrix
        y: (n,) target
        lam: L1 penalty
        """
        n = K.shape[0]
        # Lipschitz constant L = max eigenvalue of K.T @ K
        # Approximate L for speed
        # For RBF kernels, eigenvalues are bounded. 
        # Let's estimate L using power iteration on a few samples or just use a safe upper bound.
        # A safe upper bound for L is ||K||_F^2 / n roughly? Or just compute max eig of K.T K if n is small.
        # n=200, K.T K is 200x200. Computing eigvals is fast (~10ms).
        KtK = K.T @ K
        eigvals = np.linalg.eigvalsh(KtK)
        L = max(eigvals[-1], 1e-6)
        step_size = 1.0 / L
        
        # Precompute K.T @ y
        Kt_y = K.T @ y
        
        w = np.zeros(n)
        w_prev = np.zeros(n)
        t = 1.0
        
        for i in range(max_iter):
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            w_new = w_prev + (t - 1) / t_new * (w_prev - w)
            
            # Gradient step
            grad = K.T @ (K @ w_new - y)
            w_temp = w_new - step_size * grad
            
            # Shrinkage (Soft Thresholding)
            threshold = lam * step_size
            w = np.sign(w_temp) * np.maximum(np.abs(w_temp) - threshold, 0)
            
            # Check convergence
            if np.linalg.norm(w - w_prev) < tol * (np.linalg.norm(w) + 1e-8):
                break
            
            w_prev = w
            t = t_new
            
        return w

    best_w = None
    best_train_mse = np.inf
    best_ls = None
    
    # Search over length scales and L1 penalties
    # L1 penalties: We want to try a range. 
    # If lam is too high, w becomes 0. If too low, it's like Ridge.
    # Let's sample logs from 1e-4 to 1e-1.
    lams = np.logspace(-4, -1, 10)
    
    for ls in length_scales:
        if time.time() > deadline - 5:
            break
        
        K = rbf_kernel(X_norm, X_norm, ls, 1.0)
        
        for lam in lams:
            if time.time() > deadline - 5:
                break
            
            try:
                w = solve_lasso(K, y_norm, lam)
                
                # Check if solution is trivial (all zeros)
                if np.allclose(w, 0):
                    continue
                
                # Calculate training MSE
                pred_train = K @ w
                mse = np.mean((pred_train - y_norm)**2)
                
                if mse < best_train_mse:
                    best_train_mse = mse
                    best_w = w
                    best_ls = ls
                    
            except Exception:
                continue

    if best_w is None:
        # Fallback: Simple linear regression or constant
        def predict(X_test):
            return np.full(X_test.shape[0], y_mean)
        return predict

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        
        # Compute K_test_train
        K_tt = rbf_kernel(X_test_norm, X_norm, best_ls, 1.0)
        
        # Prediction
        pred_norm = K_tt @ best_w
        pred = pred_norm * y_std + y_mean
        
        return pred

    return predict
# EVOLVE-BLOCK-END
