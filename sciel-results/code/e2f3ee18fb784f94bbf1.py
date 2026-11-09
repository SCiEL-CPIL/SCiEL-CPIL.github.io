"""Symbolic regression via Sparse Gaussian Process with composite kernel."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _rbf_kernel(X1, X2, ls):
    """
    RBF kernel: exp(-0.5 * ||x1 - x2||^2 / ls^2)
    ls can be scalar or array of shape (d,)
    """
    if np.isscalar(ls):
        ls = np.full(X1.shape[1], ls)
    X1_s = X1 / ls
    X2_s = X2 / ls
    sq_dists = np.sum(X1_s**2, axis=1)[:, None] + np.sum(X2_s**2, axis=1)[None, :] - 2 * (X1_s @ X2_s.T)
    np.maximum(sq_dists, 0, out=sq_dists)
    return np.exp(-0.5 * sq_dists)

def _poly_kernel(X1, X2, degree, gamma, coef0):
    """
    Polynomial kernel: (gamma * <x1, x2> + coef0)^degree
    """
    return (gamma * (X1 @ X2.T) + coef0) ** degree

def _compute_composite_kernel(X1, X2, params):
    """
    Compute the composite kernel matrix.
    params: dict with keys 'rbf_ls', 'poly_w', 'poly_deg', 'poly_gamma', 'poly_c0'
    """
    d = X1.shape[1]
    ls = params['rbf_ls']
    K = _rbf_kernel(X1, X2, ls)
    
    if params['poly_w'] > 1e-8:
        K = K + params['poly_w'] * _poly_kernel(X1, X2, params['poly_deg'], params['poly_gamma'], params['poly_c0'])
        
    return K

def _kmeans(X, k, rng, max_iter=50):
    """Simple k-means clustering."""
    n, d = X.shape
    if k >= n:
        return X.copy()
    # Initialize centers using k-means++ style
    centers = np.empty((k, d))
    centers[0] = X[rng.randint(n)]
    for i in range(1, k):
        d2 = np.min([np.sum((X - centers[j])**2, axis=1) for j in range(i)], axis=0)
        total = d2.sum()
        if total < 1e-12:
            centers[i] = X[rng.randint(n)]
        else:
            probs = d2 / total
            idx = rng.choice(n, p=probs)
            centers[i] = X[idx]
    
    for _ in range(max_iter):
        # Assign points to nearest center
        d2 = np.sum(X**2, axis=1)[:, None] + np.sum(centers**2, axis=1)[None, :] - 2 * X @ centers.T
        assignments = np.argmin(d2, axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]
        
        if np.allclose(centers, new_centers, atol=1e-8):
            centers = new_centers
            break
        centers = new_centers
    
    return centers

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 290.0  # 10s margin before 300s limit
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-8
    y_centered = y_train - y_mean
    
    # Normalize inputs to [0, 1] for stable kernel scaling
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    X_norm = (X_train - x_min) / x_range
    
    # Precompute 5-fold CV indices
    n_folds = 5
    indices = np.arange(n)
    # Shuffle indices for random CV folds
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = [indices[i::n_folds] for i in range(n_folds)]
    
    # Initial guess
    # Typical length scales for normalized data [0,1] are around 0.1 to 1.0
    init_ls = 0.5 * np.ones(d)
    init_alpha = 1e-2
    init_poly_w = 0.1
    
    x0 = np.concatenate([
        np.log(init_ls), 
        [np.log(init_alpha)], 
        [np.log(init_poly_w)]
    ])
    
    bounds = [(-2.0, 2.0)] * d + [(-10.0, 2.0), (-10.0, 2.0)]
    
    def objective(x):
        if time.time() > deadline - 5:
            return np.inf
        ls = np.exp(x[:d])
        alpha = np.exp(x[d])
        poly_w = np.exp(x[d+1])
        
        # Clamp values to reasonable ranges to prevent numerical issues
        ls = np.clip(ls, 1e-3, 10.0)
        alpha = np.clip(alpha, 1e-6, 10.0)
        poly_w = np.clip(poly_w, 0.0, 10.0)
        
        params = {
            'rbf_ls': ls,
            'poly_w': poly_w,
            'poly_deg': 2,
            'poly_gamma': 1.0,
            'poly_c0': 1.0
        }
        
        # Compute full kernel matrix for training data
        K_full = _compute_composite_kernel(X_norm, X_norm, params)
        
        cv_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            K_train = K_full[np.ix_(train_idx, train_idx)]
            y_train_fold = y_centered[train_idx]
            y_val_fold = y_centered[val_idx]
            
            # Use sparse GP approximation for CV
            # Select inducing points as a subset of training points
            m = min(50, len(train_idx))
            # Use k-means to select inducing points from training data
            rng_ind = np.random.RandomState(0)
            Z = _kmeans(X_norm[train_idx], m, rng_ind)
            
            K_ii = _compute_composite_kernel(Z, Z, params) + alpha * np.eye(m)
            K_zi = _compute_composite_kernel(Z, X_norm[train_idx], params)
            K_zz = _compute_composite_kernel(Z, X_norm[val_idx], params)
            
            # Solve for weights: w = K_ii^{-1} K_zi y_train
            try:
                L = np.linalg.cholesky(K_ii)
                K_ii_inv_y = np.linalg.solve(L.T, np.linalg.solve(L, K_zi.T @ y_train_fold))
                pred_val = K_zz.T @ K_ii_inv_y
            except np.linalg.LinAlgError:
                # Fallback to full kernel ridge if sparse fails
                coeffs = np.linalg.lstsq(K_train + alpha * np.eye(len(train_idx)), y_train_fold, rcond=None)[0]
                K_val = K_full[np.ix_(val_idx, train_idx)]
                pred_val = K_val @ coeffs
                
            mse = np.mean((pred_val - y_val_fold) ** 2)
            cv_mses.append(mse)
            
        return np.mean(cv_mses)

    # Optimize hyperparameters
    best_x = x0
    best_score = np.inf
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass

    # If optimization failed or timed out, try a few random restarts or simpler grid if needed
    # But L-BFGS-B is usually robust. Let's add a simple refinement if we have time.
    if time.time() < deadline - 10:
        # Try Nelder-Mead for local refinement if L-BFGS-B didn't converge perfectly
        try:
            res2 = minimize(objective, best_x, method='Nelder-Mead', bounds=bounds,
                            options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            if res2.fun < best_score:
                best_score = res2.fun
                best_x = res2.x
        except Exception:
            pass

    # Extract best parameters
    ls = np.exp(best_x[:d])
    alpha = np.exp(best_x[d])
    poly_w = np.exp(best_x[d+1])
    
    ls = np.clip(ls, 1e-3, 10.0)
    alpha = np.clip(alpha, 1e-6, 10.0)
    poly_w = np.clip(poly_w, 0.0, 10.0)
    
    final_params = {
        'rbf_ls': ls,
        'poly_w': poly_w,
        'poly_deg': 2,
        'poly_gamma': 1.0,
        'poly_c0': 1.0
    }
    
    # Final fit using sparse GP approximation
    # Select inducing points from training data
    m = min(50, n)
    rng_final = np.random.RandomState(123)
    Z = _kmeans(X_norm, m, rng_final)
    
    K_ii = _compute_composite_kernel(Z, Z, final_params) + alpha * np.eye(m)
    K_zx = _compute_composite_kernel(Z, X_norm, final_params)
    
    # Solve for weights
    try:
        L = np.linalg.cholesky(K_ii)
        # w = K_ii^{-1} K_zx y
        K_zx_y = K_zx.T @ y_centered
        w = np.linalg.solve(L.T, np.linalg.solve(L, K_zx_y))
    except np.linalg.LinAlgError:
        # Fallback to full kernel ridge if sparse fails
        K_final = _compute_composite_kernel(X_norm, X_norm, final_params)
        A = K_final + alpha * np.eye(n)
        try:
            w = np.linalg.solve(A, y_centered)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, y_centered, rcond=None)[0]
        Z = X_norm  # Use all points as inducing points in fallback
        m = n
    
    def predict(X):
        # Normalize inputs
        X_test_norm = (X - x_min) / x_range
        K_tz = _compute_composite_kernel(X_test_norm, Z, final_params)
        pred = K_tz @ w + y_mean
        # Ensure finite values
        pred = np.nan_to_num(pred, nan=y_mean, posinf=y_mean + 1e6, neginf=y_mean - 1e6)
        return pred

    return predict
# EVOLVE-BLOCK-END
