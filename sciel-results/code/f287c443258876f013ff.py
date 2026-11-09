"""Symbolic regression via SINDy-style sparse regression with rich basis library and ensemble fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
import itertools

def _build_sindy_library(X, max_poly_deg=4, max_trig_freq=5, max_exp_scale=3):
    """
    Build a SINDy-style library of candidate basis functions.
    Includes: constant, linear, polynomial terms (up to max_poly_deg),
    trig terms (sin/cos of k*x_i), and exponential terms (exp(-s*x_i)).
    """
    n, d = X.shape
    library = [np.ones((n, 1))]
    
    # Linear terms
    for i in range(d):
        library.append(X[:, i:i+1])
    
    # Polynomial terms (up to max_poly_deg)
    for deg in range(2, max_poly_deg + 1):
        # Generate all monomials of degree 'deg' in 'd' variables
        for indices in itertools.combinations_with_replacement(range(d), deg):
            term = np.ones((n, 1))
            for idx in indices:
                term = term * X[:, idx:idx+1]
            library.append(term)
    
    # Trigonometric terms: sin(k * pi * x_i) and cos(k * pi * x_i)
    for i in range(d):
        for k in range(1, max_trig_freq + 1):
            arg = k * np.pi * X[:, i:i+1]
            library.append(np.sin(arg))
            library.append(np.cos(arg))
    
    # Exponential terms: exp(-s * x_i)
    for i in range(d):
        for s in range(1, max_exp_scale + 1):
            library.append(np.exp(-s * X[:, i:i+1]))
    
    # Interaction terms between different variables (degree 2-3)
    if d >= 2:
        for deg in range(2, min(4, max_poly_deg + 1)):
            for indices in itertools.combinations(range(d), deg):
                term = np.ones((n, 1))
                for idx in indices:
                    term = term * X[:, idx:idx+1]
                library.append(term)
    
    # Normalize columns to unit norm for better conditioning
    Phi = np.hstack(library)
    norms = np.linalg.norm(Phi, axis=0, keepdims=True)
    norms[norms < 1e-12] = 1.0
    Phi = Phi / norms
    
    return Phi

def _sparse_regression_lars(Phi, y, n_nonzero_max=10, deadline=None):
    """
    Perform sparse regression using LARS (Least Angle Regression) or Lasso.
    We use a simple iterative thresholding approach for speed.
    Returns the sparse coefficient vector.
    """
    n, p = Phi.shape
    alpha = np.zeros(p)
    
    # Use a simple soft-thresholding iterative method (IST)
    # Or use scipy's lasso via lsqr with regularization
    # For speed, we'll use a few iterations of coordinate descent or just regularized least squares
    
    # Actually, let's use a simple approach: try different sparsity levels
    # and pick the one with best validation score (handled externally)
    
    # For now, return dense solution as fallback, but we want sparse
    # Let's use a simple greedy forward selection
    residual = y.copy()
    selected = []
    alpha = np.zeros(p)
    
    # Precompute Gram matrix for speed (if p is not too large)
    if p <= 500:
        G = Phi.T @ Phi
        Gy = Phi.T @ y
    else:
        G = None
        Gy = None
    
    for step in range(n_nonzero_max):
        if G is not None:
            # Compute correlations
            if len(selected) == 0:
                corr = Gy
            else:
                # residual correlation
                corr = Gy - G[:, selected] @ alpha[selected]
        else:
            corr = Phi.T @ residual
        
        # Find variable with max absolute correlation
        max_idx = np.argmax(np.abs(corr))
        if max_idx in selected:
            continue
        
        selected.append(max_idx)
        sel = np.array(selected)
        
        # Solve least squares on selected variables
        if len(sel) == 1:
            alpha_sel = Gy[sel] / (G[sel, sel] + 1e-10) if G is not None else (Phi[:, sel].T @ y) / (Phi[:, sel].T @ Phi[:, sel] + 1e-10)
        else:
            if G is not None:
                G_sel = G[np.ix_(sel, sel)]
                Gy_sel = Gy[sel]
                try:
                    alpha_sel = np.linalg.solve(G_sel + 1e-10 * np.eye(len(sel)), Gy_sel)
                except:
                    alpha_sel = np.linalg.lstsq(G_sel, Gy_sel, rcond=None)[0]
            else:
                try:
                    alpha_sel = np.linalg.lstsq(Phi[:, sel], y, rcond=None)[0]
                except:
                    alpha_sel = np.zeros(len(sel))
        
        alpha[sel] = alpha_sel
        residual = y - Phi[:, sel] @ alpha_sel
        
        # Check if improvement is significant
        if step > 0:
            # Simple stopping criterion
            if np.max(np.abs(alpha_sel)) < 1e-6:
                break
    
    return alpha, selected

def fit(X_train, y_train):
    """Fit a SINDy-style sparse regression model with ensemble fallback."""
    start_time = time.time()
    deadline = start_time + 290.0  # 40s safety margin from 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Build SINDy library
    Phi = _build_sindy_library(X_norm, max_poly_deg=3, max_trig_freq=4, max_exp_scale=2)
    p = Phi.shape[1]
    
    # Validation split
    rng = np.random.RandomState(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    half = n // 2
    val_idx = idx[:half]
    train_idx = idx[half:]
    
    best_score = np.inf
    best_alpha = None
    best_selected = None
    best_model_type = 'sindy'
    
    # Try sparse regression with different sparsity levels
    for n_nonzero in [3, 5, 8, 12]:
        if time.time() > deadline - 30:
            break
        
        alpha, selected = _sparse_regression_lars(Phi[train_idx], y[train_idx], n_nonzero_max=n_nonzero)
        
        # Evaluate on validation
        pred_val = Phi[val_idx, selected] @ alpha[selected]
        mse_val = np.mean((pred_val - y[val_idx]) ** 2)
        
        if mse_val < best_score:
            best_score = mse_val
            best_alpha = alpha
            best_selected = selected
    
    # If SINDy didn't find a good model, try kernel ridge regression as fallback
    if best_score > np.var(y) * 0.5:
        # RBF kernel fallback
        diff = X_norm[:, None, :] - X_norm[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        
        l2_candidates = [1.0, 2.0, 4.0, 8.0, 16.0]
        alpha_candidates = [1e-3, 1e-2, 0.1]
        
        for l2 in l2_candidates:
            if time.time() > deadline - 20:
                break
            K = np.exp(-l2 * sq_dist)
            for alpha in alpha_candidates:
                K_train = K[np.ix_(train_idx, train_idx)]
                A = K_train + alpha * np.eye(len(train_idx))
                try:
                    w = np.linalg.solve(A, y[train_idx])
                    K_val = K[np.ix_(val_idx, train_idx)]
                    pred_val = K_val @ w
                    mse_val = np.mean((pred_val - y[val_idx]) ** 2)
                    
                    if mse_val < best_score:
                        best_score = mse_val
                        best_l2 = l2
                        best_alpha_kr = alpha
                        best_w = w
                        best_model_type = 'kr'
                except:
                    continue
    
    # Final fit on all data
    if best_model_type == 'sindy' and best_selected is not None:
        # Refit on all data
        alpha_full, selected_full = _sparse_regression_lars(Phi, y, n_nonzero_max=max(len(best_selected), 5))
        final_alpha = alpha_full
        final_selected = selected_full
        final_Phi = Phi
    else:
        # RBF fallback final fit
        if 'best_l2' not in locals():
            best_l2 = 4.0
            best_alpha_kr = 1e-2
        K_final = np.exp(-best_l2 * sq_dist)
        A_final = K_final + best_alpha_kr * np.eye(n)
        try:
            w_final = np.linalg.solve(A_final, y)
        except:
            w_final = np.linalg.lstsq(A_final, y, rcond=None)[0]
        final_l2 = best_l2
        final_w = w_final
        final_X_norm = X_norm
    
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_norm = (X_new - X_min) / X_range
        
        if best_model_type == 'sindy':
            Phi_new = _build_sindy_library(X_new_norm, max_poly_deg=3, max_trig_freq=4, max_exp_scale=2)
            return Phi_new[:, final_selected] @ final_alpha[final_selected]
        else:
            diff_new = X_new_norm[:, None, :] - final_X_norm[None, :, :]
            sq_dist_new = np.sum(diff_new ** 2, axis=2)
            K_new = np.exp(-final_l2 * sq_dist_new)
            return K_new @ final_w
    
    return predict
# EVOLVE-BLOCK-END
