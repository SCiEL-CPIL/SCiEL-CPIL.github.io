"""Symbolic regression via SINDy sparse regression with kernel fallback and validation-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement

def _build_sindy_library(X, max_poly_degree=4, max_trig_freq=2):
    """Build SINDy library of candidate basis functions."""
    n, d = X.shape
    library = [np.ones((n, 1))]
    
    # Polynomial terms
    for degree in range(1, max_poly_degree + 1):
        for combo in combinations_with_replacement(range(d), degree):
            feat = np.ones(n)
            for idx in combo:
                feat *= X[:, idx]
            library.append(feat[:, np.newaxis])
    
    # Trig terms (sin/cos of each dimension, and products up to degree 2)
    for freq in range(1, max_trig_freq + 1):
        for dim in range(d):
            library.append(np.sin(np.pi * freq * X[:, dim])[:, np.newaxis])
            library.append(np.cos(np.pi * freq * X[:, dim])[:, np.newaxis])
    
    # Products of trig terms (degree 2)
    if d >= 2:
        for i in range(d):
            for j in range(i, d):
                library.append((np.sin(np.pi * X[:, i]) * np.sin(np.pi * X[:, j]))[:, np.newaxis])
                library.append((np.sin(np.pi * X[:, i]) * np.cos(np.pi * X[:, j]))[:, np.newaxis])
                library.append((np.cos(np.pi * X[:, i]) * np.cos(np.pi * X[:, j]))[:, np.newaxis])
    
    # Exponential terms
    for dim in range(d):
        library.append(np.exp(X[:, dim])[:, np.newaxis])
        library.append(np.exp(-X[:, dim])[:, np.newaxis])
        # sqrt terms
        library.append(np.sqrt(np.maximum(X[:, dim], 0))[:, np.newaxis])
        # log terms
        library.append(np.log(np.maximum(X[:, dim], 1e-8))[:, np.newaxis])
    
    return np.hstack(library)

def _sparse_regression(Phi, y, alpha, max_iter=50):
    """
    Perform sparse regression via iterative LARS-like approach.
    Uses forward selection with ridge regression at each step.
    """
    n, p = Phi.shape
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    # Start with just the bias
    selected = [0]  # index of ones column
    best_mse = np.sum(y_centered**2) / n
    coefficients = np.zeros(p)
    
    for iteration in range(max_iter):
        # Compute residuals
        if len(selected) == 1:
            residuals = y_centered
        else:
            Phi_sel = Phi[:, selected]
            # Ridge regression on selected features
            A = Phi_sel.T @ Phi_sel + alpha * np.eye(len(selected))
            b = Phi_sel.T @ y_centered
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(A, b, rcond=None)
            pred = Phi_sel @ w
            residuals = y_centered - pred
            current_mse = np.sum(residuals**2) / n
            
            # Check if adding more helps
            if current_mse < best_mse * 0.99:
                best_mse = current_mse
            else:
                break
        
        # Find the feature that best correlates with residuals
        correlations = np.abs(Phi.T @ residuals) / (np.sqrt(np.sum(Phi**2, axis=0)) + 1e-12)
        
        # Exclude already selected
        correlations[selected] = 0
        
        best_feat = np.argmax(correlations)
        
        if correlations[best_feat] < 1e-6:
            break
        
        selected.append(best_feat)
    
    # Final fit on selected features
    Phi_sel = Phi[:, selected]
    A = Phi_sel.T @ Phi_sel + alpha * np.eye(len(selected))
    b = Phi_sel.T @ y_centered
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    
    # Build full coefficient vector
    full_coeffs = np.zeros(p)
    full_coeffs[selected] = w
    
    # Add back the mean
    return full_coeffs, y_mean

def _rbf_kernel(X1, X2, length_scales):
    """Compute RBF kernel with per-dimension length scales."""
    X1s = X1 / length_scales
    X2s = X2 / length_scales
    sq1 = np.sum(X1s**2, axis=1)[:, None]
    sq2 = np.sum(X2s**2, axis=1)[None, :]
    sq_dists = sq1 + sq2 - 2.0 * (X1s @ X2s.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.exp(-0.5 * sq_dists)

def _poly_kernel(X1, X2, degree, gamma, coef0):
    """Compute polynomial kernel."""
    return (gamma * (X1 @ X2.T) + coef0) ** degree

def _periodic_kernel(X1, X2, period, length_scale):
    """Compute periodic kernel."""
    diff = X1[:, None, :] - X2[None, :, :]
    sin_diff = np.sin(np.pi * diff / period)
    sq_sum = np.sum(sin_diff**2, axis=2)
    return np.exp(-2.0 * sq_sum / (length_scale**2))

def _compute_composite_kernel(X1, X2, params):
    """Compute composite kernel: RBF + polynomial + periodic."""
    K = np.zeros((X1.shape[0], X2.shape[0]))
    
    if params.get('rbf_weight', 0) > 1e-12:
        K += params['rbf_weight'] * _rbf_kernel(X1, X2, params['rbf_ls'])
    
    if params.get('poly_weight', 0) > 1e-12:
        K += params['poly_weight'] * _poly_kernel(X1, X2, params['poly_degree'], 
                                                   params['poly_gamma'], params['poly_coef0'])
    
    if params.get('per_weight', 0) > 1e-12:
        K += params['per_weight'] * _periodic_kernel(X1, X2, params['per_period'], params['per_ls'])
    
    return K

def _kmeans(X, n_clusters, max_iters=50, n_init=5, seed=42):
    """Simple k-means."""
    n_samples, n_features = X.shape
    n_clusters = min(n_clusters, n_samples)
    rng = np.random.RandomState(seed)
    best_inertia = float('inf')
    best_centroids = None
    
    for init in range(n_init):
        idx = rng.choice(n_samples, n_clusters, replace=False)
        centroids = X[idx].copy()
        
        for _ in range(max_iters):
            dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        inertia = np.sum(dists[np.arange(n_samples), np.argmin(dists, axis=1)]**2)
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
    
    return best_centroids

def fit(X_train, y_train):
    """
    Hybrid symbolic regression: SINDy sparse regression + kernel ridge fallback.
    Uses validation to select between SINDy and kernel approaches.
    """
    start_time = time.time()
    deadline = start_time + 295  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1]
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-12] = 1.0
    X_norm = (X - min_vals) / range_vals
    
    # Validation split
    rng = np.random.RandomState(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    val_size = int(0.2 * n_samples)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    X_tr = X_norm[tr_idx]
    y_tr = y[tr_idx]
    X_val = X_norm[val_idx]
    y_val = y[val_idx]
    
    best_val_mse = float('inf')
    best_method = None
    best_params = None
    
    # ============================================================
    # METHOD 1: SINDy sparse regression
    # ============================================================
    sindy_configs = [
        (3, 1, 1e-4),
        (4, 1, 1e-4),
        (4, 2, 1e-4),
        (5, 2, 1e-3),
        (3, 2, 1e-5),
    ]
    
    for max_poly, max_trig, alpha in sindy_configs:
        if time.time() > deadline - 30:
            break
        
        try:
            # Build library on training set
            Phi_tr = _build_sindy_library(X_tr, max_poly_degree=max_poly, max_trig_freq=max_trig)
            Phi_val = _build_sindy_library(X_val, max_poly_degree=max_poly, max_trig_freq=max_trig)
            
            # Check for numerical issues
            if not np.all(np.isfinite(Phi_tr)) or not np.all(np.isfinite(Phi_val)):
                continue
            
            # Sparse regression
            coeffs, y_mean = _sparse_regression(Phi_tr, y_tr, alpha)
            
            # Evaluate on validation
            y_val_pred = Phi_val @ coeffs + y_mean
            mse_val = np.mean((y_val_pred - y_val)**2)
            
            if mse_val < best_val_mse:
                best_val_mse = mse_val
                best_method = 'sindy'
                best_params = {'max_poly': max_poly, 'max_trig': max_trig, 'alpha': alpha}
                
        except Exception:
            continue
    
    # ============================================================
    # METHOD 2: Kernel ridge regression (composite kernel)
    # ============================================================
    rbf_ls_grid = [0.1, 0.2, 0.3, 0.5, 1.0]
    alpha_grid = [1e-5, 1e-4, 1e-3, 1e-2]
    
    for ls_val in rbf_ls_grid:
        if time.time() > deadline - 30:
            break
        for alpha in alpha_grid:
            if time.time() > deadline - 30:
                break
            
            try:
                params = {
                    'rbf_weight': 1.0, 'rbf_ls': np.full(n_features, ls_val),
                    'poly_weight': 0.0, 'poly_degree': 2, 'poly_gamma': 1.0, 'poly_coef0': 1.0,
                    'per_weight': 0.0, 'per_period': 2.0, 'per_ls': 1.0
                }
                
                K_tr = _compute_composite_kernel(X_tr, X_tr, params)
                K_val_tr = _compute_composite_kernel(X_val, X_tr, params)
                
                # Ridge regression
                A = K_tr + alpha * np.eye(len(y_tr))
                b = y_tr
                try:
                    coeffs = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
                
                y_val_pred = K_val_tr @ coeffs
                mse_val = np.mean((y_val_pred - y_val)**2)
                
                if mse_val < best_val_mse:
                    best_val_mse = mse_val
                    best_method = 'kernel_rbf'
                    best_params = {'ls': ls_val, 'alpha': alpha}
                    
            except Exception:
                continue
    
    # ============================================================
    # METHOD 3: RBF with k-means centers
    # ============================================================
    n_centers_list = [10, 20, 30]
    gamma_list = [2.0, 5.0, 10.0, 20.0]
    
    for n_centers in n_centers_list:
        if time.time() > deadline - 30:
            break
        for gamma in gamma_list:
            if time.time() > deadline - 30:
                break
            
            try:
                centers_tr = _kmeans(X_tr, n_centers, n_init=3, seed=42)
                
                # Compute RBF features
                def _rbf_feats(Xq, centers, gamma):
                    X_sq = np.sum(Xq**2, axis=1, keepdims=True)
                    C_sq = np.sum(centers**2, axis=1, keepdims=True)
                    sq_dists = X_sq + C_sq.T - 2 * (Xq @ centers.T)
                    sq_dists = np.maximum(sq_dists, 0)
                    return np.exp(-gamma * sq_dists)
                
                Phi_tr_rbf = _rbf_feats(X_tr, centers_tr, gamma)
                Phi_val_rbf = _rbf_feats(X_val, centers_tr, gamma)
                
                # Add polynomial features
                def _poly_feats(Xq, deg=2):
                    n, d = Xq.shape
                    feats = [np.ones((n, 1))]
                    for deg_i in range(1, deg + 1):
                        for combo in combinations_with_replacement(range(d), deg_i):
                            feat = np.ones(n)
                            for idx in combo:
                                feat *= Xq[:, idx]
                            feats.append(feat[:, np.newaxis])
                    return np.hstack(feats)
                
                Phi_tr_poly = _poly_feats(X_tr, 2)
                Phi_val_poly = _poly_feats(X_val, 2)
                
                Phi_tr_full = np.hstack([Phi_tr_rbf, Phi_tr_poly])
                Phi_val_full = np.hstack([Phi_val_rbf, Phi_val_poly])
                
                n_feat = Phi_tr_full.shape[1]
                A = Phi_tr_full.T @ Phi_tr_full + 1e-4 * np.eye(n_feat)
                b = Phi_tr_full.T @ y_tr
                try:
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w, *_ = np.linalg.lstsq(A, b, rcond=None)
                
                y_val_pred = Phi_val_full @ w
                mse_val = np.mean((y_val_pred - y_val)**2)
                
                if mse_val < best_val_mse:
                    best_val_mse = mse_val
                    best_method = 'rbf_kmeans'
                    best_params = {'n_centers': n_centers, 'gamma': gamma, 'centers': centers_tr}
                    
            except Exception:
                continue
    
    # ============================================================
    # Refit on all data with best method
    # ============================================================
    if best_method is None:
        # Fallback: simple linear regression
        best_method = 'linear'
        best_params = {}
    
    if best_method == 'sindy':
        max_poly = best_params['max_poly']
        max_trig = best_params['max_trig']
        alpha = best_params['alpha']
        
        Phi_full = _build_sindy_library(X_norm, max_poly_degree=max_poly, max_trig_freq=max_trig)
        if not np.all(np.isfinite(Phi_full)):
            # Fallback to linear
            Phi_full = np.hstack([X_norm, np.ones((n_samples, 1))])
            y_mean = 0
            A = Phi_full.T @ Phi_full + alpha * np.eye(Phi_full.shape[1])
            b = Phi_full.T @ y
            try:
                w_full = np.linalg.solve(A, b)
            except:
                w_full, *_ = np.linalg.lstsq(A, b, rcond=None)
            final_type = 'linear_fallback'
            final_w = w_full
            final_mean = 0
        else:
            w_full, y_mean = _sparse_regression(Phi_full, y, alpha)
            final_type = 'sindy'
            final_w = w_full
            final_mean = y_mean
    
    elif best_method == 'kernel_rbf':
        ls = best_params['ls']
        alpha = best_params['alpha']
        
        params = {
            'rbf_weight': 1.0, 'rbf_ls': np.full(n_features, ls),
            'poly_weight': 0.0, 'poly_degree': 2, 'poly_gamma': 1.0, 'poly_coef0': 1.0,
            'per_weight': 0.0, 'per_period': 2.0, 'per_ls': 1.0
        }
        
        K_full = _compute_composite_kernel(X_norm, X_norm, params)
        A = K_full + alpha * np.eye(n_samples)
        try:
            coeffs_full = np.linalg.solve(A, y)
        except:
            coeffs_full, *_ = np.linalg.lstsq(A, y, rcond=None)
        
        final_type = 'kernel_rbf'
        final_coeffs = coeffs_full
        final_params = params
    
    elif best_method == 'rbf_kmeans':
        n_centers = best_params['n_centers']
        gamma = best_params['gamma']
        
        centers_full = _kmeans(X_norm, n_centers, n_init=10, seed=42)
        
        def _rbf_feats(Xq, centers, gamma):
            X_sq = np.sum(Xq**2, axis=1, keepdims=True)
            C_sq = np.sum(centers**2, axis=1, keepdims=True)
            sq_dists = X_sq + C_sq.T - 2 * (Xq @ centers.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-gamma * sq_dists)
        
        def _poly_feats(Xq, deg=2):
            n, d = Xq.shape
            feats = [np.ones((n, 1))]
            for deg_i in range(1, deg + 1):
                for combo in combinations_with_replacement(range(d), deg_i):
                    feat = np.ones(n)
                    for idx in combo:
                        feat *= Xq[:, idx]
                    feats.append(feat[:, np.newaxis])
            return np.hstack(feats)
        
        Phi_full_rbf = _rbf_feats(X_norm, centers_full, gamma)
        Phi_full_poly = _poly_feats(X_norm, 2)
        Phi_full = np.hstack([Phi_full_rbf, Phi_full_poly])
        
        n_feat = Phi_full.shape[1]
        A = Phi_full.T @ Phi_full + 1e-4 * np.eye(n_feat)
        b = Phi_full.T @ y
        try:
            w_full = np.linalg.solve(A, b)
        except:
            w_full, *_ = np.linalg.lstsq(A, b, rcond=None)
        
        final_type = 'rbf_kmeans'
        final_w = w_full
        final_centers = centers_full
        final_gamma = gamma
    
    elif best_method == 'linear':
        Phi_full = np.hstack([X_norm, np.ones((n_samples, 1))])
        A = Phi_full.T @ Phi_full + 1e-6 * np.eye(Phi_full.shape[1])
        b = Phi_full.T @ y
        try:
            w_full = np.linalg.solve(A, b)
        except:
            w_full, *_ = np.linalg.lstsq(A, b, rcond=None)
        
        final_type = 'linear'
        final_w = w_full
    
    # Capture for closure
    final_min_vals = min_vals
    final_range_vals = range_vals
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        X_norm_test = (X - final_min_vals) / final_range_vals
        
        if final_type == 'sindy':
            Phi_test = _build_sindy_library(X_norm_test, max_poly_degree=best_params['max_poly'], 
                                            max_trig_freq=best_params['max_trig'])
            if not np.all(np.isfinite(Phi_test)):
                Phi_test = np.hstack([X_norm_test, np.ones((X_norm_test.shape[0], 1))])
                return Phi_test @ final_w
            return Phi_test @ final_w + final_mean
        
        elif final_type == 'linear_fallback':
            Phi_test = np.hstack([X_norm_test, np.ones((X_norm_test.shape[0], 1))])
            return Phi_test @ final_w
        
        elif final_type == 'kernel_rbf':
            K_test = _compute_composite_kernel(X_norm_test, X_norm, final_params)
            return K_test @ final_coeffs
        
        elif final_type == 'rbf_kmeans':
            def _rbf_feats(Xq, centers, gamma):
                X_sq = np.sum(Xq**2, axis=1, keepdims=True)
                C_sq = np.sum(centers**2, axis=1, keepdims=True)
                sq_dists = X_sq + C_sq.T - 2 * (Xq @ centers.T)
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-gamma * sq_dists)
            
            def _poly_feats(Xq, deg=2):
                n, d = Xq.shape
                feats = [np.ones((n, 1))]
                for deg_i in range(1, deg + 1):
                    for combo in combinations_with_replacement(range(d), deg_i):
                        feat = np.ones(n)
                        for idx in combo:
                            feat *= Xq[:, idx]
                        feats.append(feat[:, np.newaxis])
                return np.hstack(feats)
            
            Phi_test_rbf = _rbf_feats(X_norm_test, final_centers, final_gamma)
            Phi_test_poly = _poly_feats(X_norm_test, 2)
            Phi_test = np.hstack([Phi_test_rbf, Phi_test_poly])
            return Phi_test @ final_w
        
        elif final_type == 'linear':
            Phi_test = np.hstack([X_norm_test, np.ones((X_norm_test.shape[0], 1))])
            return Phi_test @ final_w
        
        else:
            # Ultimate fallback: mean
            return np.full(X_norm_test.shape[0], np.mean(y))
    
    return predict
# EVOLVE-BLOCK-END
