"""Symbolic regression with ensemble of models and adaptive selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def _fit_ridge(X, y, lam=1e-6):
    """Fit ridge regression, return weights and intercept."""
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    b = Xb.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _predict_ridge(X, w):
    """Predict using ridge model."""
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    return Xb @ w

def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    # Degree 2
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Degree 3 (only if degree >= 3)
    if degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _rbf_features(X, centers, gamma):
    """Generate RBF features."""
    n = X.shape[0]
    d = X.shape[1]
    # Normalize centers
    center_mean = centers.mean(axis=0)
    center_std = centers.std(axis=0) + 1e-10
    X_norm = (X - center_mean) / center_std
    C_norm = (centers - center_mean) / center_std
    
    # Compute distances
    diff = X_norm[:, np.newaxis, :] - C_norm[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.exp(-gamma * dist_sq)

def _fourier_features(X, n_freqs=5):
    """Generate Fourier features."""
    n, d = X.shape
    cols = [np.ones(n)]
    # Normalize X
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    for freq in range(1, n_freqs + 1):
        for j in range(d):
            cols.append(np.sin(2 * np.pi * freq * X_norm[:, j]))
            cols.append(np.cos(2 * np.pi * freq * X_norm[:, j]))
    return np.column_stack(cols)

def _hybrid_features(X, n_freqs=5):
    """Hybrid: linear + quadratic + cubic pure + sin/cos at multiple frequencies."""
    n, d = X.shape
    cols = [np.ones(n)]
    # Linear
    for i in range(d):
        cols.append(X[:, i])
    # Quadratic
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Cubic pure powers
    for i in range(d):
        cols.append(X[:, i] ** 3)
    # Fourier
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    for freq in range(1, n_freqs + 1):
        for j in range(d):
            cols.append(np.sin(2 * np.pi * freq * X_norm[:, j]))
            cols.append(np.cos(2 * np.pi * freq * X_norm[:, j]))
    return np.column_stack(cols)

def _gp_kernel(X1, X2, lengthscale, noise, linear_mix):
    """Compute GP kernel with RBF + linear + noise."""
    # RBF kernel
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    K_rbf = np.exp(-0.5 * dist_sq / (lengthscale ** 2))
    
    # Linear kernel
    K_lin = X1 @ X2.T
    
    # Combine
    K = K_rbf + linear_mix * K_lin
    
    # Add noise for X1 == X2
    if X1.shape[0] == X2.shape[0] and np.allclose(X1, X2):
        K = K + noise * np.eye(X1.shape[0])
    
    return K

def _gp_marginal_likelihood(params, X, y):
    """Compute negative log marginal likelihood for GP."""
    try:
        lengthscale = np.exp(params[0])
        noise = np.exp(params[1])
        linear_mix = np.exp(params[2])
        
        K = _gp_kernel(X, X, lengthscale, noise, linear_mix)
        K += 1e-10 * np.eye(K.shape[0])
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        nll = 0.5 * y.T @ alpha + 0.5 * log_det + 0.5 * X.shape[0] * np.log(2 * np.pi)
        
        if not np.isfinite(nll):
            return 1e10
        
        return nll
    except Exception:
        return 1e10

def _gp_predict(X_train, y_train, X_test, params):
    """Make predictions with GP."""
    try:
        lengthscale = np.exp(params[0])
        noise = np.exp(params[1])
        linear_mix = np.exp(params[2])
        
        K_train = _gp_kernel(X_train, X_train, lengthscale, noise, linear_mix)
        K_train += 1e-10 * np.eye(K_train.shape[0])
        
        K_test = _gp_kernel(X_test, X_train, lengthscale, 0, linear_mix)
        
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            # Fall back to ridge
            w = _fit_ridge(np.column_stack([np.ones(len(X_train)), X_train]), y_train, 1e-6)
            pred = np.column_stack([np.ones(len(X_test)), X_test]) @ w
            return np.where(np.isfinite(pred), pred, 0.0)
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        v = np.linalg.solve(L, K_test.T)
        
        mu = v.T @ alpha
        return mu
    except Exception:
        # Fall back to ridge
        w = _fit_ridge(np.column_stack([np.ones(len(X_train)), X_train]), y_train, 1e-6)
        pred = np.column_stack([np.ones(len(X_test)), X_test]) @ w
        return np.where(np.isfinite(pred), pred, 0.0)

def _cv_score_multi_lambda(X, y, feature_fn, n_folds=5, lam_grid=None):
    """Cross-validate a feature function with lambda sweep."""
    if lam_grid is None:
        lam_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    best_score = np.inf
    best_lam = lam_grid[0]
    
    for lam in lam_grid:
        mse_sum = 0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            try:
                F_train = feature_fn(X_train)
                F_val = feature_fn(X_val)
                w = _fit_ridge(F_train, y_train, lam)
                pred = _predict_ridge(F_val, w)
                if not np.all(np.isfinite(pred)):
                    mse_sum = np.inf
                    break
                mse_sum += np.mean((pred - y_val) ** 2)
            except Exception:
                mse_sum = np.inf
                break
        if mse_sum < best_score:
            best_score = mse_sum / n_folds
            best_lam = lam
    
    return best_score, best_lam

def _cv_score_gp(X, y, n_folds=5):
    """Cross-validate GP with parameter optimization."""
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    best_score = np.inf
    best_params = None
    
    for fold in range(n_folds):
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Optimize GP parameters
        result = minimize_scalar(
            _gp_marginal_likelihood,
            bounds=[(-5, 5), (-10, 2), (-5, 5)],
            args=(X_train, y_train),
            method='bounded',
            options={'maxiter': 50}
        )
        
        if result.fun < 1e9:
            params = result.x
            pred = _gp_predict(X_train, y_train, X_val, params)
            if np.all(np.isfinite(pred)):
                mse = np.mean((pred - y_val) ** 2)
                if mse < best_score:
                    best_score = mse
                    best_params = params
    
    return best_score, best_params

def fit(X_train, y_train):
    """Fit a regression model using ensemble of models with adaptive selection."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Normalize for feature generation
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    # Lambda grid for ridge
    lam_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    
    # Candidate feature functions for ridge models
    candidates = []
    
    # Polynomial degree 2
    candidates.append(('poly2', lambda Xn: _poly_features(Xn, 2)))
    
    # Polynomial degree 3
    if d <= 3:
        candidates.append(('poly3', lambda Xn: _poly_features(Xn, 3)))
    
    # RBF with different gammas (use more centers)
    rbf_centers = X_norm[::max(1, n // 20)]
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        candidates.append((f'rbf_g{gamma}', lambda Xn, g=gamma, c=rbf_centers: _rbf_features(Xn, c, g)))
    
    # Fourier features
    candidates.append(('fourier5', lambda Xn: _fourier_features(Xn, 5)))
    candidates.append(('fourier10', lambda Xn: _fourier_features(Xn, 10)))
    
    # Hybrid features
    candidates.append(('hybrid5', lambda Xn: _hybrid_features(Xn, 5)))
    
    # Evaluate candidates with cross-validation + lambda sweep
    best_score = np.inf
    best_fn = None
    best_name = None
    best_lam = 1e-6
    best_type = 'ridge'
    
    for name, fn in candidates:
        if time.time() > deadline - 60:
            break
        try:
            score, lam = _cv_score_multi_lambda(X_norm, y, fn, n_folds=5, lam_grid=lam_grid)
            if score < best_score:
                best_score = score
                best_fn = fn
                best_name = name
                best_lam = lam
                best_type = 'ridge'
        except Exception:
            continue
    
    # Try GP if time allows
    if time.time() < deadline - 30:
        try:
            gp_score, gp_params = _cv_score_gp(X_norm, y, n_folds=5)
            if gp_score < best_score:
                best_score = gp_score
                best_fn = None
                best_name = 'gp'
                best_lam = None
                best_type = 'gp'
                best_params = gp_params
        except Exception:
            pass
    
    # If no valid candidate, fall back to simple linear
    if best_fn is None and best_type != 'gp':
        best_fn = lambda Xn: np.column_stack([np.ones(len(Xn)), Xn])
        best_name = 'linear'
        best_lam = 1e-6
        best_type = 'ridge'
    
    # Fit final model on all data
    if best_type == 'ridge':
        F = best_fn(X_norm)
        w = _fit_ridge(F, y, lam=best_lam)
        
        # Create predict function
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            X_test_norm = (X_test - X_mean) / X_std
            F_test = best_fn(X_test_norm)
            pred = _predict_ridge(F_test, w)
            # Ensure finite output
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
    else:
        # GP prediction
        gp_params = best_params
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            X_test_norm = (X_test - X_mean) / X_std
            pred = _gp_predict(X_norm, y, X_test_norm, gp_params)
            # Ensure finite output
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
    
    return predict
# EVOLVE-BLOCK-END
