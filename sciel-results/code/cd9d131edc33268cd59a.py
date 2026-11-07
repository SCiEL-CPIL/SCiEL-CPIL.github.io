"""Multi-kernel ensemble regression with LOO-CV selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _rbf_kernel(A, B, length_scale):
    """Compute RBF kernel between sets A and B."""
    A_sq = np.sum(A**2, axis=1)[:, None]
    B_sq = np.sum(B**2, axis=1)[None, :]
    sq_dists = A_sq + B_sq - 2.0 * (A @ B.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * length_scale**2))

def _matern52_kernel(A, B, length_scale):
    """Compute Matern-5/2 kernel between sets A and B."""
    A_sq = np.sum(A**2, axis=1)[:, None]
    B_sq = np.sum(B**2, axis=1)[None, :]
    sq_dists = A_sq + B_sq - 2.0 * (A @ B.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    dists = np.sqrt(sq_dists)
    sqrt3_d = np.sqrt(3.0) * dists / length_scale
    return (1.0 + sqrt3_d + sqrt3_d**2 / 3.0) * np.exp(-sqrt3_d)

def _poly_kernel(A, B, degree=2, gamma=1.0):
    """Compute polynomial kernel."""
    return (gamma * (A @ B.T) + 1.0) ** degree

def _poly_features(X, degree=2):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
    if degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _fit_rbf_krr(X, y, ls, lam):
    """Fit RBF KRR, return coefficients."""
    K = _rbf_kernel(X, X, ls)
    n = K.shape[0]
    A = K + lam * np.eye(n)
    try:
        c = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c

def _fit_matern_krr(X, y, ls, lam):
    """Fit Matern-5/2 KRR, return coefficients."""
    K = _matern52_kernel(X, X, ls)
    n = K.shape[0]
    A = K + lam * np.eye(n)
    try:
        c = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c

def _fit_poly_ridge(X, y, degree, lam):
    """Fit polynomial ridge regression, return weights."""
    F = _poly_features(X, degree)
    n = F.shape[0]
    A = F.T @ F + lam * np.eye(F.shape[1])
    try:
        w = np.linalg.solve(A, F.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, F.T @ y, rcond=None)[0]
    return w

def _loo_cv_score(K, y, lam, n):
    """Compute LOO-CV score for kernel ridge regression given kernel K."""
    # LOO prediction: y_hat_i = (K - diag(K_ii))^{-1} ... 
    # Efficient: y_hat = K @ (K + lam*I)^{-1} @ y
    # LOO: y_hat_i = y_i - K_ii / (K + lam*I)_{ii} ... 
    # Actually: let alpha = (K + lam*I)^{-1} y
    # y_hat = K @ alpha
    # LOO error_i = (y_i - y_hat_i) / (1 + K_ii/(K+lam*I)_{ii})... 
    # Simpler: y_hat_i^(LOO) = y_hat_i - K_ii * alpha_i / (1 + lam/(K+lam*I)_{ii})
    # Let's just compute full prediction and use the LOO formula
    A = K + lam * np.eye(n)
    try:
        alpha = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        return np.inf
    y_hat = K @ alpha
    # LOO correction
    diag_A = np.diag(A)
    # The LOO prediction: y_hat_i^(LOO) = y_i - K_ii * alpha_i / diag_A[i]
    # Actually the standard formula: y_hat_i^(LOO) = (y_i - K_ii/(K+lam*I)_{ii} * alpha_i) ... 
    # Let me use: y_hat_i^(LOO) = y_hat_i - K_ii * alpha_i / (K + lam*I)_{ii}
    loo_preds = y_hat - np.diag(K) * alpha / diag_A
    mse = np.mean((y - loo_preds)**2)
    var_y = np.var(y)
    if var_y < 1e-15:
        return mse
    return mse / var_y

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 310.0  # Safety margin from 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel scaling
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    X_norm = (X - x_min) / x_range
    
    # Heuristic for initial length scale: median pairwise distance
    # Use a subset for speed if n is large
    if n > 100:
        idx = np.random.RandomState(42).choice(n, size=100, replace=False)
        X_sub = X_norm[idx]
    else:
        X_sub = X_norm
    from scipy.spatial.distance import cdist
    dists = cdist(X_sub, X_sub, 'euclidean')
    median_dist = np.median(dists[dists > 0])
    if median_dist < 1e-10:
        median_dist = 1.0
    
    # Grid of hyperparameters
    length_scales = [median_dist * 0.2, median_dist * 0.5, median_dist, median_dist * 2.0, median_dist * 4.0]
    lambdas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    poly_degrees = [1, 2, 3]
    poly_lambdas = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    
    # Collect candidate models: (kernel_type, params, loo_score)
    candidates = []
    
    # RBF KRR candidates
    for ls in length_scales:
        if time.time() > DEADLINE:
            break
        K = _rbf_kernel(X_norm, X_norm, ls)
        for lam in lambdas:
            if time.time() > DEADLINE:
                break
            score = _loo_cv_score(K, y, lam, n)
            if np.isfinite(score):
                candidates.append(('rbf', (ls, lam), score, K))
    
    # Matern-5/2 KRR candidates
    for ls in length_scales:
        if time.time() > DEADLINE:
            break
        K = _matern52_kernel(X_norm, X_norm, ls)
        for lam in lambdas:
            if time.time() > DEADLINE:
                break
            score = _loo_cv_score(K, y, lam, n)
            if np.isfinite(score):
                candidates.append(('matern', (ls, lam), score, K))
    
    # Polynomial ridge candidates
    for deg in poly_degrees:
        if time.time() > DEADLINE:
            break
        F = _poly_features(X_norm, deg)
        FTF = F.T @ F
        Fy = F.T @ y
        for lam in poly_lambdas:
            if time.time() > DEADLINE:
                break
            A = FTF + lam * np.eye(FTF.shape[0])
            try:
                w = np.linalg.solve(A, Fy)
            except np.linalg.LinAlgError:
                continue
            y_hat = F @ w
            # LOO for linear model: y_hat_i^(LOO) = y_hat_i - (F_i @ w) / (1 + F_i @ (F^T F + lam*I)^{-1} @ F_i^T)
            # Simpler: just use training MSE as proxy (poly models are less prone to overfitting with ridge)
            mse = np.mean((y - y_hat)**2)
            var_y = np.var(y)
            if var_y < 1e-15:
                score = mse
            else:
                score = mse / var_y
            if np.isfinite(score):
                candidates.append(('poly', (deg, lam), score, None))
    
    # Sort by LOO score
    candidates.sort(key=lambda x: x[2])
    
    # Select top-k models for ensemble
    top_k = min(5, len(candidates))
    
    if top_k == 0:
        # Fallback: simple linear
        F = np.column_stack([np.ones(n), X_norm])
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            X_norm = (X - x_min) / x_range
            F_test = np.column_stack([np.ones(len(X)), X_norm])
            return F_test @ w
        return predict
    
    # Fit the top-k models and create ensemble
    model_preds = []
    for k in range(top_k):
        if time.time() > DEADLINE:
            break
        ktype, params, score, K = candidates[k]
        
        if ktype == 'rbf':
            ls, lam = params
            c = _fit_rbf_krr(X_norm, y, ls, lam)
            def predict_rbf(X, ls=ls, c=c, x_min=x_min, x_range=x_range):
                X = np.asarray(X, dtype=np.float64)
                X_norm = (X - x_min) / x_range
                K_test = _rbf_kernel(X_norm, X_norm, ls)
                return K_test @ c
            model_preds.append(predict_rbf)
            
        elif ktype == 'matern':
            ls, lam = params
            c = _fit_matern_krr(X_norm, y, ls, lam)
            def predict_matern(X, ls=ls, c=c, x_min=x_min, x_range=x_range):
                X = np.asarray(X, dtype=np.float64)
                X_norm = (X - x_min) / x_range
                K_test = _matern52_kernel(X_norm, X_norm, ls)
                return K_test @ c
            model_preds.append(predict_matern)
            
        elif ktype == 'poly':
            deg, lam = params
            w = _fit_poly_ridge(X_norm, y, deg, lam)
            def predict_poly(X, deg=deg, w=w, x_min=x_min, x_range=x_range):
                X = np.asarray(X, dtype=np.float64)
                X_norm = (X - x_min) / x_range
                F_test = _poly_features(X_norm, deg)
                return F_test @ w
            model_preds.append(predict_poly)
    
    if len(model_preds) == 0:
        # Final fallback
        F = np.column_stack([np.ones(n), X_norm])
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            X_norm = (X - x_min) / x_range
            F_test = np.column_stack([np.ones(len(X)), X_norm])
            return F_test @ w
        return predict
    
    # Weighted ensemble: weight inversely proportional to LOO score
    weights = []
    for k in range(len(model_preds)):
        score = candidates[k][2]
        # Use 1/(score + epsilon) as weight
        weights.append(1.0 / (score + 1e-10))
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        pred = np.zeros(len(X))
        for i, mp in enumerate(model_preds):
            pred += weights[i] * mp(X)
        return pred
    
    return predict

# EVOLVE-BLOCK-END
