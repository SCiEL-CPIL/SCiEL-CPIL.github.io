"""
Structural Change: Ensemble of diverse basis function regressors with k-fold CV selection.
Instead of a single RBF kernel with one bandwidth, we train multiple models:
1. Polynomial regressors of varying degrees.
2. RBF kernel regressors with different length scales.
3. Product kernels (RBF * Polynomial) for capturing mixed behaviors.
Each model is evaluated using 5-fold cross-validation. The final prediction is the
average of the top-performing models (or all models if they are all good), which
provides robustness against overfitting to a specific kernel hyperparameter and
handles equations that might be better fit by a combination of features rather
than a single smooth kernel.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n_samples, n_features = X.shape

    # Normalize inputs to [0, 1] for numerical stability and consistent kernel scales
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    X_norm = (X - min_vals) / range_vals

    # Define candidate models
    # Each model is a function that takes X (normalized) and returns feature matrix or predictions directly
    # We will use linear regression on features for all, which is fast and stable with regularization.

    def make_poly_features(X, degree):
        """Generate polynomial features up to given degree."""
        n_samples, n_features = X.shape
        cols = [np.ones(n_samples)]
        if degree >= 1:
            cols.append(X)
        for d in range(2, degree + 1):
            for combo in combinations_with_replacement(range(n_features), d):
                prod = np.ones(n_samples)
                for idx in combo:
                    prod *= X[:, idx]
                cols.append(prod)
        return np.column_stack(cols)

    def make_rbf_features(X, length_scale):
        """Generate RBF features using all training points as centers (Nystrom/KRR style)."""
        # For efficiency and stability, we use the training points themselves as centers.
        # This is equivalent to Kernel Ridge Regression.
        # To keep the "feature matrix" view consistent, we actually solve the KRR problem directly
        # in the training loop, but here we just define the kernel function.
        # However, to fit the "linear on features" framework, we'd need to invert the kernel.
        # Instead, let's treat KRR models separately in the evaluation loop.
        pass # This function is a placeholder to indicate type. Actual logic in cv_eval.

    # List of candidate configurations: (type, params)
    candidates = []

    # Polynomials: degrees 1 to 4 (max depends on n_features to avoid explosion)
    max_deg = 3 if n_features > 2 else 4
    for deg in range(1, max_deg + 1):
        candidates.append(('poly', deg))

    # RBF Kernels: various length scales
    # Length scales in normalized space [0,1]^d
    # Small l -> high freq, Large l -> low freq
    rbfs = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    for l in rbfs:
        candidates.append(('rbf', l))

    # Product Kernels: RBF * Poly (degree 1 or 2)
    # This can capture "smooth but with varying slope" behaviors
    for l in [0.5, 1.0]:
        for deg in [1, 2]:
            candidates.append(('prod', (l, deg)))

    # Cross-Validation Setup
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[:n_samples % n_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

    def evaluate_model(model_type, params):
        """
        Performs 5-fold CV. Returns mean CV MSE.
        If time runs out, returns inf.
        """
        if time.time() > deadline - 10:
            return float('inf')
            
        mse_sum = 0.0
        valid_folds = 0
        
        for i in range(n_folds):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_tr = X_norm[tr_idx]
            y_tr = y[tr_idx]
            X_val = X_norm[val_idx]
            y_val = y[val_idx]
            
            try:
                if model_type == 'poly':
                    deg = params
                    # Check if feature count is too high relative to samples
                    # Rough estimate of poly features
                    n_feat_est = 1
                    if deg >= 1: n_feat_est += n_features
                    for d in range(2, deg + 1):
                        # Number of combinations with replacement of n_features taken d
                        # C(n+d-1, d)
                        n_feat_est += int(np.prod([n_features + d - 1 - k for k in range(d)]) / np.prod([k+1 for k in range(d)]))
                    
                    if n_feat_est > len(y_tr):
                        # Too many features, use high regularization or skip
                        # We'll let lstsq handle it, but it might be unstable
                        pass
                    
                    F_tr = make_poly_features(X_tr, deg)
                    F_val = make_poly_features(X_val, deg)
                    
                    # Ridge regression for stability
                    lam = 1e-2
                    if n_feat_est > len(y_tr):
                        lam = 1e-1
                    
                    # Solve (F^T F + lam I) w = F^T y
                    A = F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1])
                    b = F_tr.T @ y_tr
                    w = np.linalg.solve(A, b)
                    
                    y_pred = F_val @ w
                    mse = np.mean((y_pred - y_val)**2)
                    mse_sum += mse
                    valid_folds += 1

                elif model_type == 'rbf':
                    l = params
                    gamma = 1.0 / (2.0 * l**2)
                    
                    # Compute kernel matrices
                    # K(x, x') = exp(-gamma * ||x-x'||^2)
                    def sq_dists(A, B):
                        A_sq = np.sum(A**2, axis=1, keepdims=True)
                        B_sq = np.sum(B**2, axis=1, keepdims=True)
                        sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
                        return np.maximum(sq_dists, 0)
                    
                    sq_dists_tr = sq_dists(X_tr, X_tr)
                    K_tr = np.exp(-gamma * sq_dists_tr)
                    
                    # Regularization
                    reg = 1e-6
                    K_tr_reg = K_tr + reg * np.eye(len(y_tr))
                    
                    # Solve K alpha = y
                    alpha = np.linalg.solve(K_tr_reg, y_tr)
                    
                    sq_dists_val = sq_dists(X_val, X_tr)
                    K_val = np.exp(-gamma * sq_dists_val)
                    y_pred = K_val @ alpha
                    
                    mse = np.mean((y_pred - y_val)**2)
                    mse_sum += mse
                    valid_folds += 1

                elif model_type == 'prod':
                    l, deg = params
                    gamma = 1.0 / (2.0 * l**2)
                    
                    # Product kernel: K(x, x') = K_rbf(x, x') * K_poly(x, x')
                    # K_poly is linear: (1 + x^T x') or just x^T x'?
                    # Let's use (1 + x^T x') for degree 1 equivalent, or higher.
                    # Simpler: Use features. F_prod(x) = [1, x, x*x, ...] * rbf_basis?
                    # Actually, implementing product kernel in feature space is tricky if we mix.
                    # Let's stick to Kernel Space: K = K_rbf * K_lin
                    # K_lin(x, x') = 1 + x^T x'
                    
                    def sq_dists(A, B):
                        A_sq = np.sum(A**2, axis=1, keepdims=True)
                        B_sq = np.sum(B**2, axis=1, keepdims=True)
                        sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
                        return np.maximum(sq_dists, 0)
                    
                    sq_dists_tr = sq_dists(X_tr, X_tr)
                    K_rbf_tr = np.exp(-gamma * sq_dists_tr)
                    K_lin_tr = 1.0 + (X_tr @ X_tr.T)
                    K_tr = K_rbf_tr * K_lin_tr
                    
                    reg = 1e-6
                    K_tr_reg = K_tr + reg * np.eye(len(y_tr))
                    alpha = np.linalg.solve(K_tr_reg, y_tr)
                    
                    sq_dists_val = sq_dists(X_val, X_tr)
                    K_rbf_val = np.exp(-gamma * sq_dists_val)
                    K_lin_val = 1.0 + (X_val @ X_tr.T)
                    K_val = K_rbf_val * K_lin_val
                    y_pred = K_val @ alpha
                    
                    mse = np.mean((y_pred - y_val)**2)
                    mse_sum += mse
                    valid_folds += 1

            except Exception:
                continue
        
        if valid_folds == 0:
            return float('inf')
            
        return mse_sum / valid_folds

    # Evaluate all candidates
    cv_scores = []
    for m_type, params in candidates:
        if time.time() > deadline - 20:
            break
        score = evaluate_model(m_type, params)
        cv_scores.append((score, m_type, params))
        
    # Sort by CV score
    cv_scores.sort(key=lambda x: x[0])
    
    # Select top models. 
    # To be robust, we average the top 3 models that have finite scores and are significantly better than the worst.
    # If the top model is much better, we might just use it. But averaging top-k is generally safer against CV noise.
    # Let's take the top 3.
    
    selected_models = []
    best_score = cv_scores[0][0] if cv_scores else float('inf')
    
    # Filter out bad models (inf or very high relative to best)
    # If best is inf, we have a problem.
    if best_score == float('inf'):
        # Fallback to simple linear regression
        F = make_poly_features(X_norm, 1)
        lam = 1e-2
        A = F.T @ F + lam * np.eye(F.shape[1])
        b = F.T @ y
        w = np.linalg.solve(A, b)
        def predict(X):
            X = np.asarray(X, dtype=float)
            X_norm_test = (X - min_vals) / range_vals
            F_test = make_poly_features(X_norm_test, 1)
            return F_test @ w
        return predict

    # Select top 3 finite scores
    for score, m_type, params in cv_scores:
        if np.isfinite(score):
            selected_models.append((m_type, params))
        if len(selected_models) >= 3:
            break
            
    if not selected_models:
        # Fallback
        F = make_poly_features(X_norm, 1)
        lam = 1e-2
        A = F.T @ F + lam * np.eye(F.shape[1])
        b = F.T @ y
        w = np.linalg.solve(A, b)
        def predict(X):
            X = np.asarray(X, dtype=float)
            X_norm_test = (X - min_vals) / range_vals
            F_test = make_poly_features(X_norm_test, 1)
            return F_test @ w
        return predict

    # Refit selected models on ALL data and store parameters
    # Structure: list of (type, params, solver_data)
    refit_models = []
    
    for m_type, params in selected_models:
        if time.time() > deadline - 5:
            break
            
        try:
            if m_type == 'poly':
                deg = params
                F_tr = make_poly_features(X_norm, deg)
                # Use same regularization logic
                n_feat_est = F_tr.shape[1]
                lam = 1e-2 if n_feat_est <= n_samples else 1e-1
                A = F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1])
                b = F_tr.T @ y
                w = np.linalg.solve(A, b)
                refit_models.append(('poly', deg, w))
                
            elif m_type == 'rbf':
                l = params
                gamma = 1.0 / (2.0 * l**2)
                def sq_dists(A, B):
                    A_sq = np.sum(A**2, axis=1, keepdims=True)
                    B_sq = np.sum(B**2, axis=1, keepdims=True)
                    sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
                    return np.maximum(sq_dists, 0)
                
                sq_dists_tr = sq_dists(X_norm, X_norm)
                K_tr = np.exp(-gamma * sq_dists_tr)
                reg = 1e-6
                K_tr_reg = K_tr + reg * np.eye(n_samples)
                alpha = np.linalg.solve(K_tr_reg, y)
                refit_models.append(('rbf', l, alpha))
                
            elif m_type == 'prod':
                l, deg = params
                gamma = 1.0 / (2.0 * l**2)
                def sq_dists(A, B):
                    A_sq = np.sum(A**2, axis=1, keepdims=True)
                    B_sq = np.sum(B**2, axis=1, keepdims=True)
                    sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
                    return np.maximum(sq_dists, 0)
                
                sq_dists_tr = sq_dists(X_norm, X_norm)
                K_rbf_tr = np.exp(-gamma * sq_dists_tr)
                K_lin_tr = 1.0 + (X_norm @ X_norm.T)
                K_tr = K_rbf_tr * K_lin_tr
                reg = 1e-6
                K_tr_reg = K_tr + reg * np.eye(n_samples)
                alpha = np.linalg.solve(K_tr_reg, y)
                refit_models.append(('prod', (l, deg), alpha))
                
        except Exception:
            continue

    # If no models refit successfully, fallback
    if not refit_models:
        F = make_poly_features(X_norm, 1)
        lam = 1e-2
        A = F.T @ F + lam * np.eye(F.shape[1])
        b = F.T @ y
        w = np.linalg.solve(A, b)
        def predict(X):
            X = np.asarray(X, dtype=float)
            X_norm_test = (X - min_vals) / range_vals
            F_test = make_poly_features(X_norm_test, 1)
            return F_test @ w
        return predict

    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - min_vals) / range_vals
        predictions = []
        
        for m_type, params, data in refit_models:
            try:
                if m_type == 'poly':
                    deg = params
                    w = data
                    F_test = make_poly_features(X_norm_test, deg)
                    pred = F_test @ w
                    predictions.append(pred)
                    
                elif m_type == 'rbf':
                    l = params
                    gamma = 1.0 / (2.0 * l**2)
                    alpha = data
                    def sq_dists(A, B):
                        A_sq = np.sum(A**2, axis=1, keepdims=True)
                        B_sq = np.sum(B**2, axis=1, keepdims=True)
                        sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
                        return np.maximum(sq_dists, 0)
                    
                    sq_dists_val = sq_dists(X_norm_test, X_norm)
                    K_val = np.exp(-gamma * sq_dists_val)
                    pred = K_val @ alpha
                    predictions.append(pred)
                    
                elif m_type == 'prod':
                    l, deg = params
                    gamma = 1.0 / (2.0 * l**2)
                    alpha = data
                    def sq_dists(A, B):
                        A_sq = np.sum(A**2, axis=1, keepdims=True)
                        B_sq = np.sum(B**2, axis=1, keepdims=True)
                        sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
                        return np.maximum(sq_dists, 0)
                    
                    sq_dists_val = sq_dists(X_norm_test, X_norm)
                    K_rbf_val = np.exp(-gamma * sq_dists_val)
                    K_lin_val = 1.0 + (X_norm_test @ X_norm.T)
                    K_val = K_rbf_val * K_lin_val
                    pred = K_val @ alpha
                    predictions.append(pred)
            except Exception:
                continue

        if not predictions:
            # Absolute fallback: mean of y
            return np.full(X.shape[0], np.mean(y))
            
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        
        # Ensure finite
        if not np.all(np.isfinite(avg_pred)):
            avg_pred = np.nan_to_num(avg_pred, nan=np.mean(y), posinf=np.max(y), neginf=np.min(y))
            
        return avg_pred

    return predict
# EVOLVE-BLOCK-END
