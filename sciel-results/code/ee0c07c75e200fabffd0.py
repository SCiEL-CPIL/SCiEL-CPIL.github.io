"""
Hybrid Symbolic Regression: Polynomial + Sparse Kernel (Nyström) Ensemble.

Strategy:
1. Try a range of polynomial degrees (1-6) using Ridge Regression.
2. Try a range of RBF kernels using Nyström approximation (Sparse Kernel Regression).
   This is much faster than full KRR and allows extensive hyperparameter search.
3. Select the best model based on a penalized training error (AIC-like) to avoid
   the overhead and variance of K-fold CV on small datasets.
4. Return a weighted average of the top 2-3 distinct model families if they perform similarly,
   or the single best model.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement

def _rbf_kernel(A, B, length_scale):
    """Compute RBF kernel between sets A and B."""
    A_sq = np.sum(A**2, axis=1)[:, None]
    B_sq = np.sum(B**2, axis=1)[None, :]
    sq_dists = A_sq + B_sq - 2.0 * (A @ B.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * length_scale**2))

def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            feat = np.ones(n)
            for idx in combo:
                feat = feat * X[:, idx]
            cols.append(feat)
    return np.column_stack(cols)

def _fit_poly_ridge(X, y, degree, lam):
    """Fit polynomial ridge regression. Returns predict function and effective degrees of freedom."""
    F = _poly_features(X, degree)
    A = F.T @ F + lam * np.eye(F.shape[1])
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    
    def predict(X_test):
        F_test = _poly_features(X_test, degree)
        return F_test @ w
    
    # Approximate effective degrees of freedom for AIC
    # For Ridge: df = trace(F (F'F + lam I)^-1 F')
    # Approximation: number of non-zero coefficients
    df = np.sum(np.abs(w) > 1e-8)
    return predict, df

def _fit_sparse_kernel(X, y, length_scale, n_landmarks, lam):
    """Fit Sparse Kernel Regression using Nyström approximation.
    
    We select n_landmarks points as centers.
    Feature map: phi(x) = K(x, L) @ (K(L,L) + lam*I)^-1
    Then we do linear regression on phi(x).
    """
    n = X.shape[0]
    if n_landmarks >= n:
        n_landmarks = n
        
    # Select landmarks: Uniform random sampling is usually sufficient for smooth functions
    # Greedy would be better but slower. Let's use random for speed in the search loop.
    # We use a fixed seed based on hash of parameters for reproducibility in search,
    # but here we just pass X and y.
    rng = np.random.RandomState(42) # Fixed seed for determinism in candidate generation
    landmarks = rng.choice(n, size=n_landmarks, replace=False)
    
    X_L = X[landmarks]
    
    # Compute K_L (landmarks x landmarks)
    K_L = _rbf_kernel(X_L, X_L, length_scale)
    
    # Regularize and invert K_L
    # Note: lam here is for the kernel matrix stability, not the regression lambda
    K_L_reg = K_L + 1e-6 * np.eye(n_landmarks)
    try:
        K_L_inv = np.linalg.inv(K_L_reg)
    except np.linalg.LinAlgError:
        K_L_inv = np.linalg.pinv(K_L_reg)
    
    # Compute K_XL (n x landmarks)
    K_XL = _rbf_kernel(X, X_L, length_scale)
    
    # Nyström Feature Map: Phi = K_XL @ K_L_inv
    # The model is y = Phi @ w
    # We fit w using Ridge Regression to avoid overfitting in the landmark space
    Phi = K_XL @ K_L_inv
    
    # Ridge Regression for the coefficients w
    A = Phi.T @ Phi + lam * np.eye(n_landmarks)
    b = Phi.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    
    def predict(X_test):
        K_test_L = _rbf_kernel(X_test, X_L, length_scale)
        Phi_test = K_test_L @ K_L_inv
        return Phi_test @ w
    
    df = np.sum(np.abs(w) > 1e-8)
    return predict, df

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin from 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better numerical stability and kernel scaling
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    X_norm = (X - x_min) / x_range
    
    var_y = np.var(y)
    if var_y < 1e-10:
        var_y = 1.0
    
    # Compute median distance for length scale heuristics
    if n > 100:
        rng_sub = np.random.RandomState(0)
        idx_sub = rng_sub.choice(n, size=100, replace=False)
        X_sub = X_norm[idx_sub]
        dists_sub = np.sqrt(np.sum((X_sub[:, None, :] - X_sub[None, :, :])**2, axis=2))
        np.fill_diagonal(dists_sub, np.inf)
        median_dist = np.median(dists_sub[dists_sub > 0])
    else:
        dists_full = np.sqrt(np.sum((X_norm[:, None, :] - X_norm[None, :, :])**2, axis=2))
        np.fill_diagonal(dists_full, np.inf)
        median_dist = np.median(dists_full[dists_full > 0])
    
    if median_dist == 0 or np.isnan(median_dist):
        median_dist = 1.0

    candidates = [] # List of (aic_score, predict_fn, family_type)
    
    # 1. Polynomial Candidates
    # Degrees 1 to 6
    for deg in range(1, 7):
        if time.time() > DEADLINE:
            break
        for lam in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
            if time.time() > DEADLINE:
                break
            try:
                pred_fn, df = _fit_poly_ridge(X_norm, y, deg, lam)
                y_pred = pred_fn(X_norm)
                mse = np.mean((y - y_pred)**2)
                # AIC = n * log(RSS/n) + 2 * k
                # Use AICc for small n
                k = df + 1 # +1 for variance
                aic = n * np.log(mse / n) + 2 * k
                # Penalty for large degree to prefer simplicity
                aic += deg * 0.5 
                
                candidates.append((aic, pred_fn, f"poly_{deg}"))
            except Exception:
                continue

    # 2. Sparse Kernel (Nyström) Candidates
    # Length scales around median distance
    ls_list = [median_dist * f for f in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]]
    # Number of landmarks: small for smooth, larger for complex
    nl_list = [10, 20, 50, 100]
    # Regularization for the linear regression part
    lam_list = [1e-6, 1e-4, 1e-2]
    
    for ls in ls_list:
        if time.time() > DEADLINE:
            break
        for nl in nl_list:
            if time.time() > DEADLINE:
                break
            for lam in lam_list:
                if time.time() > DEADLINE:
                    break
                try:
                    pred_fn, df = _fit_sparse_kernel(X_norm, y, ls, nl, lam)
                    y_pred = pred_fn(X_norm)
                    mse = np.mean((y - y_pred)**2)
                    k = df + 1
                    aic = n * np.log(mse / n) + 2 * k
                    # Penalty for complexity (number of landmarks)
                    aic += nl * 0.1
                    
                    candidates.append((aic, pred_fn, f"kernel_{ls:.2f}_{nl}"))
                except Exception:
                    continue
    
    # 3. Linear Fallback (always safe)
    try:
        F_lin = np.column_stack([np.ones(n), X_norm])
        w_lin = np.linalg.lstsq(F_lin, y, rcond=None)[0]
        def lin_predict(X_test):
            Xn = (X_test - x_min) / x_range
            F_test = np.column_stack([np.ones(len(Xn)), Xn])
            return F_test @ w_lin
        y_pred = lin_predict(X_norm)
        mse = np.mean((y - y_pred)**2)
        k = d + 1
        aic = n * np.log(mse / n) + 2 * k
        candidates.append((aic, lin_predict, "linear"))
    except Exception:
        pass

    if len(candidates) == 0:
        # Ultimate fallback: constant
        y_mean = np.mean(y)
        def predict_const(X):
            return np.full(len(X), y_mean)
        return predict_const

    # Sort by AIC
    candidates.sort(key=lambda x: x[0])
    
    # Select top models
    # We will take the best model.
    # If the second best is very close (within 2 AIC units) and from a different family,
    # we might average them for robustness.
    
    best_aic, best_pred, best_family = candidates[0]
    
    # Check for ensemble opportunity
    ensemble_preds = [best_pred]
    ensemble_weights = [1.0]
    
    # Look at the next few candidates
    for i in range(1, min(3, len(candidates))):
        aic_i, pred_i, fam_i = candidates[i]
        # If AIC is close and family is different, include in ensemble
        if aic_i - best_aic < 2.0 and not fam_i.startswith(best_family.split('_')[0]):
            ensemble_preds.append(pred_i)
            ensemble_weights.append(1.0)
            break # Only add one extra model for simplicity
    
    if len(ensemble_preds) == 1:
        final_predict = best_pred
    else:
        # Weighted average
        w_sum = sum(ensemble_weights)
        weights = [w / w_sum for w in ensemble_weights]
        def ensemble_predict(X_test):
            preds = [p(X_test) for p in ensemble_preds]
            result = np.zeros_like(preds[0])
            for p, w in zip(preds, weights):
                result += w * p
            return result
        final_predict = ensemble_predict

    # Wrap to ensure finite output and correct shape
    def safe_predict(X):
        X = np.asarray(X, dtype=np.float64)
        try:
            pred = final_predict(X)
            pred = np.asarray(pred, dtype=np.float64)
            # Handle any numerical issues
            pred = np.where(np.isfinite(pred), pred, np.mean(y))
            return pred
        except Exception:
            return np.full(len(X), np.mean(y))
    
    return safe_predict

# EVOLVE-BLOCK-END
