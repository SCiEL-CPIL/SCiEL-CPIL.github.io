"""Rewritten program: robust L1-regularized polynomial fit for unresolved tendency closure."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so


def _fista_lasso(V, y, lam, max_iter=2000, tol=1e-7):
    """
    Solve Lasso problem: min 0.5/N * ||y - Vw||^2 + lam * ||w||_1
    using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
    """
    n, d = V.shape
    # Lipschitz constant L for the gradient of the smooth part
    # L = max eigenvalue of (V^T V) / n
    try:
        VtV = V.T @ V
        eigvals = np.linalg.eigvalsh(VtV)
        L = max(eigvals.max() / n, 1e-12)
    except np.linalg.LinAlgError:
        L = 1.0

    w = np.zeros(d)
    w_prev = np.zeros(d)
    t = 1.0

    for _ in range(max_iter):
        # Gradient step
        grad = (V.T @ (V @ w - y)) / n
        w_new = w - grad / L

        # Proximal operator (soft thresholding)
        thresh = lam / L
        w_shrink = np.sign(w_new) * np.maximum(np.abs(w_new) - thresh, 0.0)

        # Check convergence
        if np.linalg.norm(w_shrink - w) < tol * (1 + np.linalg.norm(w)):
            w = w_shrink
            break

        # FISTA update
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        w = w_shrink + ((t - 1) / t_new) * (w_shrink - w)
        t = t_new

    return w


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Uses a robust L1 regularization (Lasso) implemented via FISTA to select
    the most relevant polynomial terms. The regularization strength is selected
    via 5-fold cross-validation to balance bias and variance.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    x = x.ravel()
    y = y.ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Build full polynomial design matrix (degrees 0-5)
    V = np.vander(x, N=6, increasing=True)
    
    # Normalize V and y for numerical stability in FISTA
    # FISTA is scale-sensitive, so normalization is crucial
    V_mean = V.mean(axis=0)
    V_std = V.std(axis=0)
    V_std[V_std < 1e-12] = 1.0
    V_norm = (V - V_mean) / V_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_norm = np.zeros_like(y)
        y_std_eff = 1.0
    else:
        y_norm = (y - y_mean) / y_std
        y_std_eff = y_std
    
    # Grid of L1 penalties to try
    # We search in log space
    log_lams = np.linspace(-6, 2, 50)
    lams = np.exp(log_lams)
    
    # 5-Fold Cross Validation
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42) # For deterministic CV split
    indices = np.random.permutation(indices)
    folds = np.array_split(indices, n_folds)
    
    best_lam = 0.0
    best_cv_mse = np.inf
    
    for lam in lams:
        cv_mses = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            V_train = V_norm[train_idx]
            y_train = y_norm[train_idx]
            V_val = V_norm[val_idx]
            y_val = y_norm[val_idx]
            
            # Solve Lasso on training fold
            w_fold = _fista_lasso(V_train, y_train, lam)
            
            # Evaluate on validation fold
            pred_val = V_val @ w_fold
            mse_val = np.mean((y_val - pred_val) ** 2)
            cv_mses.append(mse_val)
        
        avg_cv_mse = np.mean(cv_mses)
        if avg_cv_mse < best_cv_mse:
            best_cv_mse = avg_cv_mse
            best_lam = lam
    
    # Final fit on all data with best lam
    w_final = _fista_lasso(V_norm, y_norm, best_lam)
    
    # Convert back to original scale
    # y_norm = V_norm @ w_final + (y - y_mean)/y_std
    # y = y_mean + y_std * (V_norm @ w_final)
    # y = y_mean + y_std * ((V - V_mean)/V_std @ w_final)
    # y = y_mean + (V - V_mean) @ (w_final / V_std) * y_std ? No.
    # Let's derive:
    # y = y_mean + y_std * (V_norm @ w)
    # V_norm = (V - V_mean) / V_std  (element-wise division by column std)
    # Let c be the coefficients in original space.
    # We want V @ c = y_mean + y_std * (V - V_mean) @ (w / V_std)
    # Note: (V - V_mean) @ (w / V_std) is not simply V @ c.
    # Actually, V_norm[:, j] = (V[:, j] - V_mean[j]) / V_std[j]
    # So V_norm @ w = sum_j w_j * (V[:, j] - V_mean[j]) / V_std[j]
    # = sum_j (w_j / V_std[j]) * V[:, j] - sum_j (w_j / V_std[j]) * V_mean[j]
    # Let c_j = w_j / V_std[j].
    # Then V_norm @ w = V @ c - V_mean @ c
    # So y = y_mean + y_std * (V @ c - V_mean @ c)
    # y = y_mean - y_std * V_mean @ c + y_std * V @ c
    # This is an affine transformation.
    # The prompt asks for coefficients c_i such that sum(c_i * x**i) approximates y.
    # This implies y = V @ c_orig.
    # Our model is y = V @ (y_std * c) + (y_mean - y_std * V_mean @ c)
    # So c_orig = y_std * c
    # And the intercept is shifted. But the prompt asks for a polynomial sum(c_i * x**i).
    # This form includes the constant term c_0.
    # So we can just return c_orig = y_std * c?
    # Let's check:
    # If we return c_orig, then V @ c_orig = y_std * V @ c.
    # But our fit is y = y_std * V @ c + bias.
    # So V @ c_orig != y.
    # We need to absorb the bias into c_orig?
    # No, c_orig is a vector of 6 coefficients.
    # V @ c_orig = sum c_orig_i x^i.
    # Our prediction is P = y_mean + y_std * (V_norm @ w).
    # We want V @ c_orig = P.
    # This is a linear system for c_orig? No, P is a vector.
    # We want to find c_orig such that V @ c_orig is the best approximation to P?
    # Actually, P is already the prediction.
    # We can just compute c_orig by projecting P onto the polynomial space?
    # But P is already in the polynomial space (it's a linear combination of columns of V_norm, which are linear combinations of columns of V).
    # Specifically, P = V @ (y_std * c) + (y_mean - y_std * V_mean @ c) * 1.
    # The term (y_mean - y_std * V_mean @ c) is a constant.
    # So P = V @ (y_std * c) + constant.
    # The constant can be added to the first coefficient (degree 0) of c_orig.
    # c_orig[0] += constant
    # c_orig[i] = y_std * c[i] for i > 0.
    
    c = w_final / V_std
    bias = y_mean - y_std * np.dot(V_mean, c)
    
    c_orig = np.zeros(6)
    c_orig[0] = c[0] * y_std + bias
    c_orig[1:] = c[1:] * y_std
    
    # Final check
    if not np.all(np.isfinite(c_orig)):
        # Fallback to OLS
        try:
            c_ols = np.linalg.lstsq(V, y, rcond=None)[0]
            c_orig = c_ols
        except Exception:
            c_orig = np.zeros(6)
            
    return c_orig
# EVOLVE-BLOCK-END
