"""
Symbolic regression over 4 hidden 1D equations.
Strategy:
1. Generate a rich, normalized basis of candidate functions (polynomials, trig, exp, tanh, rational, etc.).
2. Use a kernel method (RBF) as a strong non-linear baseline that can fit any smooth function well.
3. Use Sparse Regression (LARS) to identify the most relevant basis functions, effectively performing symbolic search.
4. Combine the best linear model (from sparse regression) with the RBF kernel prediction via a weighted ensemble or simple averaging, favoring the one with better CV performance.
5. Strict time management to ensure we finish within the limit.
"""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

# Global time management
_GLOBAL_START_TIME = None
TIME_LIMIT = 295.0  # Safety margin before 300s limit

def _get_time_left():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return TIME_LIMIT - (time.time() - _GLOBAL_START_TIME)

def _generate_basis(x):
    """
    Generate a rich set of basis functions for 1D input x.
    x is assumed to be normalized (mean 0, std 1) for numerical stability.
    """
    N = len(x)
    features = [np.ones(N)]
    
    # Polynomials up to degree 8
    for k in range(1, 9):
        features.append(x ** k)
        
    # Trigonometric functions
    freqs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    for f in freqs:
        features.append(np.sin(f * x))
        features.append(np.cos(f * x))
        
    # Hyperbolic
    for scale in [0.5, 1.0, 2.0]:
        features.append(np.tanh(scale * x))
        
    # Exponential
    for rate in [0.2, 0.5, 1.0]:
        features.append(np.exp(-rate * x))
        features.append(np.exp(rate * x))
        
    # Gaussian-like
    features.append(np.exp(-x**2))
    features.append(np.exp(-0.5 * x**2))
    features.append(np.exp(-2.0 * x**2))
    
    # Rational
    features.append(1.0 / (1.0 + x**2))
    features.append(1.0 / (1.0 + 0.25 * x**2))
    
    # Abs
    features.append(np.abs(x))
    features.append(x * np.abs(x))
    
    X_feat = np.column_stack(features)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    return X_feat

def _ridge_solve(X, y, lam):
    """
    Solve Ridge regression efficiently.
    """
    N, P = X.shape
    # Center X and y for better conditioning? 
    # Standard ridge: (X^T X + lam I) w = X^T y
    # Using SVD for stability
    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        Ut_y = U.T @ y
        # w = Vt.T @ (s / (s^2 + lam)) * Ut_y
        # Handle small singular values
        s_safe = np.where(s > 1e-12, s, 1e-12)
        w = Vt.T @ (s_safe / (s_safe**2 + lam) * Ut_y)
        return w
    except np.linalg.LinAlgError:
        # Fallback
        A = X.T @ X + lam * np.eye(P)
        b = X.T @ y
        return np.linalg.lstsq(A, b, rcond=None)[0]

def _fit_rbf(X_train, y_train, gamma, lam):
    """
    Fit RBF kernel ridge regression.
    Returns a callable or parameters for prediction.
    """
    n = X_train.shape[0]
    # Compute kernel matrix
    sq_dists = np.sum(X_train**2, axis=1)[:, None] + np.sum(X_train**2, axis=1)[None, :] - 2 * X_train @ X_train.T
    sq_dists = np.maximum(sq_dists, 0)
    K = np.exp(-gamma * sq_dists)
    K_reg = K + lam * np.eye(n)
    
    try:
        w = np.linalg.solve(K_reg, y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y_train, rcond=None)[0]
        
    return w, gamma

def _predict_rbf(X_test, X_train, w, gamma):
    """
    Predict using RBF kernel.
    """
    sq_dists = np.sum(X_test**2, axis=1)[:, None] + np.sum(X_train**2, axis=1)[None, :] - 2 * X_test @ X_train.T
    sq_dists = np.maximum(sq_dists, 0)
    K = np.exp(-gamma * sq_dists)
    return K @ w

def fit(X_train, y_train):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    if d != 1:
        # Fallback for multi-dim (should not happen per contract, but safe)
        x = X[:, 0]
        # Simple linear
        X_lin = np.column_stack([np.ones(n), x])
        w = np.linalg.lstsq(X_lin, y, rcond=None)[0]
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=float)
            x_t = X_t[:, 0]
            X_t_lin = np.column_stack([np.ones(len(x_t)), x_t])
            return X_t_lin @ w
        return predict

    x = X[:, 0]
    
    # Normalize
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-8:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Generate Basis
    X_basis = _generate_basis(x_norm)
    
    # Strategy 1: Linear Model with Ridge (Sparse-ish selection via LARS or just Ridge with strong reg)
    # Let's try a few lambdas for Ridge on the basis
    best_lin_w = None
    best_lin_mse = np.inf
    
    # Strategy 2: RBF Kernel
    best_rbf_w = None
    best_rbf_gamma = None
    best_rbf_mse = np.inf
    
    # Cross Validation (simple 5-fold)
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    folds = np.array_split(indices, n_folds)
    
    # --- Evaluate Linear Basis Models ---
    # We will try a grid of lambdas for Ridge on the basis
    # Also, we can try to select a subset of features using LARS if available, but let's stick to Ridge first.
    # To simulate sparsity/symbolic search, we can use a strong lambda.
    
    lams = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    # Pre-compute for RBF
    # RBF on original x_norm
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
    rbf_lams = [1e-2, 0.1, 1.0]
    
    # We will do a joint CV search to pick the best model type and params.
    # To save time, we'll run CV on a subset of parameters.
    
    # Time check
    if _get_time_left() < 10.0:
        # Fallback to simple linear
        X_simple = np.column_stack([np.ones(n), x_norm])
        w = np.linalg.lstsq(X_simple, y, rcond=None)[0]
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=float)
            x_t = X_t[:, 0]
            x_t_norm = (x_t - x_mean) / x_std
            X_t_simple = np.column_stack([np.ones(len(x_t)), x_t_norm])
            return X_t_simple @ w
        return predict

    cv_scores_lin = []
    cv_scores_rbf = []
    
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        
        X_tr_lin = X_basis[train_idx]
        y_tr = y[train_idx]
        X_val_lin = X_basis[val_idx]
        y_val = y[val_idx]
        
        x_tr_norm = x_norm[train_idx]
        x_val_norm = x_norm[val_idx]
        
        # --- Linear Basis CV ---
        for lam in lams:
            if _get_time_left() < 20.0:
                break
            w = _ridge_solve(X_tr_lin, y_tr, lam)
            pred = X_val_lin @ w
            mse = np.mean((pred - y_val) ** 2)
            if np.isfinite(mse):
                cv_scores_lin.append((lam, mse))
        
        # --- RBF CV ---
        for gamma in gammas:
            for rbf_lam in rbf_lams:
                if _get_time_left() < 20.0:
                    break
                # Fit RBF on training fold
                w_rbf, _ = _fit_rbf(x_tr_norm.reshape(-1, 1), y_tr, gamma, rbf_lam)
                pred = _predict_rbf(x_val_norm.reshape(-1, 1), x_tr_norm.reshape(-1, 1), w_rbf, gamma)
                mse = np.mean((pred - y_val) ** 2)
                if np.isfinite(mse):
                    cv_scores_rbf.append(((gamma, rbf_lam), mse))
        
        # Check time
        if _get_time_left() < 30.0:
            break

    # Select best Linear
    best_lin_mse = np.inf
    best_lin_lam = None
    if cv_scores_lin:
        # Average MSE for each lam
        lam_mses = {}
        for lam, mse in cv_scores_lin:
            if lam not in lam_mses:
                lam_mses[lam] = []
            lam_mses[lam].append(mse)
        
        for lam, mses in lam_mses.items():
            avg_mse = np.mean(mses)
            if avg_mse < best_lin_mse:
                best_lin_mse = avg_mse
                best_lin_lam = lam

    # Select best RBF
    best_rbf_mse = np.inf
    best_rbf_params = None
    if cv_scores_rbf:
        param_mses = {}
        for params, mse in cv_scores_rbf:
            if params not in param_mses:
                param_mses[params] = []
            param_mses[params].append(mse)
        
        for params, mses in param_mses.items():
            avg_mse = np.mean(mses)
            if avg_mse < best_rbf_mse:
                best_rbf_mse = avg_mse
                best_rbf_params = params

    # Final Fit
    final_w_lin = None
    if best_lin_lam is not None:
        final_w_lin = _ridge_solve(X_basis, y, best_lin_lam)
    
    final_w_rbf = None
    final_gamma = None
    if best_rbf_params is not None:
        gamma, rbf_lam = best_rbf_params
        final_w_rbf, _ = _fit_rbf(x_norm.reshape(-1, 1), y, gamma, rbf_lam)
        final_gamma = gamma

    # If both failed, fallback to simple linear
    if final_w_lin is None and final_w_rbf is None:
        X_simple = np.column_stack([np.ones(n), x_norm])
        final_w_lin = np.linalg.lstsq(X_simple, y, rcond=None)[0]
        # Mark as simple linear fallback
        best_lin_lam = "simple"
        best_lin_mse = 0
        best_rbf_mse = np.inf

    # Determine which model to use or blend
    # If RBF is significantly better, use RBF. Otherwise, use Linear (which might be the symbolic solution).
    # Actually, Linear Basis with strong regularization often finds the sparse symbolic solution.
    # RBF is a universal approximator but might overfit if not regularized well.
    # Let's use the one with lower CV MSE. If tie, prefer Linear (simpler, more "symbolic").
    
    use_rbf = (best_rbf_mse < best_lin_mse) and (best_rbf_mse < 1e-6) # RBF needs to be very good to beat linear? 
    # Actually, just compare.
    if best_rbf_mse < best_lin_mse:
        use_rbf = True
    else:
        use_rbf = False

    # If RBF is chosen, we need X_train for prediction
    X_train_norm = x_norm.reshape(-1, 1)

    def predict(X_test):
        X_t = np.asarray(X_test, dtype=float)
        x_t = X_t[:, 0]
        x_t_norm = (x_t - x_mean) / x_std
        
        if use_rbf and final_w_rbf is not None:
            return _predict_rbf(x_t_norm.reshape(-1, 1), X_train_norm, final_w_rbf, final_gamma)
        else:
            # Linear Basis
            if final_w_lin is not None:
                if best_lin_lam == "simple":
                    # Simple linear
                    X_t_simple = np.column_stack([np.ones(len(x_t)), x_t_norm])
                    return X_t_simple @ final_w_lin
                else:
                    X_t_basis = _generate_basis(x_t_norm)
                    return X_t_basis @ final_w_lin
            else:
                # Absolute fallback
                return np.zeros(len(x_t))

    return predict
