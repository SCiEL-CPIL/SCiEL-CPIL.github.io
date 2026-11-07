import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 310.0  # Safety margin from 330s limit

    N, D = X_train.shape

    # Normalize inputs to [0, 1] for numerical stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges

    # Center y
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean

    # Split for validation (20% holdout)
    n_val = max(10, int(N * 0.2))
    val_idx = np.arange(N - n_val, N)
    train_idx = np.arange(0, N - n_val)

    X_tr = X_norm[train_idx]
    y_tr = y_centered[train_idx]
    X_val = X_norm[val_idx]
    y_val = y_centered[val_idx]

    n_tr = len(y_tr)

    # --- Model 1: Polynomial Kernel Ridge Regression ---

    def poly_kernel(X1, X2, degree, coef0=1.0, gamma=1.0):
        """Polynomial kernel: (gamma * X1 . X2 + coef0)^degree"""
        K = (gamma * (X1 @ X2.T) + coef0) ** degree
        return K

    def fit_poly_ridge(X_tr, y_tr, X_val, y_val, degree, coef0, gamma, alpha):
        """Fit polynomial kernel ridge regression and return val MSE."""
        K_tr = poly_kernel(X_tr, X_tr, degree, coef0, gamma)
        K_val_tr = poly_kernel(X_val, X_tr, degree, coef0, gamma)

        A = K_tr + alpha * np.eye(n_tr)
        try:
            w = solve(A, y_tr, assume_a='pos')
            y_pred = K_val_tr @ w
            mse = np.mean((y_pred - y_val) ** 2)
            return mse, w, K_val_tr
        except:
            return np.inf, None, None

    # --- Model 2: RBF Kernel with Optimized Hyperparameters ---

    def rbf_kernel(X1, X2, length_scale, signal_var):
        """Compute RBF kernel with given hyperparameters."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K = signal_var * np.exp(-sq_dists / (2 * length_scale**2))
        return K

    def neg_log_likelihood(params):
        """Negative log marginal likelihood for GP hyperparameters."""
        length_scale = np.exp(params[0])  # ensure positive
        signal_var = np.exp(params[1])    # ensure positive
        noise_var = np.exp(params[2])     # ensure positive

        K = rbf_kernel(X_tr, X_tr, length_scale, signal_var) + noise_var * np.eye(n_tr)

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            nll = 0.5 * y_tr @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n_tr * np.log(2 * np.pi)
            return nll
        except:
            return 1e10

    # --- Model 3: Matern-5/2 Gaussian Process ---

    def matern52_kernel(X1, X2, length_scale, signal_var):
        """Matern-5/2 kernel."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        sqrt3 = np.sqrt(3.0)
        K = signal_var * (1 + sqrt3 * dists + dists**2) * np.exp(-sqrt3 * dists)
        return K

    def fit_matern52_gp(X_tr, y_tr, X_val, y_val, length_scale, signal_var, noise_var):
        """Fit Matern-5/2 GP and return val MSE."""
        K_tr = matern52_kernel(X_tr, X_tr, length_scale, signal_var) + noise_var * np.eye(n_tr)
        K_val_tr = matern52_kernel(X_val, X_tr, length_scale, signal_var)

        try:
            L = np.linalg.cholesky(K_tr)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            y_pred = K_val_tr @ alpha
            mse = np.mean((y_pred - y_val) ** 2)
            return mse, alpha, L
        except:
            return np.inf, None, None

    # --- Search for best model ---

    best_mse = np.inf
    best_model_type = None
    best_params = None

    # Search space for polynomial kernel ridge
    poly_configs = [
        (2, 1.0, 1.0, 1e-3),
        (2, 1.0, 1.0, 1e-2),
        (2, 1.0, 1.0, 1e-1),
        (3, 1.0, 1.0, 1e-3),
        (3, 1.0, 1.0, 1e-2),
        (3, 1.0, 1.0, 1e-1),
        (4, 1.0, 1.0, 1e-2),
        (4, 1.0, 1.0, 1e-1),
        (2, 0.5, 1.0, 1e-2),
        (3, 0.5, 1.0, 1e-2),
        (2, 1.0, 0.5, 1e-2),
        (3, 1.0, 0.5, 1e-2),
    ]

    for degree, coef0, gamma, alpha in poly_configs:
        if time.time() > deadline - 80:
            break
        mse, w, K_val_tr = fit_poly_ridge(X_tr, y_tr, X_val, y_val, degree, coef0, gamma, alpha)
        if mse < best_mse:
            best_mse = mse
            best_model_type = 'poly'
            best_params = (degree, coef0, gamma, alpha, w)

    # --- Optimize RBF GP hyperparameters ---
    best_rbf_params = None
    best_rbf_nll = np.inf

    # Coarse grid search for RBF
    length_scale_grid = np.logspace(-1, 1, 10)  # 0.1 to 10
    signal_var_grid = np.logspace(-1, 2, 8)     # 0.1 to 100
    noise_var_grid = np.logspace(-4, -1, 5)     # 0.0001 to 0.1

    for ls in length_scale_grid:
        if time.time() > deadline - 60:
            break
        for sv in signal_var_grid:
            for nv in noise_var_grid:
                params = np.array([np.log(ls), np.log(sv), np.log(nv)])
                nll = neg_log_likelihood(params)
                if nll < best_rbf_nll:
                    best_rbf_nll = nll
                    best_rbf_params = params

    # Refine with local optimization
    if best_rbf_params is not None:
        try:
            result = minimize(neg_log_likelihood, best_rbf_params, method='L-BFGS-B',
                             bounds=[(-5, 5), (-2, 5), (-8, -1)], maxiter=50)
            if result.fun < best_rbf_nll:
                best_rbf_nll = result.fun
                best_rbf_params = result.x
        except:
            pass

    if best_rbf_params is not None:
        ls_rbf = np.exp(best_rbf_params[0])
        sv_rbf = np.exp(best_rbf_params[1])
        nv_rbf = np.exp(best_rbf_params[2])

        K_tr_rbf = rbf_kernel(X_tr, X_tr, ls_rbf, sv_rbf) + nv_rbf * np.eye(n_tr)
        K_val_tr_rbf = rbf_kernel(X_val, X_tr, ls_rbf, sv_rbf)

        try:
            w_rbf = solve(K_tr_rbf, y_tr, assume_a='pos')
            y_pred_rbf = K_val_tr_rbf @ w_rbf
            mse_rbf = np.mean((y_pred_rbf - y_val) ** 2)

            if mse_rbf < best_mse:
                best_mse = mse_rbf
                best_model_type = 'rbf_gp'
                best_params = (ls_rbf, sv_rbf, nv_rbf, w_rbf)
        except:
            pass

    # Search space for Matern-5/2 GP
    matern_configs = [
        (0.1, 1.0, 1e-4),
        (0.1, 1.0, 1e-3),
        (0.5, 1.0, 1e-4),
        (0.5, 1.0, 1e-3),
        (1.0, 1.0, 1e-4),
        (1.0, 1.0, 1e-3),
        (0.5, 0.5, 1e-3),
        (1.0, 0.5, 1e-3),
        (0.5, 2.0, 1e-3),
        (1.0, 2.0, 1e-3),
    ]

    for ls, sv, nv in matern_configs:
        if time.time() > deadline - 40:
            break
        mse, alpha, L = fit_matern52_gp(X_tr, y_tr, X_val, y_val, ls, sv, nv)
        if mse < best_mse:
            best_mse = mse
            best_model_type = 'matern'
            best_params = (ls, sv, nv, alpha)

    # If no model was selected (shouldn't happen), use a simple fallback
    if best_model_type is None:
        best_model_type = 'poly'
        best_params = (2, 1.0, 1.0, 1e-2, None)

    # --- Refit on all training data ---

    if best_model_type == 'poly':
        degree, coef0, gamma, alpha, _ = best_params
        K_full = poly_kernel(X_norm, X_norm, degree, coef0, gamma)
        A_full = K_full + alpha * np.eye(N)
        try:
            w_full = solve(A_full, y_centered, assume_a='pos')
        except:
            w_full = np.zeros(N)

        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = poly_kernel(X_test_norm, X_norm, degree, coef0, gamma)
            return K_test @ w_full + y_mean

    elif best_model_type == 'rbf_gp':
        ls_rbf, sv_rbf, nv_rbf, _ = best_params
        K_full = rbf_kernel(X_norm, X_norm, ls_rbf, sv_rbf) + nv_rbf * np.eye(N)
        try:
            w_full = solve(K_full, y_centered, assume_a='pos')
        except:
            w_full = np.zeros(N)

        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = rbf_kernel(X_test_norm, X_norm, ls_rbf, sv_rbf)
            return K_test @ w_full + y_mean

    elif best_model_type == 'matern':
        ls, sv, nv, _ = best_params
        K_full = matern52_kernel(X_norm, X_norm, ls, sv) + nv * np.eye(N)
        try:
            L_full = np.linalg.cholesky(K_full)
            alpha_full = np.linalg.solve(L_full.T, np.linalg.solve(L_full, y_centered))
        except:
            alpha_full = np.zeros(N)

        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = matern52_kernel(X_test_norm, X_norm, ls, sv)
            return K_test @ alpha_full + y_mean

    else:
        # Fallback: simple linear regression
        X_feat = np.column_stack([np.ones(N), X_norm])
        A = X_feat.T @ X_feat + 1e-3 * np.eye(X_feat.shape[1])
        try:
            w_full = solve(A, X_feat.T @ y_centered)
        except:
            w_full = np.zeros(X_feat.shape[1])

        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_feat_test = np.column_stack([np.ones(X.shape[0]), X_test_norm])
            return X_feat_test @ w_full + y_mean

    return predict
