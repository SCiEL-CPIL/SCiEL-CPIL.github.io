"""Symbolic regression over 10 hidden equations: multi-stage oracle selection."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def _loocv_ridge(X, y, kernel_fn, alpha_grid):
    """Compute LOO-CV MSE for kernel ridge regression with given kernel and alpha grid."""
    n = len(y)
    K = kernel_fn(X, X)
    best_mse = np.inf
    best_alpha = None
    for alpha in alpha_grid:
        K_reg = K + alpha * np.eye(n)
        try:
            # Efficient LOO for KRR: yhat_loo_i = y_i - K_ii * ((K+alpha*I)^{-1})_{ii}
            # Compute diagonal of inverse without full inverse
            # Use Cholesky for stability
            L = np.linalg.cholesky(K_reg)
            # Solve L v = I to get v = L^{-1}, then (K_reg)^{-1} = v^T v
            # But we only need diagonal: diag((K_reg)^{-1}) = diag((L L^T)^{-1}) = sum_j (L^{-1}_{ij})^2
            # Actually: (L L^T)^{-1} = L^{-T} L^{-1}, so diag = sum_j (L^{-1}_{ji})^2 = sum_j (L^{-T}_{ij})^2
            # Let's just compute the inverse diagonal via solving
            # For each i: e_i^T (K_reg)^{-1} e_i = e_i^T v_i where v_i = (K_reg)^{-1} e_i
            # This is O(n^3) total if we do it smartly, but for n=200 it's fine to just invert
            inv_K = np.linalg.inv(K_reg)
            diag_inv = np.diag(inv_K)
            yhat_loo = y - np.diag(K) * diag_inv
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except np.linalg.LinAlgError:
            continue
    return best_mse, best_alpha, K


def _loocv_gp(X, y, length_scale_grid, variance_grid, noise_grid):
    """Compute LOO-CV MSE for Matern-5/2 GP with hyperparameter search."""
    n = len(y)
    d = X.shape[1]
    best_mse = np.inf
    best_params = None
    
    for ls in length_scale_grid:
        for var in variance_grid:
            for noise in noise_grid:
                # Compute squared distances
                X_sq = np.sum(X ** 2, axis=1)
                sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X @ X.T
                sq_dists = np.maximum(sq_dists, 0)
                dists = np.sqrt(sq_dists)
                
                # Matern-5/2 kernel
                r = dists / ls
                K_prior = var * (1 + r + r ** 2 / 3) * np.exp(-r)
                K_obs = K_prior + noise * np.eye(n)
                
                try:
                    L = np.linalg.cholesky(K_obs)
                    # Solve for alpha = K_obs^{-1} y
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    # LOO: yhat_i = y_i - K_prior_ii * (K_obs^{-1})_{ii}
                    # K_prior_ii = var for all i
                    # Need diag(K_obs^{-1})
                    inv_K_obs = np.linalg.inv(K_obs)
                    diag_inv = np.diag(inv_K_obs)
                    yhat_loo = y - var * diag_inv
                    mse = np.mean((y - yhat_loo) ** 2)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = (ls, var, noise)
                except (np.linalg.LinAlgError, ValueError):
                    continue
    
    return best_mse, best_params


def _loocv_linear_ridge(X, y, alpha_grid):
    """LOO-CV for linear ridge regression with intercept."""
    n, d = X.shape
    B = np.column_stack([np.ones(n), X])
    m = B.shape[1]
    best_mse = np.inf
    best_alpha = None
    
    for alpha in alpha_grid:
        A = B.T @ B + alpha * np.eye(m)
        b = B.T @ y
        try:
            w = np.linalg.solve(A, b)
            # Hat matrix H = B (B^T B + alpha I)^{-1} B^T
            # For LOO: yhat_loo_i = y_i - H_ii / (1 - H_ii) * (y_i - yhat_i)
            # But for ridge, the hat matrix is H = B (B^T B + alpha I)^{-1} B^T
            # and yhat = H y, so y_i - yhat_i = (1 - H_ii) y_i - sum_{j!=i} H_ij y_j
            # Actually the standard LOO formula for linear models with hat matrix H:
            # yhat_loo_i = (y_i - H_ii * yhat_i) / (1 - H_ii) ... no
            # yhat_loo_i = y_i - (y_i - yhat_i) / (1 - H_ii)
            # = y_i - (1 - H_ii)^{-1} * (y_i - yhat_i)
            # Wait: yhat_i = H_ii y_i + sum_{j!=i} H_ij y_j
            # yhat_loo_i = sum_{j!=i} H_ij y_j (refit without i)
            # The relationship: yhat_loo_i = yhat_i - H_ii (y_i - yhat_i) / (1 - H_ii)
            # = (yhat_i - H_ii y_i + H_ii yhat_i) / (1 - H_ii)
            # = (yhat_i (1 + H_ii) - H_ii y_i) / (1 - H_ii)
            # Hmm, let me use the simpler formula: yhat_loo_i = y_i - (y_i - yhat_i)/(1 - H_ii)
            yhat = B @ w
            H = B @ np.linalg.inv(B.T @ B + alpha * np.eye(m)) @ B.T
            h_ii = np.diag(H)
            # Avoid division by zero
            h_ii_safe = np.where(np.abs(1 - h_ii) < 1e-10, 1e-10, 1 - h_ii)
            yhat_loo = y - (y - yhat) / h_ii_safe
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except np.linalg.LinAlgError:
            continue
    
    return best_mse, best_alpha


def _loocv_poly_ridge(X, y, degree, alpha_grid):
    """LOO-CV for polynomial feature ridge regression."""
    n, d = X.shape
    
    def poly_features(X_in, deg):
        cols = [np.ones(len(X_in))]
        # Linear terms
        for j in range(X_in.shape[1]):
            cols.append(X_in[:, j])
        # Quadratic and higher
        for deg_cur in range(2, deg + 1):
            for i in range(X_in.shape[1]):
                for j in range(i, X_in.shape[1]):
                    term = np.ones(len(X_in))
                    # This is inefficient but d is small (1-3)
                    # For degree 2: x_i * x_j
                    # For degree 3: x_i*x_j*x_k, etc.
                    # Let me just do degree 2 for simplicity and efficiency
                    if deg_cur == 2:
                        term = X_in[:, i] * X_in[:, j]
                    else:
                        # Skip higher degrees for now, keep it simple
                        continue
                    cols.append(term)
        return np.column_stack(cols)
    
    B = poly_features(X, degree)
    m = B.shape[1]
    best_mse = np.inf
    best_alpha = None
    
    for alpha in alpha_grid:
        A = B.T @ B + alpha * np.eye(m)
        b = B.T @ y
        try:
            w = np.linalg.solve(A, b)
            yhat = B @ w
            H = B @ np.linalg.inv(B.T @ B + alpha * np.eye(m)) @ B.T
            h_ii = np.diag(H)
            h_ii_safe = np.where(np.abs(1 - h_ii) < 1e-10, 1e-10, 1 - h_ii)
            yhat_loo = y - (y - yhat) / h_ii_safe
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except np.linalg.LinAlgError:
            continue
    
    return best_mse, best_alpha


def fit(X_train, y_train):
    X = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize inputs for better conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    models = []
    cv_scores = []
    
    # Model 1: Linear Ridge Regression
    alpha_grid_lin = [0.001, 0.01, 0.1, 1.0, 10.0]
    mse1, alpha1 = _loocv_linear_ridge(X_norm, y, alpha_grid_lin)
    if mse1 < np.inf and alpha1 is not None:
        B = np.column_stack([np.ones(n), X_norm])
        A = B.T @ B + alpha1 * np.eye(B.shape[1])
        w = np.linalg.solve(A, B.T @ y)
        def pred_linear(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            B_test = np.column_stack([np.ones(len(X_test)), X_test_norm])
            return B_test @ w
        models.append(('linear_ridge', pred_linear))
        cv_scores.append(mse1)
    
    # Model 2: Polynomial (degree 2) Ridge Regression
    alpha_grid_poly = [0.001, 0.01, 0.1, 1.0, 10.0]
    mse2, alpha2 = _loocv_poly_ridge(X_norm, y, 2, alpha_grid_poly)
    if mse2 < np.inf and alpha2 is not None:
        def poly_features(X_in, deg=2):
            cols = [np.ones(len(X_in))]
            for j in range(X_in.shape[1]):
                cols.append(X_in[:, j])
            for i in range(X_in.shape[1]):
                for j in range(i, X_in.shape[1]):
                    cols.append(X_in[:, i] * X_in[:, j])
            return np.column_stack(cols)
        B = poly_features(X_norm)
        A = B.T @ B + alpha2 * np.eye(B.shape[1])
        w = np.linalg.solve(A, B.T @ y)
        def pred_poly(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            B_test = poly_features(X_test_norm)
            return B_test @ w
        models.append(('poly_ridge', pred_poly))
        cv_scores.append(mse2)
    
    # Model 3: Polynomial Kernel Ridge Regression (degree 2)
    def poly_kernel(X1, X2, degree=2):
        return (1 + X1 @ X2.T / d) ** degree
    
    alpha_grid_krr = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    mse3, alpha3, K3 = _loocv_ridge(X_norm, y, lambda a, b: poly_kernel(a, b, 2), alpha_grid_krr)
    if mse3 < np.inf and alpha3 is not None:
        K_reg = K3 + alpha3 * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def pred_krr(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            K_test = poly_kernel(X_test_norm, X_norm, 2)
            return K_test @ w
        models.append(('krr_poly2', pred_krr))
        cv_scores.append(mse3)
    
    # Model 4: Polynomial Kernel Ridge Regression (degree 3)
    mse4, alpha4, K4 = _loocv_ridge(X_norm, y, lambda a, b: poly_kernel(a, b, 3), alpha_grid_krr)
    if mse4 < np.inf and alpha4 is not None:
        K_reg = K4 + alpha4 * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def pred_krr3(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            K_test = poly_kernel(X_test_norm, X_norm, 3)
            return K_test @ w
        models.append(('krr_poly3', pred_krr3))
        cv_scores.append(mse4)
    
    # Model 5: Gaussian Process (Matern-5/2)
    length_scale_grid = [0.3, 0.5, 1.0, 2.0]
    variance_grid = [0.5, 1.0, 2.0]
    noise_grid = [0.01, 0.1, 1.0]
    mse5, gp_params = _loocv_gp(X_norm, y, length_scale_grid, variance_grid, noise_grid)
    if mse5 < np.inf and gp_params is not None:
        ls, var, noise = gp_params
        X_sq = np.sum(X_norm ** 2, axis=1)
        sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X_norm @ X_norm.T
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        r = dists / ls
        K_prior = var * (1 + r + r ** 2 / 3) * np.exp(-r)
        K_obs = K_prior + noise * np.eye(n)
        L = np.linalg.cholesky(K_obs)
        alpha_gp = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        def pred_gp(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            X_test_sq = np.sum(X_test_norm ** 2, axis=1)
            sq_dists_test = X_test_sq[:, None] + X_sq[None, :] - 2 * X_test_norm @ X_norm.T
            sq_dists_test = np.maximum(sq_dists_test, 0)
            dists_test = np.sqrt(sq_dists_test)
            r_test = dists_test / ls
            K_test = var * (1 + r_test + r_test ** 2 / 3) * np.exp(-r_test)
            return K_test @ alpha_gp
        models.append(('gp_matern52', pred_gp))
        cv_scores.append(mse5)
    
    # Select best model based on LOO CV score
    if not models:
        # Fallback: simple mean
        y_mean = np.mean(y)
        return lambda X: np.full(len(X), y_mean)
    
    best_idx = np.argmin(cv_scores)
    best_predict_fn = models[best_idx][1]
    
    def predict(X_test):
        X_test = np.array(X_test, dtype=np.float64)
        try:
            preds = best_predict_fn(X_test)
            # Ensure finite predictions
            preds = np.where(np.isfinite(preds), preds, np.mean(y))
            return preds
        except Exception:
            return np.full(len(X_test), np.mean(y))
    
    return predict
# EVOLVE-BLOCK-END
