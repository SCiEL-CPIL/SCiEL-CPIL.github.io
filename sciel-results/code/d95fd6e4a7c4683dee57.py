"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _gp_neg_log_likelihood(params, x, y, L, alpha, y_mean, sigma_f, sigma_n):
    """Negative log marginal likelihood for GP with RBF kernel."""
    # params: [log_sigma_f, log_length_scale, log_sigma_n]
    log_sigma_f, log_l, log_sigma_n = params
    sigma_f_val = np.exp(log_sigma_f)
    l_val = np.exp(log_l)
    sigma_n_val = np.exp(log_sigma_n)
    
    # Compute K(x, x) + sigma_n^2 I
    # We precomputed L and alpha for the initial params, but here we need to recompute
    # Actually, it's better to recompute everything inside the objective for correctness
    # during optimization, or pass the necessary data.
    # For efficiency, we can compute the kernel matrix once per iteration.
    
    x = x  # already centered or not? We centered y, so x is original.
    # RBF kernel: exp(-0.5 * (x_i - x_j)^2 / l^2)
    # This can be memory intensive for N=200, but 200x200 is small.
    
    # Compute squared distances
    x_col = x.reshape(-1, 1)
    sq_dist = (x_col - x_col.T) ** 2
    K = sigma_f_val ** 2 * np.exp(-0.5 * sq_dist / (l_val ** 2))
    K += sigma_n_val ** 2 * np.eye(len(x))
    
    # Cholesky decomposition
    try:
        L_new = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10  # Return large value if singular
    
    # alpha = K^{-1} y
    try:
        alpha_new = np.linalg.solve(L_new, np.linalg.solve(L_new.T, y - y_mean))
    except np.linalg.LinAlgError:
        return 1e10
    
    # Negative log marginal likelihood:
    # -0.5 * y^T K^{-1} y + 0.5 * log(det(K)) + 0.5 * N * log(2*pi)
    # log(det(K)) = 2 * sum(log(diag(L)))
    N = len(x)
    neg_ll = -0.5 * (y - y_mean) @ alpha_new + np.sum(np.log(np.diag(L_new))) + 0.5 * N * np.log(2 * np.pi)
    
    return neg_ll


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    d = X.shape[1]
    n = X.shape[0]

    # If d > 1, we could try multivariate, but the task says 1 input for all 4 equations
    # So we assume d=1. If d!=1, fallback to poly.
    if d != 1:
        # Fallback for safety, though unlikely
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)
        return lambda X: features(X) @ w

    x = X[:, 0]
    y = y.flatten()

    # Sort by x for consistent processing (though GP doesn't strictly require it)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Center y
    y_mean = np.mean(y_sorted)
    y_centered = y_sorted - y_mean

    # Initialize hyperparameters
    # sigma_f: signal variance
    # l: length scale
    # sigma_n: noise variance (should be small for noiseless data)
    
    # Heuristic initialization
    sigma_f_init = np.std(y_centered)
    if sigma_f_init < 1e-6:
        sigma_f_init = 1.0
    l_init = np.std(x_sorted)
    if l_init < 1e-6:
        l_init = 1.0
    sigma_n_init = 1e-4  # Small noise for numerical stability

    # Log-space parameters for optimization
    params_init = np.array([
        np.log(sigma_f_init),
        np.log(l_init),
        np.log(sigma_n_init)
    ])

    # Precompute x column for distance calculation
    x_col = x_sorted.reshape(-1, 1)
    sq_dist = (x_col - x_col.T) ** 2

    def objective(params):
        if time.time() > deadline:
            return 1e10
        log_sigma_f, log_l, log_sigma_n = params
        sigma_f_val = np.exp(log_sigma_f)
        l_val = np.exp(log_l)
        sigma_n_val = np.exp(log_sigma_n)
        
        K = sigma_f_val ** 2 * np.exp(-0.5 * sq_dist / (l_val ** 2))
        K += sigma_n_val ** 2 * np.eye(len(x_sorted))
        
        try:
            L_new = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        
        try:
            alpha_new = np.linalg.solve(L_new, np.linalg.solve(L_new.T, y_centered))
        except np.linalg.LinAlgError:
            return 1e10
        
        N = len(x_sorted)
        neg_ll = -0.5 * y_centered @ alpha_new + np.sum(np.log(np.diag(L_new))) + 0.5 * N * np.log(2 * np.pi)
        
        return neg_ll

    # Optimize hyperparameters
    best_params = params_init
    best_neg_ll = objective(params_init)
    
    try:
        res = minimize(objective, params_init, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success and res.fun < best_neg_ll:
            best_params = res.x
            best_neg_ll = res.fun
    except Exception:
        pass

    # Final GP model with best parameters
    log_sigma_f, log_l, log_sigma_n = best_params
    sigma_f = np.exp(log_sigma_f)
    l = np.exp(log_l)
    sigma_n = np.exp(log_sigma_n)

    K = sigma_f ** 2 * np.exp(-0.5 * sq_dist / (l ** 2))
    K += sigma_n ** 2 * np.eye(len(x_sorted))
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # Fallback to identity if Cholesky fails
        L = np.eye(len(x_sorted))
        sigma_f = 1.0
        l = 1.0
        sigma_n = 1.0
        K = np.eye(len(x_sorted))
        L = np.eye(len(x_sorted))

    try:
        alpha = np.linalg.solve(L, np.linalg.solve(L.T, y_centered))
    except np.linalg.LinAlgError:
        alpha = np.zeros(len(x_sorted))

    # Define predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        x_test = X_test[:, 0]
        
        # Compute K(x_test, x_train)
        x_test_col = x_test.reshape(-1, 1)
        sq_dist_test = (x_test_col - x_col.T) ** 2
        K_test = sigma_f ** 2 * np.exp(-0.5 * sq_dist_test / (l ** 2))
        
        # Prediction mean: y_mean + K_test @ alpha
        y_pred = y_mean + K_test @ alpha
        
        # Ensure finite values
        y_pred = np.where(np.isfinite(y_pred), y_pred, y_mean)
        
        return y_pred

    return predict

# EVOLVE-BLOCK-END
