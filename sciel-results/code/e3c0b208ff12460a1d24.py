"""Symbolic regression using sparse GP with Random Fourier Features."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import warnings

def fit(X_train, y_train):
    t_start = time.time()
    DEADLINE = 305.0  # Safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    n, d = X.shape
    y_flat = y.flatten()

    # Standardize X and y
    mu_x = X.mean(axis=0)
    std_x = X.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    X_std = (X - mu_x) / std_x
    
    mu_y = y_flat.mean()
    std_y = y_flat.std()
    if std_y < 1e-8:
        std_y = 1.0
    y_std = (y_flat - mu_y) / std_y

    # Rich polynomial feature function (degree 3)
    def poly_features(X, degree=3):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        if degree >= 3:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)

    # Compute polynomial fallback as safety net
    X_poly = poly_features(X)
    w_poly, *_ = np.linalg.lstsq(X_poly, y_flat, rcond=None)
    
    def predict_poly(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        return poly_features(X_test) @ w_poly

    # Random Fourier Features for RBF kernel approximation
    # K(x, x') ≈ exp(-||x-x'||^2 / (2*sigma^2)) ≈ Z(x)^T Z(x')
    # where Z(x) = sqrt(2/M) * [cos(w_1^T x), sin(w_1^T x), ...]
    # w_i ~ N(0, I / sigma^2)
    def rff_features(X, M, sigma, seed=42):
        rng = np.random.RandomState(seed)
        # sigma is the length scale; w ~ N(0, I/sigma^2)
        W = rng.randn(M, X.shape[1]) / sigma
        B = rng.rand(M) * 2 * np.pi
        Z = np.sqrt(2.0 / M) * np.column_stack([
            np.cos(X @ W.T + B),
            np.sin(X @ W.T + B)
        ])
        return Z

    # Select inducing points via k-means
    def select_inducing_points(X_std, m, seed=42):
        rng = np.random.RandomState(seed)
        # Simple k-means implementation
        if m >= len(X_std):
            return X_std.copy()
        # Initialize with random points
        idx = rng.choice(len(X_std), m, replace=False)
        centers = X_std[idx].copy()
        for _ in range(10):
            # Assign points to nearest center
            dists = np.linalg.norm(X_std[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            # Update centers
            new_centers = centers.copy()
            for j in range(m):
                mask = assignments == j
                if np.sum(mask) > 0:
                    new_centers[j] = X_std[mask].mean(axis=0)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return centers

    # Sparse GP with RFF kernel using inducing points
    # We'll use a subset of training data as inducing points
    # and solve the sparse GP system
    
    def rbf_kernel(X1, X2, length_scale, variance):
        sq_dist = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0)
        K = variance * np.exp(-sq_dist / (2 * length_scale**2))
        return K

    # Sparse GP log marginal likelihood with inducing points
    def sparse_gp_lml(params):
        length_scale = np.exp(params[0])
        variance = np.exp(params[1])
        noise = np.exp(params[2])
        
        # Select inducing points
        m = min(30, n)
        Z = select_inducing_points(X_std, m, seed=42)
        
        K_ZZ = rbf_kernel(Z, Z, length_scale, variance)
        K_ZZ += 1e-6 * np.eye(m)
        
        K_ZX = rbf_kernel(Z, X_std, length_scale, variance)
        K_XX_diag = variance  # diagonal of K(X,X)
        
        try:
            L, info = cho_factor(K_ZZ, lower=True, check_finite=False)
            if info > 0:
                return -1e10
            
            # Compute alpha_Z = K_ZZ^{-1} K_ZX y
            K_ZX_y = K_ZX @ y_std
            alpha_Z = cho_solve((L, info), K_ZX_y, check_finite=False)
            
            # Approximate LML
            # Using the variational bound or the simplified form
            # E[y^T K_ZZ^{-1} K_ZX K_XZ K_ZZ^{-1} K_ZX y] - trace terms
            pred_var = np.sum((K_ZX @ alpha_Z)**2)
            trace_term = np.sum(np.diag(K_ZX @ cho_solve((L, info), K_ZX.T, check_finite=False)))
            
            lml = 0.5 * pred_var - 0.5 * trace_term - 0.5 * n * np.log(2 * np.pi * noise)
            return -lml
        except:
            return -1e10

    # Multi-start hyperparameter optimization for sparse GP
    best_lml = -1e10
    best_params = None
    
    # Try multiple initializations
    initializations = [
        np.array([0.0, 0.0, -6.0]),
        np.array([-0.5, 0.0, -5.0]),
        np.array([0.5, 0.5, -6.0]),
        np.array([1.0, 0.0, -4.0]),
        np.array([-1.0, 0.5, -7.0]),
    ]
    
    for init in initializations:
        if time.time() > DEADLINE - 40:
            break
        try:
            result = minimize(sparse_gp_lml, init, method='L-BFGS-B', 
                              options={'maxiter': 60, 'ftol': 1e-6, 'gtol': 1e-4})
            if result.fun < best_lml:
                best_lml = result.fun
                best_params = result.x.copy()
        except:
            pass

    # Check if sparse GP is worth using
    use_sparse_gp = (best_params is not None and best_lml > -1e9)
    
    if not use_sparse_gp or time.time() > DEADLINE - 25:
        return predict_poly

    # Extract best hyperparameters
    best_length_scale = np.exp(best_params[0])
    best_variance = np.exp(best_params[1])
    best_noise = np.exp(best_params[2])

    # Select inducing points
    m = min(30, n)
    Z = select_inducing_points(X_std, m, seed=42)
    
    # Precompute Cholesky factor for inducing points
    K_ZZ = rbf_kernel(Z, Z, best_length_scale, best_variance)
    K_ZZ += 1e-6 * np.eye(m)
    
    try:
        L, info = cho_factor(K_ZZ, lower=True, check_finite=False)
        if info > 0:
            raise ValueError("Cholesky factorization failed")
    except:
        return predict_poly

    # Precompute K_ZX and solve for alpha
    K_ZX = rbf_kernel(Z, X_std, best_length_scale, best_variance)
    K_ZX_y = K_ZX @ y_std
    alpha_Z = cho_solve((L, info), K_ZX_y, check_finite=False)

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        X_test_std = (X_test - mu_x) / std_x
        K_test_Z = rbf_kernel(X_test_std, Z, best_length_scale, best_variance)
        y_pred_std = K_test_Z.dot(alpha_Z)
        y_pred = y_pred_std * std_y + mu_y
        y_pred = np.where(np.isfinite(y_pred), y_pred, mu_y)
        return y_pred

    return predict
# EVOLVE-BLOCK-END
