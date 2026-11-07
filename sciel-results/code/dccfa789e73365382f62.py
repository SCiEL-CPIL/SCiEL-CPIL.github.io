# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import cKDTree
from collections import defaultdict

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 300.0  # seconds, safety margin from 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] range for better kernel scaling
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize y for numerical stability
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    def check_time():
        return (time.time() - start_time) > DEADLINE
    
    # Compute squared distances matrix for training points
    # Use efficient computation
    sq_norms = np.sum(X_norm ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X_norm @ X_norm.T
    np.maximum(sq_dists, 0, out=sq_dists)
    
    # Try multiple kernel configurations and pick the best via CV
    kernel_configs = []
    
    # RBF with various length scales
    for length_scale in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        kernel_configs.append(('rbf', length_scale, 1.0, 0.01))
    
    # RBF with different noise levels
    for length_scale in [0.3, 0.5, 0.7]:
        for noise in [0.001, 0.01, 0.1]:
            kernel_configs.append(('rbf', length_scale, noise, 0.01))
    
    # Polynomial kernels
    for degree in [2, 3, 4]:
        for gamma in [0.5, 1.0, 2.0]:
            kernel_configs.append(('poly', degree, gamma, 0.01))
    
    # Sigmoid kernels
    for gamma in [0.5, 1.0, 2.0]:
        kernel_configs.append(('sigmoid', 1, gamma, 0.01))
    
    # Rational quadratic
    for alpha in [0.5, 1.0, 2.0]:
        for length_scale in [0.3, 0.5, 0.7]:
            kernel_configs.append(('rational', length_scale, alpha, 0.01))
    
    def compute_kernel(X1, X2, kernel_type, param1, param2, noise):
        """Compute kernel matrix between X1 and X2."""
        if kernel_type == 'rbf':
            # X1: (n1, d), X2: (n2, d)
            sq_norms1 = np.sum(X1 ** 2, axis=1)
            sq_norms2 = np.sum(X2 ** 2, axis=1)
            sq_dists = sq_norms1[:, None] + sq_norms2[None, :] - 2 * X1 @ X2.T
            np.maximum(sq_dists, 0, out=sq_dists)
            K = np.exp(-sq_dists / (2 * param1 ** 2))
        elif kernel_type == 'poly':
            degree = param1
            gamma = param2
            K = (gamma * X1 @ X2.T + 1) ** degree
        elif kernel_type == 'sigmoid':
            gamma = param2
            K = np.tanh(gamma * X1 @ X2.T + 1)
        elif kernel_type == 'rational':
            length_scale = param1
            alpha = param2
            sq_norms1 = np.sum(X1 ** 2, axis=1)
            sq_norms2 = np.sum(X2 ** 2, axis=1)
            sq_dists = sq_norms1[:, None] + sq_norms2[None, :] - 2 * X1 @ X2.T
            np.maximum(sq_dists, 0, out=sq_dists)
            K = (1 + sq_dists / (2 * length_scale ** 2)) ** (-alpha)
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")
        return K
    
    # 5-fold cross-validation
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, n_folds)
    
    def cv_score(kernel_type, param1, param2, noise):
        """Compute mean CV MSE in normalized space."""
        if check_time():
            return np.inf
        total_mse = 0.0
        count = 0
        for fold in range(n_folds):
            test_idx = fold_indices[fold]
            train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != fold])
            
            X_tr = X_norm[train_idx]
            y_tr = y_norm[train_idx]
            X_te = X_norm[test_idx]
            y_te = y_norm[test_idx]
            
            K_train = compute_kernel(X_tr, X_tr, kernel_type, param1, param2, 0)
            K_train += noise * np.eye(len(X_tr))
            
            try:
                L = np.linalg.cholesky(K_train)
            except np.linalg.LinAlgError:
                # Add jitter and retry
                K_train += 1e-6 * np.eye(len(X_tr))
                try:
                    L = np.linalg.cholesky(K_train)
                except np.linalg.LinAlgError:
                    return np.inf
            
            # Solve L @ L.T @ alpha = y_tr
            # alpha = L.T^{-1} @ L^{-1} @ y_tr
            v = np.linalg.solve(L.T, y_tr)
            alpha = np.linalg.solve(L, v)
            
            K_test = compute_kernel(X_te, X_tr, kernel_type, param1, param2, 0)
            y_pred = K_test @ alpha
            
            mse = np.mean((y_pred - y_te) ** 2)
            total_mse += mse
            count += 1
            
            if check_time():
                return np.inf
        
        return total_mse / count
    
    # Evaluate all kernel configs via CV
    best_cv = np.inf
    best_config = ('rbf', 0.5, 0.01, 0.01)
    
    for kt, p1, p2, noise in kernel_configs:
        if check_time():
            break
        cv = cv_score(kt, p1, p2, noise)
        if cv < best_cv:
            best_cv = cv
            best_config = (kt, p1, p2, noise)
    
    # Try to optimize the best kernel's parameters
    if not check_time():
        best_kt, best_p1, best_p2, best_noise = best_config
        
        # Optimize length scale and noise for RBF
        if best_kt == 'rbf':
            def objective(params):
                if check_time():
                    return np.inf
                ls, noise = params
                ls = max(ls, 0.01)
                noise = max(noise, 1e-6)
                return cv_score('rbf', ls, 1.0, noise)
            
            try:
                result = minimize(objective, [best_p1, best_noise], 
                                  method='Nelder-Mead',
                                  options={'maxiter': 50, 'xatol': 0.01, 'fatol': 0.01})
                if result.fun < best_cv:
                    best_p1 = max(result.x[0], 0.01)
                    best_noise = max(result.x[1], 1e-6)
                    best_cv = result.fun
            except:
                pass
        
        # Optimize for polynomial
        elif best_kt == 'poly':
            def objective(params):
                if check_time():
                    return np.inf
                gamma, noise = params
                gamma = max(gamma, 0.01)
                noise = max(noise, 1e-6)
                return cv_score('poly', best_p1, gamma, noise)
            
            try:
                result = minimize(objective, [best_p2, best_noise],
                                  method='Nelder-Mead',
                                  options={'maxiter': 50, 'xatol': 0.01, 'fatol': 0.01})
                if result.fun < best_cv:
                    best_p2 = max(result.x[0], 0.01)
                    best_noise = max(result.x[1], 1e-6)
                    best_cv = result.fun
            except:
                pass
    
    # Fit the final GP with the best configuration
    best_kt, best_p1, best_p2, best_noise = best_config
    
    K_train = compute_kernel(X_norm, X_norm, best_kt, best_p1, best_p2, 0)
    K_train += best_noise * np.eye(n)
    
    # Ensure positive definiteness
    try:
        L = np.linalg.cholesky(K_train)
    except np.linalg.LinAlgError:
        K_train += 1e-4 * np.eye(n)
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            K_train += 1e-3 * np.eye(n)
            try:
                L = np.linalg.cholesky(K_train)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                alpha = np.linalg.pinv(K_train) @ y_norm
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    X_norm_pred = (X - X_min) / X_range
                    K_test = compute_kernel(X_norm_pred, X_norm, best_kt, best_p1, best_p2, 0)
                    y_pred_norm = K_test @ alpha
                    return y_pred_norm * y_std + y_mean
                return predict
    
    # Solve for alpha
    v = np.linalg.solve(L.T, y_norm)
    alpha = np.linalg.solve(L, v)
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm_pred = (X - X_min) / X_range
        K_test = compute_kernel(X_norm_pred, X_norm, best_kt, best_p1, best_p2, 0)
        y_pred_norm = K_test @ alpha
        y_pred = y_pred_norm * y_std + y_mean
        return y_pred
    
    return predict
# EVOLVE-BLOCK-END
