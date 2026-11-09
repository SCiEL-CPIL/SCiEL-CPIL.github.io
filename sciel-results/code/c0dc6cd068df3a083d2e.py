"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 285.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Composite kernel: RBF + polynomial + linear
    def composite_kernel(X1, X2, ls, deg, gamma_poly):
        """
        X1: (n1, d), X2: (n2, d)
        ls: length scale for RBF
        deg: polynomial degree
        gamma_poly: gamma for polynomial kernel
        """
        # RBF kernel
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        np.maximum(sq_dists, 0, out=sq_dists)
        K_rbf = np.exp(-sq_dists / (2 * ls**2))
        
        # Polynomial kernel
        K_poly = (gamma_poly * (X1 @ X2.T) + 1) ** deg
        
        # Linear kernel
        K_lin = X1 @ X2.T
        
        # Composite: weighted sum
        # Use fixed weights: RBF gets 1.0, poly gets 1.0, linear gets 1.0
        return K_rbf + K_poly + K_lin
    
    # Grid search over hyperparameters
    # Length scales for RBF
    ls_candidates = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
    # Polynomial degrees
    deg_candidates = [2, 3, 4]
    # Gamma for polynomial
    gamma_candidates = [0.5, 1.0, 2.0]
    # Regularization
    alpha_candidates = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    best_score = np.inf
    best_params = None
    best_weights = None
    best_K_train = None
    
    # Use leave-one-out or simple train error as proxy
    # For speed, we'll use training error with regularization
    
    for ls in ls_candidates:
        if time.time() > deadline - 5:
            break
        for deg in deg_candidates:
            if time.time() > deadline - 5:
                break
            for gamma in gamma_candidates:
                if time.time() > deadline - 5:
                    break
                for alpha in alpha_candidates:
                    try:
                        K = composite_kernel(X_norm, X_norm, ls, deg, gamma)
                        # Ridge regression: solve (K + alpha*I) w = y
                        w = np.linalg.solve(K + alpha * np.eye(n), y)
                        pred = K @ w
                        mse = np.mean((pred - y) ** 2)
                        
                        if mse < best_score:
                            best_score = mse
                            best_params = (ls, deg, gamma, alpha)
                            best_weights = w
                            best_K_train = K
                    except np.linalg.LinAlgError:
                        continue
                    except Exception:
                        continue
    
    # If no valid parameters found, use defaults
    if best_params is None:
        ls, deg, gamma, alpha = 0.5, 3, 1.0, 0.1
        K = composite_kernel(X_norm, X_norm, ls, deg, gamma)
        w = np.linalg.solve(K + alpha * np.eye(n), y)
        best_params = (ls, deg, gamma, alpha)
        best_weights = w
        best_K_train = K
    
    ls, deg, gamma, alpha = best_params
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        K_test = composite_kernel(X_test_norm, X_norm, ls, deg, gamma)
        return K_test @ best_weights
    
    return predict
# EVOLVE-BLOCK-END
