"""Symbolic regression via Kernel Ridge Regression with Poly+RBF kernels."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (330s total, use 320s safety margin)
_DEADLINE = time.time() + 320.0

def _time_left():
    return _DEADLINE - time.time()

def _normalize(X, X_min, X_max):
    # Avoid division by zero
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    return (X - X_min) / X_range

def _kernel_poly_rbf(X, Y, degree, gamma, c_poly=1.0, c_rbf=1.0):
    # X and Y are normalized [0,1]
    # Poly kernel: (X dot Y + 1)^degree
    # RBF kernel: exp(-gamma * ||X - Y||^2)
    XY = X @ Y.T
    K_poly = (XY + 1.0) ** degree
    
    # RBF
    # ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2XY
    X_sq = np.sum(X**2, axis=1)
    Y_sq = np.sum(Y**2, axis=1)
    sq_dists = X_sq[:, np.newaxis] + Y_sq[np.newaxis, :] - 2 * XY
    # Clip small negative values due to floating point
    sq_dists = np.maximum(sq_dists, 0.0)
    K_rbf = np.exp(-gamma * sq_dists)
    
    return c_poly * K_poly + c_rbf * K_rbf

def fit(X_train, y_train):
    X_train = np.atleast_2d(X_train)
    y_train = np.asarray(y_train, dtype=float)
    n_samples, d = X_train.shape
    
    # Normalize X_train to [0, 1]
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_norm = _normalize(X_train, X_min, X_max)
    
    best_alpha = None
    best_params = None
    best_train_mse = np.inf
    
    # Grid search hyperparameters
    # Degree 3 and 4 are common for symbolic regression
    degrees = [3, 4]
    # Gamma for RBF. Scale by 1/d is a good heuristic.
    gammas = [1.0/d, 2.0/d, 5.0/d, 10.0/d, 20.0/d, 50.0/d]
    # Regularization lambda. Noiseless data -> small lambda.
    lambdas = [1e-10, 1e-8, 1e-6, 1e-4]
    
    # Precompute X_norm squared sums for RBF speedup if needed, but n=200 is small.
    
    for degree in degrees:
        for gamma in gammas:
            for lam in lambdas:
                if _time_left() < 1.0:
                    break
                
                try:
                    K = _kernel_poly_rbf(X_norm, X_norm, degree, gamma)
                    # Add regularization
                    K_reg = K + lam * np.eye(n_samples)
                    
                    # Solve for alpha
                    alpha = np.linalg.solve(K_reg, y_train)
                    
                    # Calculate training predictions
                    y_pred = K @ alpha
                    mse = np.mean((y_pred - y_train) ** 2)
                    
                    if mse < best_train_mse:
                        best_train_mse = mse
                        best_alpha = alpha
                        best_params = (degree, gamma, lam)
                except Exception:
                    continue
        
        if _time_left() < 1.0:
            break

    # Fallback if KRR failed
    if best_alpha is None:
        # Simple polynomial regression fallback
        # Using degree 4 polynomial features
        def _poly_features(X, degree):
            d = X.shape[1]
            n = len(X)
            cols = [np.ones(n)]
            for j in range(d):
                cols.append(X[:, j])
            for deg in range(2, degree + 1):
                for i in range(d):
                    for j in range(i, d):
                        if deg == 2:
                            if i == j:
                                cols.append(X[:, i] ** 2)
                            else:
                                cols.append(X[:, i] * X[:, j])
                        else:
                            if i == j:
                                cols.append(X[:, i] ** deg)
            return np.column_stack(cols)
        
        F_train = _poly_features(X_norm, 4)
        try:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
            
            def predict_fallback(X):
                X = np.atleast_2d(X)
                X_norm_test = _normalize(X, X_min, X_max)
                F_test = _poly_features(X_norm_test, 4)
                return F_test @ w
            return predict_fallback
        except Exception:
            return lambda X: np.full(len(np.atleast_2d(X)), np.mean(y_train))

    degree, gamma, lam = best_params
    
    def predict(X):
        X = np.atleast_2d(X)
        X_norm_test = _normalize(X, X_min, X_max)
        
        # Compute kernel between test points and train points
        # K_test: (n_test, n_train)
        K_test = _kernel_poly_rbf(X_norm_test, X_norm, degree, gamma)
        
        # Prediction: K_test @ alpha
        preds = K_test @ best_alpha
        
        # Ensure finite
        if not np.all(np.isfinite(preds)):
            preds = np.where(np.isfinite(preds), preds, np.mean(y_train))
            
        return preds

    return predict
# EVOLVE-BLOCK-END
