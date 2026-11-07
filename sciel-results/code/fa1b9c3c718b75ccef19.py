"""Symbolic regression using Gaussian Process Regression with multiple kernels."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _rbf_kernel(X1, X2, length_scales):
    """RBF kernel with per-dimension length scales."""
    # Standardize using first set's mean/std
    d = X1.shape[1]
    # Compute pairwise squared distances
    X1_sq = np.sum(X1 ** 2, axis=1)
    X2_sq = np.sum(X2 ** 2, axis=1)
    # Scale by inverse length scales squared
    inv_ls_sq = 1.0 / (length_scales ** 2)
    # K(X1,X2) = exp(-0.5 * sum((x_i - x_j)^2 / l_i^2))
    # = exp(-0.5 * (sum(x_i^2/l_i^2) + sum(x_j^2/l_i^2) - 2*x_i.x_j/l_i^2))
    X1_scaled = X1 * np.sqrt(inv_ls_sq)
    X2_scaled = X2 * np.sqrt(inv_ls_sq)
    sq_dists = X1_sq * inv_ls_sq + X2_sq * inv_ls_sq - 2 * (X1_scaled @ X2_scaled.T)
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-0.5 * sq_dists)


def _poly_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    """Polynomial kernel."""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def _linear_kernel(X1, X2):
    """Linear kernel."""
    return X1 @ X2.T


def _fit_gp(X, y, kernel_type='rbf', time_budget=10.0):
    """Fit a Gaussian Process regressor."""
    n, d = X.shape
    
    # Standardize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    Xs = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    ys = (y - y_mean) / y_std
    
    start_time = time.time()
    
    if kernel_type == 'rbf':
        # Initialize length scales
        length_scales = np.ones(d)
        
        def neg_log_likelihood(theta):
            if time.time() - start_time > time_budget:
                return 1e10
            ls = np.exp(theta[:d])
            sigma_f = np.exp(theta[d])
            sigma_n = np.exp(theta[d+1])
            
            K = _rbf_kernel(Xs, Xs, ls)
            K += sigma_n ** 2 * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                nll = 0.5 * ys @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
                return nll
            except:
                return 1e10
        
        # Initial parameters: log(length_scales), log(sigma_f), log(sigma_n)
        theta0 = np.zeros(d + 2)
        theta0[:d] = 0.0  # length_scales = 1
        theta0[d] = 0.0   # sigma_f = 1
        theta0[d+1] = 0.0 # sigma_n = 1
        
        try:
            res = minimize(neg_log_likelihood, theta0, method='L-BFGS-B',
                          options={'maxiter': 100, 'maxfun': 200})
            if time.time() - start_time < time_budget:
                ls = np.exp(res.x[:d])
                sigma_f = np.exp(res.x[d])
                sigma_n = np.exp(res.x[d+1])
            else:
                ls = np.ones(d)
                sigma_f = 1.0
                sigma_n = 0.1
        except:
            ls = np.ones(d)
            sigma_f = 1.0
            sigma_n = 0.1
        
        # Compute final kernel and solve
        K = _rbf_kernel(Xs, Xs, ls)
        K += sigma_n ** 2 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
        except:
            # Fallback to ridge regression
            w = np.linalg.lstsq(Xs, ys, rcond=None)[0]
            return lambda X: _predict_ridge(X, X_mean, X_std, Xs, ys, w, y_mean, y_std)
        
        def predict(X_test):
            Xs_test = (X_test - X_mean) / X_std
            K_test = _rbf_kernel(Xs_test, Xs, ls)
            mu = K_test @ alpha
            return mu * y_std + y_mean
        
        return predict
    
    elif kernel_type == 'poly':
        # Polynomial kernel GP
        gamma = 1.0 / d
        degree = 3
        coef0 = 1.0
        
        K = _poly_kernel(Xs, Xs, degree, gamma, coef0)
        K += 0.1 ** 2 * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
        except:
            w = np.linalg.lstsq(Xs, ys, rcond=None)[0]
            return lambda X: _predict_ridge(X, X_mean, X_std, Xs, ys, w, y_mean, y_std)
        
        def predict(X_test):
            Xs_test = (X_test - X_mean) / X_std
            K_test = _poly_kernel(Xs_test, Xs, degree, gamma, coef0)
            mu = K_test @ alpha
            return mu * y_std + y_mean
        
        return predict
    
    else:  # linear
        w = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        return lambda X: _predict_ridge(X, X_mean, X_std, Xs, ys, w, y_mean, y_std)


def _predict_ridge(X, X_mean, X_std, Xs_train, ys_train, w, y_mean, y_std):
    Xs = (X - X_mean) / X_std
    return (Xs @ w) * y_std + y_mean


def fit(X_train, y_train):
    """Fit a regression model using Gaussian Process with multiple kernels."""
    start_time = time.time()
    n, d = X_train.shape
    
    # Try RBF kernel first (most flexible)
    try:
        predict_rbf = _fit_gp(X_train, y_train, 'rbf', time_budget=15.0)
    except:
        predict_rbf = None
    
    elapsed = time.time() - start_time
    
    if predict_rbf is not None and elapsed < 20.0:
        # Also try polynomial kernel and ensemble
        try:
            predict_poly = _fit_gp(X_train, y_train, 'poly', time_budget=10.0)
            
            # Evaluate on training data to weight the ensemble
            pred_rbf_train = predict_rbf(X_train)
            pred_poly_train = predict_poly(X_train)
            
            mse_rbf = np.mean((pred_rbf_train - y_train) ** 2)
            mse_poly = np.mean((pred_poly_train - y_train) ** 2)
            
            # Weight inversely proportional to MSE
            w_rbf = 1.0 / (mse_rbf + 1e-10)
            w_poly = 1.0 / (mse_poly + 1e-10)
            w_sum = w_rbf + w_poly
            
            def predict(X):
                return (w_rbf * predict_rbf(X) + w_poly * predict_poly(X)) / w_sum
            
            return predict
        except:
            return predict_rbf
    
    # Fallback: simple polynomial regression
    def features(X):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)
    
    w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
    return lambda X: features(X) @ w
# EVOLVE-BLOCK-END
