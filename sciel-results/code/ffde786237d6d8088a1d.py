import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 290.0
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    if y_var < 1e-12:
        def predict_const(X):
            return np.full(len(X), y_mean)
        return predict_const

    # Standardize inputs
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0
    X_train_std = (X_train - X_mean) / X_std
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Matern 5/2 kernel
    def matern52(X1, X2, length_scale):
        X1_sq = np.sum(X1**2, axis=1)
        X2_sq = np.sum(X2**2, axis=1)
        dist_sq = X1_sq[:, None] + X2_sq[None, :] - 2 * np.dot(X1, X2.T)
        dist_sq = np.maximum(dist_sq, 0)
        dist = np.sqrt(dist_sq)
        
        sqrt3_dist = np.sqrt(3) * dist / length_scale
        # Corrected formula: 1 + sqrt(3)r + 1.5*r^2
        # With r' = sqrt(3)r, r^2 = r'^2 / 3
        # 1.5 * r'^2 / 3 = 0.5 * r'^2
        K = (1 + sqrt3_dist + 0.5 * sqrt3_dist**2) * np.exp(-sqrt3_dist)
        return K

    def compute_k_matrix(X_std, length_scale, signal_var):
        return signal_var * matern52(X_std, X_std, length_scale)

    def negative_log_likelihood(params, X_std, y):
        try:
            log_ls, log_sv, log_nv = params
            ls = np.exp(log_ls)
            sv = np.exp(log_sv)
            nv = np.exp(log_nv)
            
            K = compute_k_matrix(X_std, ls, sv)
            K += nv * np.eye(len(y))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            
            alpha = np.linalg.solve(L, y)
            log_det_K = 2 * np.sum(np.log(np.diag(L)))
            
            nll = 0.5 * y.T @ alpha + 0.5 * log_det_K + 0.5 * len(y) * np.log(2 * np.pi)
            return nll
        except Exception:
            return 1e10

    initial_params = [
        np.log(1.0),
        np.log(max(y_var, 1e-8)),
        np.log(1e-5 * max(y_var, 1e-8))
    ]
    
    best_params = initial_params
    best_nll = negative_log_likelihood(initial_params, X_train_std, y_train)
    
    if time_left() > 10.0:
        try:
            res = minimize(
                negative_log_likelihood,
                initial_params,
                args=(X_train_std, y_train),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if res.fun < best_nll:
                best_params = res.x
                best_nll = res.fun
        except Exception:
            pass

    ls = np.exp(best_params[0])
    sv = np.exp(best_params[1])
    nv = np.exp(best_params[2])
    
    K_train = compute_k_matrix(X_train_std, ls, sv)
    K_train += nv * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_train)
    except np.linalg.LinAlgError:
        def predict_ridge(X):
            return np.full(len(X), y_mean)
        return predict_ridge

    alpha = np.linalg.solve(L, y_train)
    
    def predict(X):
        X_std = (X - X_mean) / X_std
        K_test_train = compute_k_matrix(X_std, ls, sv)
        pred = K_test_train @ alpha
        return pred

    return predict
