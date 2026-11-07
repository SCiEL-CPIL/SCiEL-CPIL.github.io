import numpy as np
import time
from scipy.linalg import solve
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 285.0  # Safety margin from 330s kill limit

    n, d = X_train.shape

    # Normalize inputs to [0, 1]
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges

    # Center y
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean

    # Split for validation
    n_val = max(10, int(n * 0.2))
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)

    X_tr_norm = X_norm[train_idx]
    y_tr = y_centered[train_idx]
    X_val_norm = X_norm[val_idx]
    y_val = y_centered[val_idx]
    n_tr = len(y_tr)

    # Helper to check time
    def time_ok():
        return time.time() < deadline - 10

    # --- Kernel Definitions ---
    def rbf_kernel(X1, X2, length_scale):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-sq_dists / (2 * length_scale**2))

    def poly_kernel(X1, X2, degree, coef0=1.0, gamma=1.0):
        return (gamma * (X1 @ X2.T) + coef0) ** degree

    def matern52_kernel(X1, X2, length_scale, signal_var=1.0):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        sqrt3 = np.sqrt(3.0)
        K = signal_var * (1 + sqrt3 * dists + dists**2) * np.exp(-sqrt3 * dists)
        return K

    def generate_polynomial_features(X, degree):
        n = X.shape[0]
        d = X.shape[1]
        feats = [np.ones(n)]
        if degree >= 1:
            for i in range(d):
                feats.append(X[:, i])
        if degree >= 2:
            for i in range(d):
                for j in range(i, d):
                    feats.append(X[:, i] * X[:, j])
        if degree >= 3:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        feats.append(X[:, i] * X[:, j] * X[:, k])
        if degree >= 4:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        for l in range(k, d):
                            feats.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
        if degree >= 5 and d == 1:
            feats.append(X[:, 0] ** 5)
            if degree >= 6:
                feats.append(X[:, 0] ** 6)
        return np.column_stack(feats)

    # --- Model Fitters (return val_mse, model_data) ---
    def fit_ridge_poly(X_tr, y_tr, X_val, y_val, degree, alpha):
        X_tr_feat = generate_polynomial_features(X_tr, degree)
        X_val_feat = generate_polynomial_features(X_val, degree)
        A = X_tr_feat.T @ X_tr_feat + alpha * np.eye(X_tr_feat.shape[1])
        b = X_tr_feat.T @ y_tr
        try:
            w = solve(A, b, assume_a='pos')
        except:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        y_pred = X_val_feat @ w
        mse = np.mean((y_pred - y_val) ** 2)
        if not np.isfinite(mse):
            return np.inf, None
        return mse, {'type': 'ridge', 'degree': degree, 'alpha': alpha, 'w': w, 'X_feat_dim': X_tr_feat.shape[1]}

    def fit_kernel_ridge(X_tr, y_tr, X_val, y_val, kernel_func, kernel_params, alpha):
        K_tr = kernel_func(X_tr, X_tr, **kernel_params)
        K_val_tr = kernel_func(X_val, X_tr, **kernel_params)
        A = K_tr + alpha * np.eye(n_tr)
        try:
            w = solve(A, y_tr, assume_a='pos')
        except:
            w = np.linalg.lstsq(A, y_tr, rcond=None)[0]
        y_pred = K_val_tr @ w
        mse = np.mean((y_pred - y_val) ** 2)
        if not np.isfinite(mse):
            return np.inf, None
        return mse, {'type': 'kernel_ridge', 'kernel_func': kernel_func, 'kernel_params': kernel_params, 'alpha': alpha, 'w': w}

    def fit_nystrom(X_tr, y_tr, X_val, y_val, n_landmarks, length_scale, alpha):
        n = X_tr.shape[0]
        n_landmarks = min(n_landmarks, n)
        landmarks = np.zeros((n_landmarks, d))
        landmarks[0] = X_tr[np.random.randint(n)]
        
        # Efficient greedy landmark selection
        for i in range(1, n_landmarks):
            # dists to nearest landmark
            dists = np.min(np.sum((X_tr[:, None, :] - landmarks[None, :i, :])**2, axis=2), axis=1)
            landmarks[i] = X_tr[np.argmax(dists)]
            
        K_XL = rbf_kernel(X_tr, landmarks, length_scale)
        K_LL = rbf_kernel(landmarks, landmarks, length_scale)
        
        A = K_LL + alpha * np.eye(n_landmarks)
        b = K_XL.T @ y_tr
        try:
            w = solve(A, b, assume_a='pos')
        except:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
            
        K_val_L = rbf_kernel(X_val, landmarks, length_scale)
        y_pred = K_val_L @ w
        mse = np.mean((y_pred - y_val) ** 2)
        if not np.isfinite(mse):
            return np.inf, None
        return mse, {'type': 'nystrom', 'n_landmarks': n_landmarks, 'length_scale': length_scale, 'alpha': alpha, 'w': w, 'landmarks': landmarks}

    # --- Search ---
    candidates = []
    
    # 1. Polynomial Ridge
    for deg in [1, 2, 3, 4]:
        for alpha in [1e-4, 1e-2, 1.0, 10.0]:
            if not time_ok(): break
            mse, data = fit_ridge_poly(X_tr_norm, y_tr, X_val_norm, y_val, deg, alpha)
            if mse < np.inf:
                candidates.append((mse, data))
    
    # 2. RBF Kernel Ridge
    for ls in [0.1, 0.2, 0.3, 0.5, 1.0]:
        for alpha in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
            if not time_ok(): break
            mse, data = fit_kernel_ridge(X_tr_norm, y_tr, X_val_norm, y_val, rbf_kernel, {'length_scale': ls}, alpha)
            if mse < np.inf:
                candidates.append((mse, data))
    
    # 3. Polynomial Kernel Ridge
    for deg in [2, 3, 4]:
        for gamma in [0.5, 1.0, 2.0]:
            for alpha in [1e-3, 1e-2, 0.1]:
                if not time_ok(): break
                mse, data = fit_kernel_ridge(X_tr_norm, y_tr, X_val_norm, y_val, poly_kernel, {'degree': deg, 'gamma': gamma, 'coef0': 1.0}, alpha)
                if mse < np.inf:
                    candidates.append((mse, data))
    
    # 4. Matern-5/2 Kernel Ridge
    for ls in [0.1, 0.2, 0.5, 1.0]:
        for alpha in [1e-3, 1e-2, 0.1]:
            if not time_ok(): break
            mse, data = fit_kernel_ridge(X_tr_norm, y_tr, X_val_norm, y_val, matern52_kernel, {'length_scale': ls, 'signal_var': 1.0}, alpha)
            if mse < np.inf:
                candidates.append((mse, data))
    
    # 5. Nyström Sparse Kernel
    for nl in [20, 50, 100]:
        for ls in [0.2, 0.3, 0.5]:
            for alpha in [1e-3, 1e-2, 0.1]:
                if not time_ok(): break
                mse, data = fit_nystrom(X_tr_norm, y_tr, X_val_norm, y_val, nl, ls, alpha)
                if mse < np.inf:
                    candidates.append((mse, data))

    if not candidates:
        # Fallback: simple linear
        mse, data = fit_ridge_poly(X_tr_norm, y_tr, X_val_norm, y_val, 1, 1.0)
        if data:
            candidates.append((mse, data))
        else:
            # Absolute fallback
            def predict(X):
                return np.full(X.shape[0], y_mean)
            return predict

    # Sort by validation MSE
    candidates.sort(key=lambda x: x[0])
    
    # Ensemble top 3 models if they are good
    # We need to refit the top models on ALL data to get their final predictors.
    # However, refitting kernel methods on all data is O(N^3).
    # For N=200, this is fast.
    
    top_k = min(3, len(candidates))
    top_models = candidates[:top_k]
    
    # We need to generate the predict functions for these top models on the FULL dataset.
    # Let's store the parameters and refit.
    
    final_predictors = []
    
    for mse, data in top_models:
        if not time_ok(): break
        
        if data['type'] == 'ridge':
            degree = data['degree']
            alpha = data['alpha']
            X_all_feat = generate_polynomial_features(X_norm, degree)
            A_full = X_all_feat.T @ X_all_feat + alpha * np.eye(X_all_feat.shape[1])
            b_full = X_all_feat.T @ y_centered
            try:
                w_full = solve(A_full, b_full, assume_a='pos')
            except:
                w_full = np.linalg.lstsq(A_full, b_full, rcond=None)[0]
                
            def make_ridge_predict(w, deg, mean, minv, rang):
                def predict(X):
                    X_test_norm = (X - minv) / rang
                    X_test_feat = generate_polynomial_features(X_test_norm, deg)
                    return X_test_feat @ w + mean
                return predict
            
            final_predictors.append(make_ridge_predict(w_full, degree, y_mean, min_vals, ranges))
            
        elif data['type'] == 'kernel_ridge':
            kf = data['kernel_func']
            kp = data['kernel_params']
            alpha = data['alpha']
            K_full = kf(X_norm, X_norm, **kp)
            A_full = K_full + alpha * np.eye(n)
            try:
                w_full = solve(A_full, y_centered, assume_a='pos')
            except:
                w_full = np.linalg.lstsq(A_full, y_centered, rcond=None)[0]
                
            def make_kernel_predict(w, kf, kp, mean, minv, rang, X_train_norm):
                def predict(X):
                    X_test_norm = (X - minv) / rang
                    K_test = kf(X_test_norm, X_train_norm, **kp)
                    return K_test @ w + mean
                return predict
            
            final_predictors.append(make_kernel_predict(w_full, kf, kp, y_mean, min_vals, ranges, X_norm))
            
        elif data['type'] == 'nystrom':
            nl = data['n_landmarks']
            ls = data['length_scale']
            alpha = data['alpha']
            
            # Refit Nyström on all data
            n_all = X_norm.shape[0]
            landmarks = np.zeros((nl, d))
            landmarks[0] = X_norm[np.random.randint(n_all)]
            for i in range(1, nl):
                dists = np.min(np.sum((X_norm[:, None, :] - landmarks[None, :i, :])**2, axis=2), axis=1)
                landmarks[i] = X_norm[np.argmax(dists)]
                
            K_XL = rbf_kernel(X_norm, landmarks, ls)
            K_LL = rbf_kernel(landmarks, landmarks, ls)
            A = K_LL + alpha * np.eye(nl)
            b = K_XL.T @ y_centered
            try:
                w_full = solve(A, b, assume_a='pos')
            except:
                w_full = np.linalg.lstsq(A, b, rcond=None)[0]
                
            def make_nystrom_predict(w, ls, mean, minv, rang, lms):
                def predict(X):
                    X_test_norm = (X - minv) / rang
                    K_test_L = rbf_kernel(X_test_norm, lms, ls)
                    return K_test_L @ w + mean
                return predict
            
            final_predictors.append(make_nystrom_predict(w_full, ls, y_mean, min_vals, ranges, landmarks))

    if not final_predictors:
        # Should not happen, but just in case
        def predict(X):
            return np.full(X.shape[0], y_mean)
        return predict

    # Ensemble: average the top predictors
    # Weighted average based on inverse MSE might be better, but simple average is robust.
    # Let's try simple average.
    
    def predict(X):
        preds = []
        for p in final_predictors:
            try:
                pred = p(X)
                if np.all(np.isfinite(pred)):
                    preds.append(pred)
            except:
                continue
        if not preds:
            return np.full(X.shape[0], y_mean)
        return np.mean(preds, axis=0)

    return predict
