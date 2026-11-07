import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 320.0  # Safety margin from 330s limit
    
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
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # K-fold cross-validation for model selection
    n_folds = 5
    fold_size = n // n_folds
    folds = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(n_folds - 1)]
    folds.append(np.arange((n_folds - 1) * fold_size, n))
    
    def time_left():
        return deadline - time.time()
    
    # Feature generation - rich basis
    def generate_features(X, mode='full'):
        n = X.shape[0]
        d = X.shape[1]
        feats = [np.ones(n)]
        
        # Linear
        for i in range(d):
            feats.append(X[:, i])
        
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
        
        # Cubic (limited for high d)
        if d <= 3:
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        if i <= j <= k:
                            feats.append(X[:, i] * X[:, j] * X[:, k])
        elif d <= 4:
            for i in range(d):
                feats.append(X[:, i]**3)
            for i in range(d):
                for j in range(i+1, d):
                    feats.append(X[:, i]**2 * X[:, j])
                    feats.append(X[:, i] * X[:, j]**2)
        
        # Quartic (only for small d)
        if d <= 2:
            for i in range(d):
                feats.append(X[:, i]**4)
            if d == 2:
                feats.append(X[:, 0]**3 * X[:, 1])
                feats.append(X[:, 0] * X[:, 1]**3)
        
        # Trigonometric
        for i in range(d):
            feats.append(np.sin(X[:, i]))
            feats.append(np.cos(X[:, i]))
            feats.append(np.sin(2 * np.pi * X[:, i]))
            feats.append(np.cos(2 * np.pi * X[:, i]))
            if d <= 2:
                feats.append(np.sin(4 * np.pi * X[:, i]))
                feats.append(np.cos(4 * np.pi * X[:, i]))
        
        # Exponential and log
        for i in range(d):
            feats.append(np.exp(-2 * X[:, i]))
            feats.append(np.exp(-X[:, i]))
            feats.append(np.exp(-0.5 * X[:, i]))
            feats.append(np.log(X[:, i] + 1e-8))
            feats.append(1.0 / (X[:, i] + 1e-8))
            feats.append(1.0 / (X[:, i]**2 + 1e-8))
        
        # Pairwise interactions with trig
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, min(d, i+2)):  # limit to nearby pairs
                    feats.append(np.sin(X[:, i]) * np.cos(X[:, j]))
                    feats.append(np.sin(X[:, i]) * np.sin(X[:, j]))
                    feats.append(np.cos(X[:, i]) * np.cos(X[:, j]))
        
        X_feat = np.column_stack(feats)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        # Clip extreme values
        X_feat = np.clip(X_feat, -1e6, 1e6)
        return X_feat
    
    # Generate features for training data
    X_feat_all = generate_features(X_norm)
    n_features = X_feat_all.shape[1]
    
    # Model 1: Ridge regression with optimal alpha
    def ridge_cv(alpha):
        scores = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
            try:
                A = X_feat_all[tr_idx].T @ X_feat_all[tr_idx] + alpha * np.eye(n_features)
                b = X_feat_all[tr_idx].T @ y_centered[tr_idx]
                w = solve(A, b, assume_a='pos')
                pred = X_feat_all[val_idx] @ w
                mse = np.mean((pred - y_centered[val_idx])**2)
                scores.append(mse)
            except:
                return np.inf
        return np.mean(scores)
    
    # Model 2: Lasso-like (L1 + L2) via coordinate descent
    def lasso_cv(alpha_l1, alpha_l2):
        scores = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
            X_tr = X_feat_all[tr_idx]
            y_tr = y_centered[tr_idx]
            X_val = X_feat_all[val_idx]
            y_val = y_centered[val_idx]
            
            # Soft thresholding coordinate descent
            w = np.zeros(n_features)
            XTX = X_tr.T @ X_tr + alpha_l2 * np.eye(n_features)
            XtY = X_tr.T @ y_tr
            
            for iteration in range(50):
                w_old = w.copy()
                for j in range(n_features):
                    if np.abs(w_old[j]) > 1e-10:
                        residual = XtY[j] - XTX[j, j] * w_old[j]
                        for m in range(n_features):
                            if m != j:
                                residual += XTX[j, m] * w_old[m]
                        w[j] = np.sign(residual) * max(0, abs(residual) - alpha_l1) / XTX[j, j]
                    else:
                        residual = XtY[j]
                        for m in range(n_features):
                            if m != j:
                                residual += XTX[j, m] * w_old[m]
                        w[j] = np.sign(residual) * max(0, abs(residual) - alpha_l1) / XTX[j, j]
                if np.max(np.abs(w - w_old)) < 1e-8:
                    break
            
            pred = X_val @ w
            mse = np.mean((pred - y_val)**2)
            scores.append(mse)
        return np.mean(scores)
    
    # Model 3: GP with RBF kernel
    def gp_cv(length_scale, signal_var, noise_var):
        scores = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
            X_tr = X_norm[tr_idx]
            y_tr = y_centered[tr_idx]
            X_val = X_norm[val_idx]
            y_val = y_centered[val_idx]
            
            def rbf(X1, X2):
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
                sq_dists = np.maximum(sq_dists, 0)
                return signal_var * np.exp(-sq_dists / (2 * length_scale**2))
            
            try:
                K_tr = rbf(X_tr, X_tr) + noise_var * np.eye(len(X_tr))
                w = solve(K_tr, y_tr, assume_a='pos')
                K_val_tr = rbf(X_val, X_tr)
                pred = K_val_tr @ w
                mse = np.mean((pred - y_val)**2)
                scores.append(mse)
            except:
                return np.inf
        return np.mean(scores)
    
    # Evaluate models
    best_model_type = 'ridge'
    best_cv_score = np.inf
    best_params = {}
    
    # Ridge alpha search
    if time_left() > 20:
        alphas = np.logspace(-4, 2, 12)
        for alpha in alphas:
            score = ridge_cv(alpha)
            if score < best_cv_score:
                best_cv_score = score
                best_model_type = 'ridge'
                best_params = {'alpha': alpha}
    
    # Lasso search
    if time_left() > 30:
        for alpha_l1 in np.logspace(-3, 1, 6):
            for alpha_l2 in np.logspace(-4, 0, 4):
                score = lasso_cv(alpha_l1, alpha_l2)
                if score < best_cv_score:
                    best_cv_score = score
                    best_model_type = 'lasso'
                    best_params = {'alpha_l1': alpha_l1, 'alpha_l2': alpha_l2}
    
    # GP hyperparameter search
    if time_left() > 40:
        # Coarse grid
        ls_grid = np.logspace(-1, 1, 8)
        sv_grid = np.logspace(-1, 2, 6)
        nv_grid = np.logspace(-4, -1, 4)
        
        for ls in ls_grid:
            if time_left() < 15:
                break
            for sv in sv_grid:
                if time_left() < 10:
                    break
                for nv in nv_grid:
                    score = gp_cv(ls, sv, nv)
                    if score < best_cv_score:
                        best_cv_score = score
                        best_model_type = 'gp'
                        best_params = {'ls': ls, 'sv': sv, 'nv': nv}
        
        # Refine best GP
        if best_model_type == 'gp' and time_left() > 20:
            def neg_mse(params):
                ls = np.exp(params[0])
                sv = np.exp(params[1])
                nv = np.exp(params[2])
                return gp_cv(ls, sv, nv)
            
            x0 = np.array([np.log(best_params['ls']), np.log(best_params['sv']), np.log(best_params['nv'])])
            try:
                result = minimize(neg_mse, x0, method='Nelder-Mead', 
                                 options={'maxiter': 50, 'xatol': 0.1, 'fatol': 1e-6})
                if result.fun < best_cv_score:
                    best_cv_score = result.fun
                    ls = np.exp(result.x[0])
                    sv = np.exp(result.x[1])
                    nv = np.exp(result.x[2])
                    best_params = {'ls': ls, 'sv': sv, 'nv': nv}
            except:
                pass
    
    # Final model selection
    if best_model_type == 'ridge':
        alpha = best_params['alpha']
        A = X_feat_all.T @ X_feat_all + alpha * np.eye(n_features)
        b = X_feat_all.T @ y_centered
        w_full = solve(A, b, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = generate_features(X_test_norm)
            X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test_feat = np.clip(X_test_feat, -1e6, 1e6)
            return X_test_feat @ w_full + y_mean
    
    elif best_model_type == 'lasso':
        alpha_l1 = best_params['alpha_l1']
        alpha_l2 = best_params['alpha_l2']
        
        # Fit lasso on all data
        w = np.zeros(n_features)
        XTX = X_feat_all.T @ X_feat_all + alpha_l2 * np.eye(n_features)
        XtY = X_feat_all.T @ y_centered
        
        for iteration in range(100):
            w_old = w.copy()
            for j in range(n_features):
                residual = XtY[j] - XTX[j, j] * w_old[j]
                for m in range(n_features):
                    if m != j:
                        residual += XTX[j, m] * w_old[m]
                w[j] = np.sign(residual) * max(0, abs(residual) - alpha_l1) / XTX[j, j]
            if np.max(np.abs(w - w_old)) < 1e-8:
                break
        
        w_full = w
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = generate_features(X_test_norm)
            X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test_feat = np.clip(X_test_feat, -1e6, 1e6)
            return X_test_feat @ w_full + y_mean
    
    elif best_model_type == 'gp':
        ls = best_params['ls']
        sv = best_params['sv']
        nv = best_params['nv']
        
        def rbf(X1, X2):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return sv * np.exp(-sq_dists / (2 * ls**2))
        
        K_full = rbf(X_norm, X_norm) + nv * np.eye(n)
        w_full = solve(K_full, y_centered, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = rbf(X_test_norm, X_norm)
            return K_test @ w_full + y_mean
    
    else:
        # Fallback: simple linear
        A = X_feat_all.T @ X_feat_all + 1.0 * np.eye(n_features)
        b = X_feat_all.T @ y_centered
        w_full = solve(A, b, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = generate_features(X_test_norm)
            X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test_feat = np.clip(X_test_feat, -1e6, 1e6)
            return X_test_feat @ w_full + y_mean
    
    # Ensure predict returns finite values
    def safe_predict(X):
        try:
            pred = predict(X)
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            return pred
        except:
            return np.full(X.shape[0], y_mean)
    
    return safe_predict
