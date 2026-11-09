"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 290.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    def build_features(X_in, max_poly_deg, use_trig=True, use_exp=True, use_log=True, use_sqrt=True, use_inv=True):
        """
        Build a rich feature matrix from input X_in.
        For each dimension, generate:
        - Polynomial terms up to max_poly_deg (all combinations)
        - Trig: sin, cos for each dim
        - Exp: exp(-x/scale) for a few scales
        - Log: log(x), log^2(x)
        - Sqrt: sqrt(x)
        - Inv: 1/x
        Plus cross-feature interactions (products of single-dim features)
        """
        n_samples = X_in.shape[0]
        d_in = X_in.shape[1]
        
        feature_list = [np.ones((n_samples, 1))]
        
        # Per-dimension features
        for j in range(d_in):
            xj = X_in[:, j:j+1]
            # Polynomials
            for p in range(1, max_poly_deg + 1):
                feature_list.append(xj ** p)
            
            if use_sqrt:
                feature_list.append(np.sqrt(np.abs(xj) + 1e-10))
            
            if use_log:
                feature_list.append(np.log(np.abs(xj) + 1e-10))
                feature_list.append(np.log(np.abs(xj) + 1e-10) ** 2)
            
            if use_inv:
                feature_list.append(1.0 / (np.abs(xj) + 1e-10))
            
            if use_trig:
                feature_list.append(np.sin(xj))
                feature_list.append(np.cos(xj))
                feature_list.append(np.sin(2 * xj))
                feature_list.append(np.cos(2 * xj))
            
            if use_exp:
                for scale in [1.0, 2.0, 5.0]:
                    feature_list.append(np.exp(-np.abs(xj) / scale))
                    feature_list.append(np.exp(xj / scale))
        
        # Cross-feature interactions: products of single-dim features
        # For d=1: just the above
        # For d>=2: add pairwise products
        if d_in >= 2:
            for j in range(d_in):
                for k in range(j+1, d_in):
                    xj = X_in[:, j:j+1]
                    xk = X_in[:, k:k+1]
                    # Basic cross terms
                    feature_list.append(xj * xk)
                    feature_list.append(xj ** 2 * xk)
                    feature_list.append(xj * xk ** 2)
                    if use_trig:
                        feature_list.append(np.sin(xj) * np.cos(xk))
                        feature_list.append(np.cos(xj) * np.sin(xk))
                    if use_exp:
                        feature_list.append(np.exp(-np.abs(xj)/2.0) * np.exp(-np.abs(xk)/2.0))
        
        # Triple interactions for d=3
        if d_in >= 3:
            x0 = X_in[:, 0:1]
            x1 = X_in[:, 1:2]
            x2 = X_in[:, 2:3]
            feature_list.append(x0 * x1 * x2)
            feature_list.append(x0 ** 2 * x1 * x2)
            feature_list.append(x0 * x1 ** 2 * x2)
            feature_list.append(x0 * x1 * x2 ** 2)
        
        F = np.hstack(feature_list)
        
        # Handle any NaN/Inf
        F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return F
    
    def ridge_fit(F_train, y_train_r, alpha):
        """Fit ridge regression: y = F @ w + b"""
        n_s = F_train.shape[0]
        try:
            A = F_train.T @ F_train + alpha * np.eye(F_train.shape[1])
            w = np.linalg.solve(A, F_train.T @ y_train_r)
        except:
            try:
                w, *_ = np.linalg.lstsq(F_train + 1e-10 * np.eye(F_train.shape[0], F_train.shape[1]), y_train_r, rcond=None)
            except:
                w = np.zeros(F_train.shape[1])
        
        def predict(X_pred):
            X_pred_norm = (X_pred - X_mean) / X_std
            F_pred = build_features(X_pred_norm, max_poly_deg, use_trig, use_exp, use_log, use_sqrt, use_inv)
            pred_r = F_pred @ w
            pred = pred_r * y_std + y_mean
            return pred
        
        return predict
    
    def loocv_score(F_train, y_train_r, alpha):
        """LOOCV score using the hat matrix trick for ridge regression."""
        n_s, p = F_train.shape
        try:
            A = F_train.T @ F_train + alpha * np.eye(p)
            w = np.linalg.solve(A, F_train.T @ y_train_r)
            # Hat matrix: H = F (F^T F + alpha I)^{-1} F^T
            # For LOOCV: e_i = (y_i - f_i) / (1 - H_ii)
            # Compute H_ii efficiently
            # H = F @ inv(A) @ F^T
            # We can compute this as: let B = inv(A), then H_ii = sum_k F_ik^2 * ... 
            # Actually, let's just compute the full hat matrix for n=200
            B = np.linalg.inv(A)
            H = F_train @ B @ F_train.T
            h_diag = np.diag(H)
            h_diag = np.clip(h_diag, 0.0, 0.999)
            pred = F_train @ w
            residuals = y_train_r - pred
            loocv_preds = pred + residuals / (1 - h_diag)
            mse = np.mean((loocv_preds - y_train_r) ** 2)
            return mse
        except:
            return np.inf
    
    # Search over feature configurations and regularization
    max_poly_degs = [2, 3, 4]
    feature_configs = [
        (True, True, True, True, True),   # all features
        (True, True, True, True, False),  # no inv
        (True, True, False, True, True),  # no log
        (True, False, True, True, True),  # no exp
        (False, True, True, True, True),  # no trig
        (True, True, True, False, True),  # no sqrt
    ]
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    best_predictor = None
    best_cv_score = np.inf
    
    # Pre-build features for each config to avoid recomputation
    for max_poly_deg in max_poly_degs:
        for (use_trig, use_exp, use_log, use_sqrt, use_inv) in feature_configs:
            if time.time() > deadline - 20:
                break
            
            try:
                F_train = build_features(X_norm, max_poly_deg, use_trig, use_exp, use_log, use_sqrt, use_inv)
            except:
                continue
            
            # Check for numerical issues
            if not np.all(np.isfinite(F_train)):
                continue
            
            # Standardize features for better conditioning
            F_mean = F_train.mean(axis=0)
            F_std = F_train.std(axis=0)
            F_std[F_std == 0] = 1.0
            F_train_std = (F_train - F_mean) / F_std
            
            for alpha in alphas:
                if time.time() > deadline - 20:
                    break
                
                try:
                    cv_score = loocv_score(F_train_std, y_norm, alpha)
                    if cv_score < best_cv_score:
                        best_cv_score = cv_score
                        # Fit on full data with this config
                        try:
                            A = F_train_std.T @ F_train_std + alpha * np.eye(F_train_std.shape[1])
                            w_full = np.linalg.solve(A, F_train_std.T @ y_norm)
                        except:
                            w_full = np.zeros(F_train_std.shape[1])
                        
                        max_poly_deg_best = max_poly_deg
                        use_trig_best = use_trig
                        use_exp_best = use_exp
                        use_log_best = use_log
                        use_sqrt_best = use_sqrt
                        use_inv_best = use_inv
                        F_mean_best = F_mean
                        F_std_best = F_std
                        w_full_best = w_full
                        
                        def predict(X_pred, w=w_full_best, F_mean_b=F_mean_best, F_std_b=F_std_best,
                                   mpd=max_poly_deg_best, ut=use_trig_best, ue=use_exp_best,
                                   ul=use_log_best, us=use_sqrt_best, ui=use_inv_best):
                            X_pred_norm = (X_pred - X_mean) / X_std
                            F_pred = build_features(X_pred_norm, mpd, ut, ue, ul, us, ui)
                            F_pred = np.nan_to_num(F_pred, nan=0.0, posinf=1e6, neginf=-1e6)
                            F_pred_std = (F_pred - F_mean_b) / F_std_b
                            pred_r = F_pred_std @ w
                            pred = pred_r * y_std + y_mean
                            return pred
                        
                        best_predictor = predict
                except Exception:
                    pass
    
    # Fallback: simple linear regression on normalized features
    if best_predictor is None:
        try:
            A = X_norm.T @ X_norm + 0.1 * np.eye(d)
            w = np.linalg.solve(A, X_norm.T @ y_norm)
            def predict(X_pred):
                X_pred_norm = (X_pred - X_mean) / X_std
                pred_r = X_pred_norm @ w
                pred = pred_r * y_std + y_mean
                return pred
            best_predictor = predict
        except:
            def predict(X_pred):
                return np.full(X_pred.shape[0], y_mean)
            best_predictor = predict
    
    return best_predictor
# EVOLVE-BLOCK-END
