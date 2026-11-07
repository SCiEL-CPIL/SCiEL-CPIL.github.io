import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin from 330s kill limit
    
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
    
    def rbf_kernel(X1, X2, length_scale, signal_var):
        """Compute RBF kernel."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K = signal_var * np.exp(-sq_dists / (2 * length_scale**2))
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K

    def generate_polynomial_features(X, degree):
        """Generate polynomial features up to 'degree'."""
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

    # ---- Sparse Kernel Regression (Nyström) ----
    def fit_nystrom(X, y, n_landmarks, length_scale, alpha):
        """
        Fit sparse kernel regression using Nyström approximation.
        Selects landmarks via k-means-like greedy selection.
        """
        n = X.shape[0]
        n_landmarks = min(n_landmarks, n)
        
        # Greedy landmark selection: start with random point, then pick point with max min-distance
        landmarks = np.zeros((n_landmarks, d))
        
        # Initial random landmark
        landmarks[0] = X[np.random.randint(n)]
        
        # Compute distances from current landmarks to all points
        # Use a simple greedy approach
        for i in range(1, n_landmarks):
            # Distance to nearest existing landmark
            dists = np.min(np.sum((X[:, None, :] - landmarks[None, :i, :])**2, axis=2), axis=1)
            # Pick point with maximum distance to nearest landmark
            landmarks[i] = X[np.argmax(dists)]
        
        # Compute kernel matrix between all points and landmarks
        K_XL = rbf_kernel(X, landmarks, length_scale, 1.0)  # (n, n_landmarks)
        K_LL = rbf_kernel(landmarks, landmarks, length_scale, 1.0)  # (n_landmarks, n_landmarks)
        
        # Solve (K_LL + alpha * I) w = K_XL^T y
        # Then prediction: K_XL w
        try:
            w = solve(K_LL + alpha * np.eye(n_landmarks), K_XL.T @ y, assume_a='pos')
        except:
            w = np.linalg.lstsq(K_LL + alpha * np.eye(n_landmarks), K_XL.T @ y, rcond=None)[0]
        
        return K_XL, w, landmarks

    def predict_nystrom(X_new, K_XL_train, w, landmarks, length_scale):
        """Predict using sparse kernel regression."""
        K_new_L = rbf_kernel(X_new, landmarks, length_scale, 1.0)
        return K_new_L @ w

    # ---- Model candidates ----
    best_model_name = None
    best_val_mse = np.inf
    best_model_data = None

    # Candidate 1: Ridge Regression with polynomial features
    for deg in [1, 2, 3, 4]:
        for alpha in [1e-4, 1e-2, 1.0, 10.0]:
            if time.time() > deadline - 20:
                break
            try:
                X_tr_feat = generate_polynomial_features(X_tr_norm, deg)
                X_val_feat = generate_polynomial_features(X_val_norm, deg)
                
                A = X_tr_feat.T @ X_tr_feat + alpha * np.eye(X_tr_feat.shape[1])
                b = X_tr_feat.T @ y_tr
                
                try:
                    w = solve(A, b, assume_a='pos')
                except:
                    w = np.linalg.lstsq(A, b, rcond=None)[0]
                    
                y_pred_val = X_val_feat @ w
                val_mse = np.mean((y_pred_val - y_val) ** 2)
                
                if np.isfinite(val_mse) and val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_model_name = f"Ridge_D{deg}_A{alpha}"
                    best_model_data = {
                        'type': 'ridge',
                        'degree': deg,
                        'alpha': alpha
                    }
            except Exception:
                continue

    # Candidate 2: Nyström Sparse Kernel Regression
    for n_landmarks in [20, 50, 100]:
        for length_scale in [0.1, 0.2, 0.3, 0.5]:
            for alpha in [1e-3, 1e-2, 0.1]:
                if time.time() > deadline - 15:
                    break
                try:
                    K_tr_L, w, landmarks = fit_nystrom(X_tr_norm, y_tr, n_landmarks, length_scale, alpha)
                    y_pred_val = predict_nystrom(X_val_norm, K_tr_L, w, landmarks, length_scale)
                    val_mse = np.mean((y_pred_val - y_val) ** 2)
                    
                    if np.isfinite(val_mse) and val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_model_name = f"Nystrom_L{n_landmarks}_LS{length_scale}_A{alpha}"
                        best_model_data = {
                            'type': 'nystrom',
                            'n_landmarks': n_landmarks,
                            'length_scale': length_scale,
                            'alpha': alpha
                        }
                except Exception:
                    continue

    # Candidate 3: Full GP with fixed hyperparameters (fallback for non-polynomial)
    for length_scale in [0.2, 0.3, 0.5]:
        if time.time() > deadline - 10:
            break
        try:
            signal_var = 1.0
            noise_var = 1e-4
            
            K_tr_tr = rbf_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
            
            try:
                w = solve(K_tr_tr, y_tr, assume_a='pos')
            except:
                w = np.linalg.lstsq(K_tr_tr, y_tr, rcond=None)[0]
                
            K_val_tr = rbf_kernel(X_val_norm, X_tr_norm, length_scale, signal_var)
            y_pred_val = K_val_tr @ w
            val_mse = np.mean((y_pred_val - y_val) ** 2)
            
            if np.isfinite(val_mse) and val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model_name = f"GP_LS{length_scale}"
                best_model_data = {
                    'type': 'gp',
                    'length_scale': length_scale,
                    'signal_var': signal_var,
                    'noise_var': noise_var
                }
        except Exception:
            continue

    # If no model was selected, default to simple linear
    if best_model_data is None:
        best_model_data = {'type': 'ridge', 'degree': 1, 'alpha': 1.0}
        
    # Refit the best model on ALL training data
    if best_model_data['type'] == 'ridge':
        degree = best_model_data['degree']
        alpha = best_model_data['alpha']
        
        X_all_feat = generate_polynomial_features(X_norm, degree)
        A_full = X_all_feat.T @ X_all_feat + alpha * np.eye(X_all_feat.shape[1])
        b_full = X_all_feat.T @ y_centered
        
        try:
            w_full = solve(A_full, b_full, assume_a='pos')
        except:
            w_full = np.linalg.lstsq(A_full, b_full, rcond=None)[0]
            
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = generate_polynomial_features(X_test_norm, degree)
            return X_test_feat @ w_full + y_mean

    elif best_model_data['type'] == 'nystrom':
        n_landmarks = best_model_data['n_landmarks']
        length_scale = best_model_data['length_scale']
        alpha = best_model_data['alpha']
        
        # Refit on all data
        K_all_L, w_full, landmarks = fit_nystrom(X_norm, y_centered, n_landmarks, length_scale, alpha)
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            return predict_nystrom(X_test_norm, K_all_L, w_full, landmarks, length_scale) + y_mean

    else:  # GP
        length_scale = best_model_data['length_scale']
        signal_var = best_model_data['signal_var']
        noise_var = best_model_data['noise_var']
        
        K_full = rbf_kernel(X_norm, X_norm, length_scale, signal_var) + noise_var * np.eye(n)
        
        try:
            w_full = solve(K_full, y_centered, assume_a='pos')
        except:
            w_full = np.linalg.lstsq(K_full, y_centered, rcond=None)[0]
            
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = rbf_kernel(X_test_norm, X_norm, length_scale, signal_var)
            return K_test @ w_full + y_mean

    return predict
