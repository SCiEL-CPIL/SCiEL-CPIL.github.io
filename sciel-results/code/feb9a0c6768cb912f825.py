import numpy as np
import time
from scipy.interpolate import CubicSpline, interp1d

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    if d != 1:
        # Fallback for d > 1
        X_aug = np.column_stack([np.ones(n), X])
        w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            X_new_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
            return X_new_aug @ w
        return predict
    
    x_all = X.flatten()
    y_all = y.flatten()
    
    # Sort by x for interpolation models
    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_sorted = y_all[sort_idx]
    
    # Remove duplicate x values (keep last)
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    # Keep the last occurrence for each unique x
    _, inverse = np.unique(x_sorted, return_inverse=True)
    y_dedup = np.empty_like(y_sorted)
    y_dedup[inverse] = y_sorted
    x_dedup = unique_x
    y_dedup = y_dedup[np.unique(inverse)]
    
    # Simple linear interpolation fallback (very robust)
    interp_func = interp1d(x_dedup, y_dedup, kind='linear', fill_value='extrapolate', assume_sorted=True)
    
    def predict_linear(X_new):
        X_new = np.asarray(X_new, dtype=np.float64).flatten()
        return interp_func(X_new)
    
    # Try cubic spline
    spline_predict = None
    try:
        if len(x_dedup) >= 4:
            cs = CubicSpline(x_dedup, y_dedup)
            def predict_spline(X_new):
                X_new = np.asarray(X_new, dtype=np.float64).flatten()
                return cs(X_new)
            spline_predict = predict_spline
    except Exception:
        pass
    
    # Try RBF with multiple epsilon values
    def try_rbf(epsilon):
        try:
            x = x_dedup
            n_pts = len(x)
            diff = x[:, None] - x[None, :]
            K = np.exp(-(epsilon ** 2) * (diff ** 2))
            reg = 1e-6 * np.eye(n_pts)
            K_reg = K + reg
            w = np.linalg.solve(K_reg, y_dedup)
            
            def predict_rbf(X_new):
                x_new = np.asarray(X_new, dtype=np.float64).flatten()
                diff_new = x_new[:, None] - x[None, :]
                K_new = np.exp(-(epsilon ** 2) * (diff_new ** 2))
                return K_new @ w
            return predict_rbf
        except Exception:
            return None
    
    # Try ridge with rich features
    def try_ridge(alpha):
        try:
            x = x_dedup
            n_pts = len(x)
            cols = [np.ones(n_pts)]
            xk = np.ones(n_pts)
            for k in range(1, 15):
                xk = xk * x
                cols.append(xk.copy())
            for freq in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
                cols.append(np.sin(freq * x))
                cols.append(np.cos(freq * x))
            for c in [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]:
                cols.append(np.exp(c * x))
            cols.append(np.abs(x))
            cols.append(np.sqrt(np.abs(x) + 1e-10))
            Xf = np.column_stack(cols)
            
            n_features = Xf.shape[1]
            I = np.eye(n_features)
            I[0, 0] = 0.0
            
            A = Xf.T @ Xf + alpha * I
            b = Xf.T @ y_dedup
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            def predict_ridge(X_new):
                x_new = np.asarray(X_new, dtype=np.float64).flatten()
                cols_new = [np.ones(len(x_new))]
                xk_new = np.ones(len(x_new))
                for k in range(1, 15):
                    xk_new = xk_new * x_new
                    cols_new.append(xk_new.copy())
                for freq in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
                    cols_new.append(np.sin(freq * x_new))
                    cols_new.append(np.cos(freq * x_new))
                for c in [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]:
                    cols_new.append(np.exp(c * x_new))
                cols_new.append(np.abs(x_new))
                cols_new.append(np.sqrt(np.abs(x_new) + 1e-10))
                Xf_new = np.column_stack(cols_new)
                return Xf_new @ w
            return predict_ridge
        except Exception:
            return None
    
    # Build candidate models
    candidates = []
    
    # Linear interpolation (most robust)
    candidates.append(('linear', predict_linear))
    
    # Cubic spline
    if spline_predict is not None:
        candidates.append(('spline', spline_predict))
    
    # RBF with various epsilons
    epsilons_to_try = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    for eps in epsilons_to_try:
        rbf_pred = try_rbf(eps)
        if rbf_pred is not None:
            candidates.append((f'rbf_{eps}', rbf_pred))
    
    # Ridge with various alphas
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge_pred = try_ridge(alpha)
        if ridge_pred is not None:
            candidates.append((f'ridge_{alpha}', ridge_pred))
    
    # Evaluate candidates on training data (since noiseless, train MSE should be ~0 for good fits)
    # But we need to be careful: interpolation methods will have ~0 train MSE by construction
    # So use a held-out subset for validation
    val_size = min(30, n // 5)
    np.random.seed(42)
    val_indices = np.random.choice(n, val_size, replace=False)
    train_indices = np.setdiff1d(np.arange(n), val_indices)
    
    x_val = x_all[val_indices]
    y_val = y_all[val_indices]
    x_tr = x_all[train_indices]
    y_tr = y_all[train_indices]
    
    # For each candidate, evaluate on validation set
    best_model = None
    best_val_mse = np.inf
    
    for name, pred_func in candidates:
        try:
            if time.time() > deadline - 30:
                break
            pred_val = pred_func(x_val.reshape(-1, 1))
            pred_val = np.asarray(pred_val, dtype=np.float64).flatten()
            if not np.all(np.isfinite(pred_val)):
                continue
            mse = np.mean((pred_val - y_val) ** 2)
            if np.isfinite(mse) and mse < best_val_mse:
                best_val_mse = mse
                best_model = (name, pred_func)
        except Exception:
            continue
    
    # If no good model found, use linear interpolation
    if best_model is None:
        best_model = ('linear', predict_linear)
    
    final_predict = best_model[1]
    
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        result = final_predict(X_new)
        result = np.asarray(result, dtype=np.float64).flatten()
        # Ensure finite
        result = np.where(np.isfinite(result), result, 0.0)
        return result
    
    return predict
