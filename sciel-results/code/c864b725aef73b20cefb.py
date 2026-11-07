import numpy as np
import time
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # seconds, safety margin before 300s kill
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # If only 1 input dimension, we can use more powerful 1D methods
    if d == 1:
        x = X.flatten()
        y_flat = y.flatten()
        
        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y_flat[sort_idx]
        
        # Handle duplicates
        unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        if len(unique_x) != len(x_sorted):
            y_sum = np.zeros_like(unique_x, dtype=float)
            np.add.at(y_sum, inverse_indices, y_sorted)
            y_sorted = y_sum / counts
            x_sorted = unique_x
        
        # Try multiple basis functions and select via cross-validation
        candidates = []
        
        # 1. Polynomial basis (degrees 2-8)
        for deg in [2, 3, 4, 5, 6, 8]:
            try:
                F = np.vander(x_sorted, N=deg+1, increasing=True)
                candidates.append(('poly_%d' % deg, F))
            except Exception:
                pass
        
        # 2. Fourier basis
        for nf in [4, 6, 8, 10, 12]:
            try:
                cols = [np.ones(len(x_sorted))]
                for k in range(1, nf + 1):
                    cols.append(np.sin(k * x_sorted))
                    cols.append(np.cos(k * x_sorted))
                F = np.column_stack(cols)
                candidates.append(('fourier_%d' % nf, F))
            except Exception:
                pass
        
        # 3. RBF basis with various numbers of centers
        x_min, x_max = np.min(x_sorted), np.max(x_sorted)
        x_range = x_max - x_min
        if x_range > 0:
            for nc in [10, 15, 20, 25, 30]:
                try:
                    centers = np.linspace(x_min, x_max, nc)
                    bandwidth = x_range / nc * 1.5
                    cols = [np.ones(len(x_sorted))]
                    for c in centers:
                        cols.append(np.exp(-((x_sorted - c) ** 2) / (2 * bandwidth ** 2)))
                    F = np.column_stack(cols)
                    candidates.append(('rbf_%d' % nc, F))
                except Exception:
                    pass
        
        # 4. B-spline-like piecewise linear basis
        for nk in [10, 15, 20, 25]:
            try:
                knots = np.linspace(x_min, x_max, nk + 1)
                cols = [np.ones(len(x_sorted))]
                for i in range(len(knots) - 1):
                    left = knots[i]
                    right = knots[i + 1]
                    center = (left + right) / 2
                    width = (right - left) / 2
                    if width > 0:
                        feat = np.clip(1 - np.abs(x_sorted - center) / width, 0, 1)
                        cols.append(feat)
                F = np.column_stack(cols)
                candidates.append(('spline_%d' % nk, F))
            except Exception:
                pass
        
        # Cross-validation to select best basis
        best_mse = np.inf
        best_w = None
        best_F = None
        best_name = None
        
        n_folds = 5
        indices = np.arange(len(x_sorted))
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(indices) // n_folds
        
        for name, F in candidates:
            if time.time() > deadline - 15:
                break
            try:
                mse_sum = 0.0
                for i in range(n_folds):
                    val_start = i * fold_size
                    val_end = val_start + fold_size if i < n_folds - 1 else len(indices)
                    val_idx = indices[val_start:val_end]
                    train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                    
                    X_train_cv = F[train_idx]
                    y_train_cv = y_sorted[train_idx]
                    X_val_cv = F[val_idx]
                    y_val_cv = y_sorted[val_idx]
                    
                    # Fit with ridge regularization
                    reg = 1e-6
                    p = X_train_cv.shape[1]
                    A = X_train_cv.T @ X_train_cv + reg * np.eye(p)
                    b = X_train_cv.T @ y_train_cv
                    try:
                        w_cv = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        w_cv = np.linalg.lstsq(X_train_cv, y_train_cv, rcond=None)[0]
                    
                    y_pred_cv = X_val_cv @ w_cv
                    mse_sum += np.mean((y_pred_cv - y_val_cv) ** 2)
                
                mse_avg = mse_sum / n_folds
                if mse_avg < best_mse:
                    best_mse = mse_avg
                    best_F = F
                    best_name = name
            except Exception:
                continue
        
        # Fit final model on all data with best basis
        if best_F is not None:
            reg = 1e-6
            p = best_F.shape[1]
            A = best_F.T @ best_F + reg * np.eye(p)
            b = best_F.T @ y_sorted
            try:
                best_w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                best_w = np.linalg.lstsq(best_F, y_sorted, rcond=None)[0]
            
            # Store parameters for prediction
            x_min_pred = x_min
            x_max_pred = x_max
            x_range_pred = x_range
            
            def predict(X_new):
                x_eval = np.asarray(X_new, dtype=float).flatten()
                try:
                    if 'poly' in best_name:
                        deg = int(best_name.split('_')[1])
                        F_new = np.vander(x_eval, N=deg+1, increasing=True)
                    elif 'fourier' in best_name:
                        nf = int(best_name.split('_')[1])
                        cols = [np.ones(len(x_eval))]
                        for k in range(1, nf + 1):
                            cols.append(np.sin(k * x_eval))
                            cols.append(np.cos(k * x_eval))
                        F_new = np.column_stack(cols)
                    elif 'rbf' in best_name:
                        nc = int(best_name.split('_')[1])
                        centers = np.linspace(x_min_pred, x_max_pred, nc)
                        bandwidth = x_range_pred / nc * 1.5
                        cols = [np.ones(len(x_eval))]
                        for c in centers:
                            cols.append(np.exp(-((x_eval - c) ** 2) / (2 * bandwidth ** 2)))
                        F_new = np.column_stack(cols)
                    elif 'spline' in best_name:
                        nk = int(best_name.split('_')[1])
                        knots = np.linspace(x_min_pred, x_max_pred, nk + 1)
                        cols = [np.ones(len(x_eval))]
                        for i in range(len(knots) - 1):
                            left = knots[i]
                            right = knots[i + 1]
                            center = (left + right) / 2
                            width = (right - left) / 2
                            if width > 0:
                                feat = np.clip(1 - np.abs(x_eval - center) / width, 0, 1)
                                cols.append(feat)
                        F_new = np.column_stack(cols)
                    else:
                        F_new = np.column_stack([np.ones(len(x_eval)), x_eval])
                    
                    y_pred = F_new @ best_w
                    return y_pred.reshape(np.asarray(X_new).shape)
                except Exception:
                    # Fallback to mean
                    return np.full(np.asarray(X_new).shape, np.mean(y_sorted))
            
            return predict
    
    # For multi-dimensional case or fallback: use polynomial regression
    # Try degrees 2-5 with cross-validation
    best_mse = np.inf
    best_w = None
    best_degree = 2
    
    for deg in [2, 3, 4, 5]:
        try:
            # Generate polynomial features
            cols = [np.ones(n)]
            for i in range(d):
                cols.append(X[:, i])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X[:, i] * X[:, j])
            if deg >= 3:
                for i in range(d):
                    for j in range(d):
                        for k in range(d):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
            
            F = np.column_stack(cols)
            
            # Quick cross-validation
            n_folds = 5
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            fold_size = n // n_folds
            
            mse_sum = 0.0
            for i in range(n_folds):
                val_start = i * fold_size
                val_end = val_start + fold_size if i < n_folds - 1 else n
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                X_train_cv = F[train_idx]
                y_train_cv = y[train_idx]
                X_val_cv = F[val_idx]
                y_val_cv = y[val_idx]
                
                reg = 1e-6
                p = X_train_cv.shape[1]
                A = X_train_cv.T @ X_train_cv + reg * np.eye(p)
                b = X_train_cv.T @ y_train_cv
                try:
                    w_cv = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w_cv = np.linalg.lstsq(X_train_cv, y_train_cv, rcond=None)[0]
                
                y_pred_cv = X_val_cv @ w_cv
                mse_sum += np.mean((y_pred_cv - y_val_cv) ** 2)
            
            mse_avg = mse_sum / n_folds
            if mse_avg < best_mse:
                best_mse = mse_avg
                best_degree = deg
        except Exception:
            continue
    
    # Final fit
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    if best_degree >= 3:
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    
    F = np.column_stack(cols)
    reg = 1e-6
    p = F.shape[1]
    A = F.T @ F + reg * np.eye(p)
    b = F.T @ y
    try:
        best_w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        best_w = np.linalg.lstsq(F, y, rcond=None)[0]
    
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        cols = [np.ones(X_new.shape[0])]
        for i in range(d):
            cols.append(X_new[:, i])
        for i in range(d):
            for j in range(i, d):
                cols.append(X_new[:, i] * X_new[:, j])
        if best_degree >= 3:
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        cols.append(X_new[:, i] * X_new[:, j] * X_new[:, k])
        F_new = np.column_stack(cols)
        return F_new @ best_w
    
    return predict
