import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize_scalar

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 270  # Safety margin for 4 equations (300s total)
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    d = X.shape[1]
    
    def time_left():
        return deadline - time.time()
        
    if d == 1:
        x = X[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Check for duplicate x values, if so, average y (though noiseless implies unique x usually)
        # For interpolation, we need strictly increasing x.
        # If duplicates exist, we should handle them.
        # Since it's noiseless, duplicates likely have same y.
        # Let's ensure unique x for splines.
        unique_x_mask = np.concatenate(([True], np.diff(x_sorted) > 1e-12))
        x_unique = x_sorted[unique_x_mask]
        # If we have duplicates, we need to average y for unique x to be safe, 
        # but with noiseless data, we can just take the first or average.
        # Let's average y for any duplicate x just in case.
        if not np.all(unique_x_mask):
            # Group by x
            unique_x_vals, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
            y_unique = np.zeros(len(unique_x_vals))
            np.add.at(y_unique, inverse_indices, y_sorted)
            y_unique /= counts
            x_sorted = unique_x_vals
            y_sorted = y_unique
            n = len(x_sorted)
        else:
            x_sorted = x_unique
            # y_sorted is already aligned with x_sorted from the sort step
            # But wait, if unique_x_mask is all True, x_sorted is x_unique.
            # y_sorted is y[sort_idx]. This is correct.
            
        if n < 4:
            # Fallback to linear or constant
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                x_pred = X_pred[:, 0]
                # Linear interpolation
                return np.interp(x_pred, x_sorted, y_sorted)
            return predict

        # Hold-out validation
        rng = np.random.RandomState(42)
        indices = np.arange(n)
        rng.shuffle(indices)
        split = int(0.8 * n)
        idx_train = indices[:split]
        idx_val = indices[split:]
        
        x_tr = x_sorted[idx_train]
        y_tr = y_sorted[idx_train]
        x_val = x_sorted[idx_val]
        y_val = y_sorted[idx_val]
        
        # Sort training subset for interpolation
        sort_tr = np.argsort(x_tr)
        x_tr_s = x_tr[sort_tr]
        y_tr_s = y_tr[sort_tr]
        
        candidates = []
        
        # 1. PCHIP
        try:
            pchip = PchipInterpolator(x_tr_s, y_tr_s)
            y_pred_val = pchip(x_val)
            mse_pchip = np.mean((y_pred_val - y_val)**2)
            candidates.append(('pchip', mse_pchip, pchip))
        except:
            pass
            
        # 2. Cubic Spline (not-a-knot)
        try:
            cs_nak = CubicSpline(x_tr_s, y_tr_s, bc_type='not-a-knot')
            y_pred_val = cs_nak(x_val)
            mse_nak = np.mean((y_pred_val - y_val)**2)
            candidates.append(('cs_nak', mse_nak, cs_nak))
        except:
            pass
            
        # 3. Cubic Spline (parabolic)
        try:
            cs_par = CubicSpline(x_tr_s, y_tr_s, bc_type='parabolic')
            y_pred_val = cs_par(x_val)
            mse_par = np.mean((y_pred_val - y_val)**2)
            candidates.append(('cs_par', mse_par, cs_par))
        except:
            pass
            
        # 4. KRR with RBF (Median distance heuristic)
        # Estimate length scale
        if len(x_tr_s) > 1:
            diffs = np.diff(x_tr_s)
            med_diff = np.median(diffs)
            if med_diff < 1e-6:
                med_diff = 1.0
            # Try a few scales around median
            best_krr_mse = np.inf
            best_krr_model = None
            for scale_factor in [0.5, 1.0, 2.0]:
                length_scale = med_diff * scale_factor
                gamma = 0.5 / (length_scale**2)
                # Build kernel
                K_tr = np.exp(-gamma * (x_tr_s[:, None] - x_tr_s[None, :])**2)
                lam = 1e-6
                A = K_tr + lam * np.eye(len(x_tr_s))
                try:
                    w = np.linalg.solve(A, y_tr_s)
                    K_val = np.exp(-gamma * (x_val[:, None] - x_tr_s[None, :])**2)
                    y_pred_val = K_val @ w
                    mse = np.mean((y_pred_val - y_val)**2)
                    if mse < best_krr_mse:
                        best_krr_mse = mse
                        # Store parameters for refit
                        best_krr_params = (gamma, x_tr_s, y_tr_s, w, lam)
                        # We can't easily store the model object for refit without rebuilding, 
                        # so we store the type and params
                        candidates.append(('krr', mse, (gamma, lam)))
                except:
                    pass
        else:
            pass

        if not candidates:
            # Fallback to nearest neighbor or linear
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                x_pred = X_pred[:, 0]
                return np.interp(x_pred, x_sorted, y_sorted)
            return predict

        # Sort candidates by MSE
        candidates.sort(key=lambda x: x[1])
        best_type = candidates[0][0]
        
        # Refit on all data
        if best_type == 'pchip':
            model = PchipInterpolator(x_sorted, y_sorted)
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                return model(X_pred[:, 0])
                
        elif best_type == 'cs_nak':
            model = CubicSpline(x_sorted, y_sorted, bc_type='not-a-knot')
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                return model(X_pred[:, 0])
                
        elif best_type == 'cs_par':
            model = CubicSpline(x_sorted, y_sorted, bc_type='parabolic')
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                return model(X_pred[:, 0])
                    
        elif best_type == 'krr':
            gamma, lam = candidates[0][2]
            K_all = np.exp(-gamma * (x_sorted[:, None] - x_sorted[None, :])**2)
            A_all = K_all + lam * np.eye(len(x_sorted))
            try:
                w_final = np.linalg.solve(A_all, y_sorted)
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    x_pred = X_pred[:, 0]
                    K_pred = np.exp(-gamma * (x_pred[:, None] - x_sorted[None, :])**2)
                    return K_pred @ w_final
            except:
                # Fallback
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    return np.interp(X_pred[:, 0], x_sorted, y_sorted)
        else:
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                return np.interp(X_pred[:, 0], x_sorted, y_sorted)
                
        return predict
        
    else:
        # Multi-dimensional fallback: KRR with RBF
        def rbf_kernel_md(X1, X2, gamma):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
            np.maximum(sq_dists, 0, out=sq_dists)
            return np.exp(-gamma * sq_dists)
        
        rng = np.random.RandomState(42)
        indices = np.arange(n)
        rng.shuffle(indices)
        split = int(0.8 * n)
        idx_train = indices[:split]
        idx_val = indices[split:]
        
        X_tr = X[idx_train]
        y_tr = y[idx_train]
        X_val = X[idx_val]
        y_val = y[idx_val]
        
        best_val_mse = np.inf
        best_gamma = 1.0
        best_alpha = None
        
        gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        for gamma in gammas:
            if time_left() < 5:
                break
            try:
                K_tr = rbf_kernel_md(X_tr, X_tr, gamma)
                lam = 1e-4
                A = K_tr + lam * np.eye(len(X_tr))
                alpha_vec = np.linalg.solve(A, y_tr)
                
                K_val = rbf_kernel_md(X_val, X_tr, gamma)
                y_pred_val = K_val @ alpha_vec
                mse = np.mean((y_pred_val - y_val)**2)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_gamma = gamma
                    best_alpha = alpha_vec
            except:
                continue
        
        # Refit on all data
        gamma = best_gamma
        K_all = rbf_kernel_md(X, X, gamma)
        lam = 1e-4
        A_all = K_all + lam * np.eye(n)
        try:
            alpha_final = np.linalg.solve(A_all, y)
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                K_pred = rbf_kernel_md(X_pred, X, gamma)
                return K_pred @ alpha_final
        except:
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                return np.full(len(X_pred), np.mean(y))
        
        return predict

# EVOLVE-BLOCK-END
