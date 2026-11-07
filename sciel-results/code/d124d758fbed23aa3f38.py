import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import cho_factor, cho_solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Flatten X if needed, though task says 1D inputs
    if X_train.ndim == 2:
        x = X_train[:, 0]
    else:
        x = X_train
    y = y_train
    
    # Sort by x for interpolation methods
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Define candidate models
    # Each model is a function that takes (x_train, y_train) and returns a predict function
    candidates = []
    
    # 1. Cubic Spline
    def make_spline_predict(x_tr, y_tr):
        # Handle potential duplicate x values by averaging y
        unique_x, unique_indices = np.unique(x_tr, return_index=True)
        if len(unique_x) != len(x_tr):
            y_agg = np.zeros_like(unique_x)
            counts = np.zeros_like(unique_x)
            np.add.at(y_agg, np.searchsorted(unique_x, x_tr), y_tr)
            np.add.at(counts, np.searchsorted(unique_x, x_tr), 1)
            y_tr = y_agg / counts
            x_tr = unique_x
        try:
            cs = CubicSpline(x_tr, y_tr)
            return lambda X: cs(X[:, 0] if X.ndim == 2 else X)
        except Exception:
            # Fallback to linear if spline fails
            return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
    candidates.append(("spline", make_spline_predict))
    
    # 2. PCHIP
    def make_pchip_predict(x_tr, y_tr):
        unique_x, unique_indices = np.unique(x_tr, return_index=True)
        if len(unique_x) != len(x_tr):
            y_agg = np.zeros_like(unique_x)
            counts = np.zeros_like(unique_x)
            np.add.at(y_agg, np.searchsorted(unique_x, x_tr), y_tr)
            np.add.at(counts, np.searchsorted(unique_x, x_tr), 1)
            y_tr = y_agg / counts
            x_tr = unique_x
        try:
            pchip = PchipInterpolator(x_tr, y_tr)
            return lambda X: pchip(X[:, 0] if X.ndim == 2 else X)
        except Exception:
            return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
    candidates.append(("pchip", make_pchip_predict))
    
    # 3. Gaussian Process with specific length scales
    def make_gp_predict(x_tr, y_tr, length_scale):
        # Kernel: RBF
        # K_ij = exp(-0.5 * (x_i - x_j)^2 / l^2)
        # We assume variance sigma^2 = 1 for simplicity, or scale y first?
        # Scaling y helps.
        y_mean = np.mean(y_tr)
        y_std = np.std(y_tr)
        if y_std < 1e-8:
            y_std = 1.0
        y_scaled = (y_tr - y_mean) / y_std
        
        X_k = x_tr[:, np.newaxis]
        diff = X_k - X_k.T
        K = np.exp(-0.5 * (diff**2) / (length_scale**2))
        
        # Add small jitter for numerical stability
        K += 1e-8 * np.eye(len(x_tr))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
        except np.linalg.LinAlgError:
            # Fallback to least squares if Cholesky fails
            try:
                alpha = np.linalg.lstsq(K, y_scaled, rcond=None)[0]
            except Exception:
                return lambda X: np.full(len(X), y_mean)
        
        def predict(X):
            x_new = X[:, 0] if X.ndim == 2 else X
            diff_new = x_new[:, np.newaxis] - X_k
            K_star = np.exp(-0.5 * (diff_new**2) / (length_scale**2))
            y_pred_scaled = K_star @ alpha
            return y_pred_scaled * y_std + y_mean
            
        return predict

    # Add a few GP candidates with different length scales
    # Estimate typical scale from data
    x_range = np.max(x_sorted) - np.min(x_sorted)
    if x_range > 0:
        typical_scale = x_range / 10.0
        for ls_factor in [0.5, 1.0, 2.0, 5.0]:
            ls = typical_scale * ls_factor
            candidates.append((f"gp_{ls:.2f}", lambda x_tr, y_tr, ls=ls: make_gp_predict(x_tr, y_tr, ls)))
    else:
        candidates.append(("gp_1.0", lambda x_tr, y_tr: make_gp_predict(x_tr, y_tr, 1.0)))

    # 4. Polynomial with Ridge (Degrees 4, 6, 8)
    def make_poly_predict(x_tr, y_tr, degree, alpha):
        # Vandermonde matrix
        V = np.vander(x_tr, N=degree+1, increasing=True)
        # Ridge: (V^T V + alpha I)^-1 V^T y
        try:
            A = V.T @ V + alpha * np.eye(degree+1)
            b = V.T @ y_tr
            w = np.linalg.solve(A, b)
        except Exception:
            w = np.zeros(degree+1)
        
        def predict(X):
            x_new = X[:, 0] if X.ndim == 2 else X
            V_new = np.vander(x_new, N=degree+1, increasing=True)
            return V_new @ w
        return predict

    for deg in [4, 6, 8]:
        for alpha in [1e-2, 1e-1, 1.0]:
             candidates.append((f"poly_{deg}_{alpha}", lambda x_tr, y_tr, d=deg, a=alpha: make_poly_predict(x_tr, y_tr, d, a)))

    # Cross-Validation to select the best model
    n = len(x)
    k_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, k_folds)
    
    model_scores = {name: [] for name, _ in candidates}
    
    for i in range(k_folds):
        if time.time() > deadline:
            break
        val_idx = fold_indices[i]
        tr_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
        
        x_val = x[val_idx]
        y_val = y[val_idx]
        x_tr = x[tr_idx]
        y_tr = y[tr_idx]
        
        for name, model_func in candidates:
            if time.time() > deadline:
                break
            try:
                predict_func = model_func(x_tr, y_tr)
                y_pred = predict_func(x_val.reshape(-1, 1))
                mse = np.mean((y_pred - y_val)**2)
                model_scores[name].append(mse)
            except Exception:
                model_scores[name].append(np.inf)
    
    # Select model with lowest mean CV MSE
    best_name = None
    best_score = np.inf
    for name, scores in model_scores.items():
        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_name = name
    
    if best_name is None:
        # Fallback to simple linear interp
        return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_sorted, y_sorted)
        
    # Find the function for the best model
    best_model_func = None
    for name, func in candidates:
        if name == best_name:
            best_model_func = func
            break
            
    if best_model_func is None:
        return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_sorted, y_sorted)

    # Train the best model on ALL data
    final_predict = best_model_func(x, y)
    
    # Wrap to ensure correct output shape and finite values
    def final_wrapper(X):
        try:
            pred = final_predict(X)
            pred = np.asarray(pred).flatten()
            # Ensure finite
            pred = np.nan_to_num(pred, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
            return pred
        except Exception:
            return np.zeros(len(X))
            
    return final_wrapper
