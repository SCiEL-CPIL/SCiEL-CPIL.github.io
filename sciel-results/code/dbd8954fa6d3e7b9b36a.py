# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    def check_time():
        return time.time() < deadline
    
    # Normalize features for numerical stability
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    # Normalize y
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    def safe_evaluate(model, X, y):
        try:
            preds = model(X)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf
    
    best_predict = None
    best_mse = np.inf
    
    def update_best(model):
        nonlocal best_predict, best_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_train, y_train)
        if mse < best_mse:
            best_mse = mse
            best_predict = model
    
    def ridge_solve(F, target, lam=1e-5):
        A = F.T @ F + lam * np.eye(F.shape[1])
        A[0, 0] -= lam  # Don't regularize intercept
        return solve(A, F.T @ target)
    
    # Structural change: Use a proper Sum-of-Sines kernel ridge regression
    # with a dense, structured frequency grid. This creates a smooth, periodic
    # basis that naturally handles bounded domains and captures interactions.
    
    def make_fourier_features(X, max_freq, include_interactions=True):
        """
        Create Fourier basis features with a dense frequency grid.
        For each dimension, we use sin and cos at frequencies pi, 2pi, ..., max_freq*pi.
        For interactions, we use sin/cos of linear combinations of pairs.
        """
        Xn = (X - X_min) / X_range
        cols = [np.ones(len(Xn))]
        
        # Main terms: sin(k*pi*x_j) and cos(k*pi*x_j) for each dim j
        for j in range(Xn.shape[1]):
            x = Xn[:, j]
            for k in range(1, max_freq + 1):
                freq = k * np.pi
                cols.append(np.sin(freq * x))
                cols.append(np.cos(freq * x))
        
        # Interaction terms for pairs of dimensions
        if include_interactions and Xn.shape[1] >= 2:
            max_int_freq = min(max_freq, 4)
            for i in range(Xn.shape[1]):
                for j in range(i + 1, Xn.shape[1]):
                    for k in range(1, max_int_freq + 1):
                        freq = k * np.pi
                        # Sum interaction
                        s = Xn[:, i] + Xn[:, j]
                        cols.append(np.sin(freq * s))
                        cols.append(np.cos(freq * s))
                        # Difference interaction
                        diff = Xn[:, i] - Xn[:, j]
                        cols.append(np.sin(freq * diff))
                        cols.append(np.cos(freq * diff))
                        # Product interaction
                        prod = Xn[:, i] * Xn[:, j]
                        cols.append(np.sin(freq * prod))
                        cols.append(np.cos(freq * prod))
        
        # Triple interactions for 3+ dimensions
        if include_interactions and Xn.shape[1] >= 3:
            max_triple_freq = min(max_freq, 2)
            for i in range(Xn.shape[1]):
                for j in range(i + 1, Xn.shape[1]):
                    for k in range(j + 1, Xn.shape[1]):
                        for freq_k in range(1, max_triple_freq + 1):
                            freq = freq_k * np.pi
                            # Sum of three
                            s3 = Xn[:, i] + Xn[:, j] + Xn[:, k]
                            cols.append(np.sin(freq * s3))
                            cols.append(np.cos(freq * s3))
        
        return np.column_stack(cols)
    
    # Try a range of frequencies and regularization levels
    freq_candidates = [2, 3, 4, 5, 6, 8, 10]
    lam_candidates = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for max_freq in freq_candidates:
        if not check_time():
            break
        for lam in lam_candidates:
            if not check_time():
                break
            try:
                F = make_fourier_features(X_train, max_freq, include_interactions=True)
                if F.shape[1] > 500:  # Cap feature dimension for stability
                    continue
                w = ridge_solve(F, y_norm, lam)
                
                def predict_fourier(X, w, y_mean, y_std, mf, inter=True):
                    F_test = make_fourier_features(X, mf, include_interactions=inter)
                    return y_mean + y_std * (F_test @ w)
                
                update_best(lambda X, w=w, mf=max_freq: predict_fourier(X, w, y_mean, y_std, mf, True))
            except Exception:
                pass
    
    # Also try without interactions to see if simpler model works better for some cases
    for max_freq in [3, 5, 8]:
        if not check_time():
            break
        for lam in [1e-4, 1e-3, 1e-2]:
            if not check_time():
                break
            try:
                F = make_fourier_features(X_train, max_freq, include_interactions=False)
                w = ridge_solve(F, y_norm, lam)
                
                def predict_fourier_no_int(X, w, y_mean, y_std, mf):
                    F_test = make_fourier_features(X, mf, include_interactions=False)
                    return y_mean + y_std * (F_test @ w)
                
                update_best(lambda X, w=w, mf=max_freq: predict_fourier_no_int(X, w, y_mean, y_std, mf))
            except Exception:
                pass
    
    # Fallback: simple linear model
    if best_predict is None:
        Xn = (X_train - X_min) / X_range
        F = np.column_stack([np.ones(n), Xn])
        w = ridge_solve(F, y_norm, 1e-5)
        
        def predict_linear(X, w, y_mean, y_std):
            Xn = (X - X_min) / X_range
            F_test = np.column_stack([np.ones(len(Xn)), Xn])
            return y_mean + y_std * (F_test @ w)
        
        return lambda X: predict_linear(X, w, y_mean, y_std)
    
    return best_predict

# EVOLVE-BLOCK-END
