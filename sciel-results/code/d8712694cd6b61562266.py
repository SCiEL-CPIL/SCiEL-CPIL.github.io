# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 305.0  # 25s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    def check_time():
        return time.time() < deadline
    
    best_predict = None
    best_val_mse = np.inf
    
    # Validation split
    val_frac = 0.15
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    X_val = X_train[n_train:]
    y_val = y_train[n_train:]
    X_fit = X_train[:n_train]
    y_fit = y_train[:n_train]
    
    def evaluate_model(model):
        try:
            preds = model(X_val)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y_val.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y_val) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf
    
    def update_best(model):
        nonlocal best_predict, best_val_mse
        if model is None:
            return
        mse = evaluate_model(model)
        if mse < best_val_mse:
            best_val_mse = mse
            best_predict = model
    
    # Normalize inputs
    X_min = X_fit.min(axis=0)
    X_max = X_fit.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    
    def normalize(X):
        return (X - X_min) / X_range
    
    X_fit_norm = normalize(X_fit)
    X_val_norm = normalize(X_val)
    
    # Strategy 1: Polynomial Regression
    def make_poly_model(degree):
        def features(X):
            Xn = normalize(X)
            cols = [np.ones(len(Xn))]
            for j in range(Xn.shape[1]):
                cols.append(Xn[:, j])
            if degree >= 2:
                for i in range(Xn.shape[1]):
                    for j in range(i, Xn.shape[1]):
                        cols.append(Xn[:, i] * Xn[:, j])
            if degree >= 3:
                for i in range(Xn.shape[1]):
                    for j in range(i, Xn.shape[1]):
                        for k in range(j, Xn.shape[1]):
                            cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k])
            if degree >= 4:
                for i in range(Xn.shape[1]):
                    for j in range(i, Xn.shape[1]):
                        for k in range(j, Xn.shape[1]):
                            for l in range(k, Xn.shape[1]):
                                cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k] * Xn[:, l])
            return np.column_stack(cols)
        
        F_fit = features(X_fit)
        try:
            for lam in [1e-8, 1e-6, 1e-4, 1e-2]:
                A = F_fit.T @ F_fit
                b = F_fit.T @ y_fit
                A_reg = A.copy()
                A_reg[1:, 1:] += lam
                try:
                    w = np.linalg.solve(A_reg, b)
                except np.linalg.LinAlgError:
                    w, *_ = np.linalg.lstsq(F_fit, y_fit, rcond=None)
                
                def predict(X):
                    return features(X) @ w
                
                mse = evaluate_model(predict)
                if mse < np.inf:
                    return predict, mse
            return None, np.inf
        except Exception:
            return None, np.inf
    
    # Strategy 2: Sum-of-Sines Kernel Regression (NEW)
    def make_sines_model(max_freq_per_dim=5, cross_terms=True):
        """
        Kernel Ridge Regression using a Sum-of-Sines kernel.
        K(x, x') = sum over frequencies of sin(2*pi*f*x) * sin(2*pi*f*x') + cos terms
        This naturally handles periodic and smooth functions on bounded domains.
        """
        def build_sine_features(Xn, max_freq, include_cross=True):
            """Build sine/cosine feature matrix for given normalized inputs."""
            n_pts = Xn.shape[0]
            cols = [np.ones(n_pts)]
            
            # Single-variable sine/cosine terms
            for j in range(Xn.shape[1]):
                for k in range(1, max_freq + 1):
                    cols.append(np.sin(2 * np.pi * k * Xn[:, j]))
                    cols.append(np.cos(2 * np.pi * k * Xn[:, j]))
            
            # Cross terms: sin(2*pi*k*(xi+xj)), sin(2*pi*k*(xi-xj))
            if include_cross and Xn.shape[1] >= 2:
                max_cross_freq = max(1, max_freq // 2)
                for i in range(Xn.shape[1]):
                    for j in range(i + 1, Xn.shape[1]):
                        for k in range(1, max_cross_freq + 1):
                            cols.append(np.sin(2 * np.pi * k * (Xn[:, i] + Xn[:, j])))
                            cols.append(np.cos(2 * np.pi * k * (Xn[:, i] + Xn[:, j])))
                            cols.append(np.sin(2 * np.pi * k * (Xn[:, i] - Xn[:, j])))
                            cols.append(np.cos(2 * np.pi * k * (Xn[:, i] - Xn[:, j])))
            
            return np.column_stack(cols)
        
        F_fit = build_sine_features(X_fit_norm, max_freq_per_dim, cross_terms)
        
        try:
            for lam in [1e-6, 1e-4, 1e-2, 1e-1]:
                A = F_fit.T @ F_fit
                b = F_fit.T @ y_fit
                A_reg = A.copy()
                A_reg[1:, 1:] += lam
                try:
                    w = np.linalg.solve(A_reg, b)
                except np.linalg.LinAlgError:
                    w, *_ = np.linalg.lstsq(F_fit, y_fit, rcond=None)
                
                def predict(X):
                    Xn = normalize(X)
                    F_test = build_sine_features(Xn, max_freq_per_dim, cross_terms)
                    return F_test @ w
                
                mse = evaluate_model(predict)
                if mse < np.inf:
                    return predict, mse
            return None, np.inf
        except Exception:
            return None, np.inf
    
    # Strategy 3: RBF Kernel Regression
    def make_rbf_model(gamma, n_centers=100):
        if n_train > n_centers:
            idx = np.linspace(0, n_train - 1, n_centers, dtype=int)
            centers = X_fit_norm[idx]
        else:
            centers = X_fit_norm
        
        diffs = X_fit_norm[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        K_fit = np.exp(-gamma * dists)
        
        try:
            for lam in [1e-8, 1e-6, 1e-4]:
                A = K_fit.T @ K_fit
                b = K_fit.T @ y_fit
                A_reg = A.copy()
                A_reg[1:, 1:] += lam
                try:
                    w = np.linalg.solve(A_reg, b)
                except np.linalg.LinAlgError:
                    w, *_ = np.linalg.lstsq(K_fit, y_fit, rcond=None)
                
                def predict(X):
                    Xn = normalize(X)
                    diffs = Xn[:, np.newaxis, :] - centers[np.newaxis, :, :]
                    dists = np.sum(diffs ** 2, axis=2)
                    K_test = np.exp(-gamma * dists)
                    return K_test @ w
                
                mse = evaluate_model(predict)
                if mse < np.inf:
                    return predict, mse
            return None, np.inf
        except Exception:
            return None, np.inf
    
    # Strategy 4: Chebyshev polynomials
    def make_chebyshev_model(degree):
        def features(X):
            Xn = normalize(X)
            cols = [np.ones(len(Xn))]
            for j in range(Xn.shape[1]):
                x = 2 * Xn[:, j] - 1
                T0 = np.ones_like(x)
                T1 = x.copy()
                cols.append(T1)
                for k in range(2, degree + 1):
                    Tk = 2 * x * T1 - T0
                    cols.append(Tk)
                    T0 = T1
                    T1 = Tk
            if Xn.shape[1] >= 2 and Xn.shape[1] <= 3 and degree <= 4:
                for i in range(Xn.shape[1]):
                    for j in range(i + 1, Xn.shape[1]):
                        cols.append(Xn[:, i] * Xn[:, j])
            return np.column_stack(cols)
        
        F_fit = features(X_fit)
        try:
            for lam in [1e-8, 1e-6, 1e-4]:
                A = F_fit.T @ F_fit
                b = F_fit.T @ y_fit
                A_reg = A.copy()
                A_reg[1:, 1:] += lam
                try:
                    w = np.linalg.solve(A_reg, b)
                except np.linalg.LinAlgError:
                    w, *_ = np.linalg.lstsq(F_fit, y_fit, rcond=None)
                
                def predict(X):
                    return features(X) @ w
                
                mse = evaluate_model(predict)
                if mse < np.inf:
                    return predict, mse
            return None, np.inf
        except Exception:
            return None, np.inf
    
    # Run strategies in order of expected effectiveness
    # 1. Polynomials (good for smooth non-periodic)
    for deg in [1, 2, 3, 4]:
        if not check_time():
            break
        model, mse = make_poly_model(deg)
        if model is not None:
            update_best(model)
    
    # 2. Sum-of-Sines (NEW - good for periodic/smooth on bounded domains)
    for mf in [2, 3, 4, 5]:
        if not check_time():
            break
        model, mse = make_sines_model(max_freq_per_dim=mf, cross_terms=True)
        if model is not None:
            update_best(model)
    
    # 3. Chebyshev
    for deg in [3, 5, 7]:
        if not check_time():
            break
        model, mse = make_chebyshev_model(deg)
        if model is not None:
            update_best(model)
    
    # 4. RBF with various gammas
    base_gamma = 1.0 / (d * 1.0)
    gammas = [base_gamma * f for f in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    
    for g in gammas:
        if not check_time():
            break
        for n_c in [50, 100]:
            if not check_time():
                break
            model, mse = make_rbf_model(g, n_centers=n_c)
            if model is not None:
                update_best(model)
    
    # Fallback
    if best_predict is None:
        return lambda X: np.full(X.shape[0], y_mean)
    
    return best_predict

# EVOLVE-BLOCK-END
