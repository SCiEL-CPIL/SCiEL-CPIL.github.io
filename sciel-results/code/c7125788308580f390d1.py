# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 315.0  # 15s margin for 330s kill
    
    n, d = X_train.shape
    
    def check_time():
        return time.time() < deadline
    
    best_predict = None
    best_val_mse = np.inf
    
    def safe_evaluate(model, X, y, val_frac=0.2):
        if n >= 20:
            val_idx = np.arange(int(n * (1 - val_frac)), n)
            tr_idx = np.arange(0, int(n * (1 - val_frac)))
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_val, y_val = X, y
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
        mse = safe_evaluate(model, X_train, y_train)
        if mse < best_val_mse:
            best_val_mse = mse
            best_predict = model
    
    # 1. Polynomial Regression (degrees 1-4)
    def make_poly_model(degree):
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            if degree >= 3:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        for k in range(j, X.shape[1]):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
            if degree >= 4:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        for k in range(j, X.shape[1]):
                            for l in range(k, X.shape[1]):
                                cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
            return np.column_stack(cols)
        
        F_train = features(X_train)
        try:
            # Ridge regression for stability
            lam = 1e-6
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        
        def predict(X):
            return features(X) @ w
        return predict
    
    # 2. Fourier Regression
    def make_fourier_model(max_freq):
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                for k in range(1, max_freq + 1):
                    cols.append(np.sin(k * X[:, j]))
                    cols.append(np.cos(k * X[:, j]))
            # Add cross terms for low dimensions if possible
            if X.shape[1] >= 2 and X.shape[1] <= 3:
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        cols.append(np.sin(X[:, i] + X[:, j]))
                        cols.append(np.cos(X[:, i] + X[:, j]))
                        cols.append(np.sin(X[:, i] - X[:, j]))
                        cols.append(np.cos(X[:, i] - X[:, j]))
            return np.column_stack(cols)
        
        F_train = features(X_train)
        try:
            lam = 1e-6
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        
        def predict(X):
            return features(X) @ w
        return predict
    
    # 3. RBF Regression
    def make_rbf_model(gamma, n_centers=50):
        # Select centers
        if n > n_centers:
            idx = np.linspace(0, n - 1, n_centers, dtype=int)
            centers = X_train[idx]
        else:
            centers = X_train
        
        # Precompute distances for training
        # F[i, j] = exp(-gamma * ||X_i - c_j||^2)
        # To avoid memory issues with large n, we compute in chunks or just use standard broadcasting if n is small
        # n=200 is small, so direct computation is fine
        diffs = X_train[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        F_train = np.exp(-gamma * dists)
        
        try:
            lam = 1e-6
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        
        def predict(X):
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)
            F_test = np.exp(-gamma * dists)
            return F_test @ w
        return predict
    
    # 4. Spline-like basis (Chebyshev)
    def make_chebyshev_model(degree):
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                # Normalize x to [-1, 1] for Chebyshev stability
                x_min = X[:, j].min()
                x_max = X[:, j].max()
                if x_max > x_min:
                    x_norm = (2 * X[:, j] - (x_max + x_min)) / (x_max - x_min)
                else:
                    x_norm = np.zeros_like(X[:, j])
                
                T0 = np.ones_like(x_norm)
                T1 = x_norm
                cols.append(T1)
                for k in range(2, degree + 1):
                    Tk = 2 * x_norm * T1 - T0
                    cols.append(Tk)
                    T0 = T1
                    T1 = Tk
            return np.column_stack(cols)
        
        F_train = features(X_train)
        try:
            lam = 1e-6
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        
        def predict(X):
            return features(X) @ w
        return predict

    # --- Strategy ---
    
    # 1. Try Polynomials
    for deg in [1, 2, 3, 4]:
        if not check_time():
            break
        model = make_poly_model(deg)
        update_best(model)
        
    # 2. Try Chebyshev
    for deg in [3, 5, 7]:
        if not check_time():
            break
        model = make_chebyshev_model(deg)
        update_best(model)
        
    # 3. Try Fourier
    for freq in [2, 4, 6]:
        if not check_time():
            break
        model = make_fourier_model(freq)
        update_best(model)
        
    # 4. Try RBF with various gammas
    # Estimate scale
    X_range = np.ptp(X_train, axis=0)
    avg_range = np.mean(X_range)
    if avg_range == 0:
        avg_range = 1.0
    
    base_gamma = 1.0 / (d * avg_range ** 2)
    gammas = [base_gamma * 0.01, base_gamma * 0.1, base_gamma * 0.5, base_gamma, base_gamma * 2, base_gamma * 5, base_gamma * 10]
    
    for g in gammas:
        if not check_time():
            break
        model = make_rbf_model(g, n_centers=50)
        update_best(model)
        
    # 5. If we have time, try a more sophisticated RBF with more centers
    if check_time():
        for g in [base_gamma * 0.5, base_gamma, base_gamma * 2]:
            if not check_time():
                break
            model = make_rbf_model(g, n_centers=100)
            update_best(model)

    # Fallback
    if best_predict is None:
        y_mean = np.mean(y_train)
        return lambda X: np.full(X.shape[0], y_mean)
        
    return best_predict

# EVOLVE-BLOCK-END
