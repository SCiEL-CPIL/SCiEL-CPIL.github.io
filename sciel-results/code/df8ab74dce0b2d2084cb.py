import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 70  # 70s per equation, 4 equations = 280s, leaves 20s buffer
    
    X = X_train.reshape(-1, 1)
    y = y_train.reshape(-1)
    n = len(y)
    
    # Standardize
    X_mean = np.mean(X)
    X_std = np.std(X)
    if X_std < 1e-8:
        X_std = 1.0
    X_scaled = (X - X_mean) / X_std
    
    # Try multiple approaches and pick the best
    
    # Approach 1: Polynomial regression with increasing degree
    best_predict = None
    best_score = np.inf
    
    # Compute R^2 for a prediction
    def compute_score(pred):
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        r2 = 1 - ss_res / ss_tot
        return 1 - r2  # Lower is better
    
    # Try polynomial degrees
    for degree in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]:
        if time.time() > deadline:
            break
        
        # Build Vandermonde matrix
        Xv = np.vander(X_scaled, N=degree + 1, increasing=True)
        
        try:
            # Ridge regression for stability
            alpha_ridge = 1e-6
            A = Xv.T @ Xv + alpha_ridge * np.eye(degree + 1)
            b = Xv.T @ y
            w = np.linalg.solve(A, b)
            
            # Predict
            pred_train = Xv @ w
            
            # Score
            score = compute_score(pred_train)
            
            if score < best_score:
                best_score = score
                best_w = w.copy()
                best_degree = degree
        except np.linalg.LinAlgError:
            continue
    
    # Approach 2: Fourier series
    for n_terms in [3, 5, 8, 10, 15, 20, 30, 40]:
        if time.time() > deadline:
            break
        
        # Fourier features
        Xf = np.ones((n, 1))
        for k in range(1, n_terms + 1):
            Xf = np.column_stack([Xf, np.sin(k * np.pi * X_scaled), np.cos(k * np.pi * X_scaled)])
        
        try:
            alpha_ridge = 1e-6
            A = Xf.T @ Xf + alpha_ridge * np.eye(Xf.shape[1])
            b = Xf.T @ y
            w = np.linalg.solve(A, b)
            
            pred_train = Xf @ w
            score = compute_score(pred_train)
            
            if score < best_score:
                best_score = score
                best_w = w.copy()
                best_fourier = n_terms
        except np.linalg.LinAlgError:
            continue
    
    # Approach 3: Gaussian Process with optimized hyperparameters
    if time.time() < deadline - 10:
        from scipy.optimize import minimize_scalar
        
        # Matérn 5/2 kernel
        def matern52(X1, X2, length_scale, variance):
            d = np.abs(X1 - X2)
            sqrt3d = np.sqrt(3) * d / length_scale
            K = variance * (1 + sqrt3d + (sqrt3d ** 2) / 3) * np.exp(-sqrt3d)
            return K
        
        def log_marginal_likelihood(params):
            length_scale = np.exp(params[0])
            variance = np.exp(params[1])
            noise = np.exp(params[2])
            
            K = matern52(X_scaled, X_scaled, length_scale, variance)
            K += noise * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                lml = -0.5 * y.dot(alpha) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
                return -lml
            except np.linalg.LinAlgError:
                return 1e10
        
        # Optimize hyperparameters
        x0 = np.array([0.0, np.log(np.var(y) + 1e-6), np.log(1e-6)])
        
        try:
            from scipy.optimize import minimize
            res = minimize(log_marginal_likelihood, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100})
            
            if time.time() < deadline - 5:
                length_scale = np.exp(res.x[0])
                variance = np.exp(res.x[1])
                noise = np.exp(res.x[2])
                
                K_train = matern52(X_scaled, X_scaled, length_scale, variance)
                K_train += noise * np.eye(n)
                
                try:
                    L = np.linalg.cholesky(K_train)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    
                    pred_train = matern52(X_scaled, X_scaled, length_scale, variance) @ alpha
                    score = compute_score(pred_train)
                    
                    if score < best_score:
                        best_score = score
                        best_gp = (length_scale, variance, noise, alpha)
                except np.linalg.LinAlgError:
                    pass
        except Exception:
            pass
    
    # Build the best predictor
    if 'best_gp' in dir():
        length_scale, variance, noise, alpha = best_gp
        
        def predict(X_pred):
            X_pred = X_pred.reshape(-1, 1)
            X_scaled_pred = (X_pred - X_mean) / X_std
            K_star = matern52(X_scaled_pred, X_scaled, length_scale, variance)
            mu = K_star @ alpha
            return mu
    elif 'best_fourier' in dir():
        n_terms = best_fourier
        w = best_w
        
        def predict(X_pred):
            X_pred = X_pred.reshape(-1, 1)
            X_scaled_pred = (X_pred - X_mean) / X_std
            Xf = np.ones((len(X_scaled_pred), 1))
            for k in range(1, n_terms + 1):
                Xf = np.column_stack([Xf, np.sin(k * np.pi * X_scaled_pred), np.cos(k * np.pi * X_scaled_pred)])
            return Xf @ w
    else:
        degree = best_degree
        w = best_w
        
        def predict(X_pred):
            X_pred = X_pred.reshape(-1, 1)
            X_scaled_pred = (X_pred - X_mean) / X_std
            Xv = np.vander(X_scaled_pred, N=degree + 1, increasing=True)
            return Xv @ w
    
    return predict
