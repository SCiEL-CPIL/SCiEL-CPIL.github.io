"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 290.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Precompute feature matrices for different model families
    def build_poly_features(X, max_deg=4):
        cols = [np.ones(len(X))]
        for degree in range(1, max_deg + 1):
            for combo in combinations_with_replacement(range(d), degree):
                term = np.ones(len(X))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        return np.column_stack(cols)
    
    def build_rbf_features(X, centers, gamma):
        """RBF features with given centers and gamma"""
        F = np.zeros((len(X), len(centers)))
        for i, c in enumerate(centers):
            F[:, i] = np.exp(-gamma * np.sum((X - c) ** 2, axis=1))
        F[:, 0] += 1.0  # add bias
        return F
    
    def build_fourier_features(X, max_freq=5):
        """Fourier features for each input dimension"""
        cols = [np.ones(len(X))]
        for j in range(d):
            xj = X[:, j]
            for k in range(1, max_freq + 1):
                cols.append(np.sin(k * xj))
                cols.append(np.cos(k * xj))
        return np.column_stack(cols)
    
    def build_extended_features(X, max_deg=3):
        """Polynomial + nonlinear transforms"""
        cols = [np.ones(len(X))]
        # Polynomial terms up to max_deg
        for degree in range(1, max_deg + 1):
            for combo in combinations_with_replacement(range(d), degree):
                term = np.ones(len(X))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        # Non-linear per input
        for j in range(d):
            xj = np.clip(X[:, j], 1e-10, None)
            cols.append(np.log(xj))
            cols.append(np.sqrt(xj))
            cols.append(1.0 / xj)
            cols.append(xj * np.log(xj))
            cols.append(np.log(xj) ** 2)
            cols.append(np.sin(xj))
            cols.append(np.cos(xj))
        # Pairwise interactions of nonlinear features
        for i in range(d):
            for j in range(i + 1, d):
                xi = np.clip(X[:, i], 1e-10, None)
                xj = np.clip(X[:, j], 1e-10, None)
                cols.append(np.log(xi) * np.log(xj))
                cols.append(np.sqrt(xi) * np.sqrt(xj))
                cols.append((1.0 / xi) * (1.0 / xj))
                cols.append(np.sin(xi) * np.cos(xj))
                cols.append(np.cos(xi) * np.sin(xj))
        return np.column_stack(cols)
    
    # Ridge regression solver
    def ridge_solve(F, y, lam):
        n_features = F.shape[1]
        try:
            A = F.T @ F + lam * np.eye(n_features)
            b = F.T @ y
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F, y, rcond=None)
        return w
    
    # Validation split for model selection
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_frac = 0.2
    val_size = int(n * val_frac)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_train_sub = X[train_idx]
    y_train_sub = y[train_idx]
    
    var_y = np.var(y)
    if var_y < 1e-15:
        var_y = 1e-15
    
    def eval_predictor(predict_fn, X_eval, y_eval):
        try:
            pred = predict_fn(X_eval)
            pred = np.asarray(pred, dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse = np.mean((pred - y_eval) ** 2)
            return mse / var_y
        except:
            return np.inf
    
    best_predictor = None
    best_score = np.inf
    
    # Candidate configurations: (name, feature_builder, params)
    candidates = []
    
    # Polynomial models
    for max_deg in [2, 3, 4]:
        for lam in [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
            candidates.append(('poly', max_deg, lam))
    
    # Extended features
    for max_deg in [2, 3]:
        for lam in [0.0, 1e-6, 1e-4, 1e-2]:
            candidates.append(('ext', max_deg, lam))
    
    # Fourier features
    for max_freq in [3, 5, 8]:
        for lam in [0.0, 1e-6, 1e-4]:
            candidates.append(('fourier', max_freq, lam))
    
    # RBF features - generate centers from training data
    if d <= 3 and n >= 50:
        np.random.seed(123)
        n_centers = min(30, n // 5)
        center_indices = np.random.choice(n, n_centers, replace=False)
        centers = X[center_indices]
        for gamma in [0.5, 1.0, 2.0, 5.0]:
            for lam in [1e-4, 1e-3, 1e-2, 0.1]:
                candidates.append(('rbf', gamma, lam))
    
    # Evaluate each candidate
    for name, p1, p2 in candidates:
        if time.time() > deadline - 30:
            break
        
        try:
            if name == 'poly':
                F_train = build_poly_features(X_train_sub, p1)
                F_val = build_poly_features(X_val, p1)
                w = ridge_solve(F_train, y_train_sub, p2)
                
                def predict_poly(X_pred, _w=w, _deg=p1):
                    F_pred = build_poly_features(X_pred, _deg)
                    return F_pred @ _w
                
                score = eval_predictor(predict_poly, X_val, y_val)
                
            elif name == 'ext':
                F_train = build_extended_features(X_train_sub, p1)
                F_val = build_extended_features(X_val, p1)
                w = ridge_solve(F_train, y_train_sub, p2)
                
                def predict_ext(X_pred, _w=w, _deg=p1):
                    F_pred = build_extended_features(X_pred, _deg)
                    return F_pred @ _w
                
                score = eval_predictor(predict_ext, X_val, y_val)
                
            elif name == 'fourier':
                F_train = build_fourier_features(X_train_sub, p1)
                F_val = build_fourier_features(X_val, p1)
                w = ridge_solve(F_train, y_train_sub, p2)
                
                def predict_fourier(X_pred, _w=w, _freq=p1):
                    F_pred = build_fourier_features(X_pred, _freq)
                    return F_pred @ _w
                
                score = eval_predictor(predict_fourier, X_val, y_val)
                
            elif name == 'rbf':
                gamma = p1
                lam = p2
                F_train = build_rbf_features(X_train_sub, centers, gamma)
                F_val = build_rbf_features(X_val, centers, gamma)
                w = ridge_solve(F_train, y_train_sub, lam)
                
                def predict_rbf(X_pred, _w=w, _gamma=gamma):
                    F_pred = build_rbf_features(X_pred, centers, _gamma)
                    return F_pred @ _w
                
                score = eval_predictor(predict_rbf, X_val, y_val)
            else:
                continue
                
        except:
            continue
        
        if score < best_score:
            best_score = score
            best_predictor = (name, p1, p2)
    
    # Refit best model on all data
    if best_predictor is not None:
        name, p1, p2 = best_predictor
        try:
            if name == 'poly':
                F = build_poly_features(X, p1)
                w = ridge_solve(F, y, p2)
                def predict(X_pred):
                    F_pred = build_poly_features(X_pred, p1)
                    return F_pred @ w
                return predict
            elif name == 'ext':
                F = build_extended_features(X, p1)
                w = ridge_solve(F, y, p2)
                def predict(X_pred):
                    F_pred = build_extended_features(X_pred, p1)
                    return F_pred @ w
                return predict
            elif name == 'fourier':
                F = build_fourier_features(X, p1)
                w = ridge_solve(F, y, p2)
                def predict(X_pred):
                    F_pred = build_fourier_features(X_pred, p1)
                    return F_pred @ w
                return predict
            elif name == 'rbf':
                gamma = p1
                lam = p2
                F = build_rbf_features(X, centers, gamma)
                w = ridge_solve(F, y, lam)
                def predict(X_pred):
                    F_pred = build_rbf_features(X_pred, centers, gamma)
                    return F_pred @ w
                return predict
        except:
            pass
    
    # Fallback: simple polynomial degree 3
    try:
        F = build_poly_features(X, 3)
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
        def predict(X_pred):
            F_pred = build_poly_features(X_pred, 3)
            return F_pred @ w
        return predict
    except:
        # Last resort: mean predictor
        mean_y = np.mean(y)
        def predict(X_pred):
            return np.full(len(X_pred), mean_y)
        return predict
# EVOLVE-BLOCK-END
