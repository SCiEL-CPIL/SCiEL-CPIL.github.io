"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 290.0  # safety margin before 300s limit
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    # Fallback: constant prediction
    def predict_const(X):
        return np.full(len(X), y_mean)
    
    if y_var < 1e-12:
        return predict_const
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Feature generation strategies
    def gen_chebyshev_features(X, degree=6):
        """Chebyshev polynomial features with cross terms up to total degree."""
        cols = [np.ones(len(X))]
        # Normalize inputs to [-1, 1] for better Chebyshev conditioning
        Xn = np.zeros_like(X, dtype=float)
        for j in range(d):
            mn, mx = X[:, j].min(), X[:, j].max()
            if mx > mn:
                Xn[:, j] = 2.0 * (X[:, j] - mn) / (mx - mn) - 1.0
            else:
                Xn[:, j] = 0.0
        
        # Generate all monomials in Chebyshev basis up to total degree
        for total_deg in range(1, degree + 1):
            # All compositions of total_deg into d parts (non-negative)
            for combo in product(range(total_deg + 1), repeat=d):
                if sum(combo) == total_deg:
                    feat = np.ones(len(X))
                    for j in range(d):
                        if combo[j] > 0:
                            # Chebyshev polynomial of degree combo[j]
                            T = np.cos(combo[j] * np.arccos(np.clip(Xn[:, j], -1, 1)))
                            feat = feat * T
                    cols.append(feat)
        return np.column_stack(cols)
    
    def gen_fourier_features(X, max_freq=4):
        """Fourier features with sin/cos at multiple frequencies."""
        cols = [np.ones(len(X))]
        for j in range(d):
            for k in range(1, max_freq + 1):
                cols.append(np.sin(2 * np.pi * k * X[:, j] / (X[:, j].max() - X[:, j].min() + 1e-8)))
                cols.append(np.cos(2 * np.pi * k * X[:, j] / (X[:, j].max() - X[:, j].min() + 1e-8)))
        # Cross terms: products of sin/cos from different dimensions
        for j1 in range(d):
            for j2 in range(j1 + 1, d):
                for k in range(1, min(max_freq, 2) + 1):
                    s1 = np.sin(2 * np.pi * k * X[:, j1] / (X[:, j1].max() - X[:, j1].min() + 1e-8))
                    c1 = np.cos(2 * np.pi * k * X[:, j1] / (X[:, j1].max() - X[:, j1].min() + 1e-8))
                    s2 = np.sin(2 * np.pi * k * X[:, j2] / (X[:, j2].max() - X[:, j2].min() + 1e-8))
                    c2 = np.cos(2 * np.pi * k * X[:, j2] / (X[:, j2].max() - X[:, j2].min() + 1e-8))
                    cols.append(s1 * c2)
                    cols.append(c1 * s2)
        return np.column_stack(cols)
    
    def gen_rbf_features(X, gammas=None):
        """RBF kernel features with multiple scales."""
        if gammas is None:
            gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        # Use a subset of training points as centers (k-means-like selection)
        n_centers = min(20, n)
        indices = np.random.choice(n, n_centers, replace=False)
        centers = X_train[indices]
        
        cols = []
        for gamma in gammas:
            for c in centers:
                dist_sq = np.sum((X - c) ** 2, axis=1)
                cols.append(np.exp(-gamma * dist_sq))
        return np.column_stack(cols)
    
    def gen_polynomial_features(X, degree=3):
        """Standard polynomial features with cross terms."""
        cols = [np.ones(len(X))]
        for total_deg in range(1, degree + 1):
            for combo in product(range(total_deg + 1), repeat=d):
                if sum(combo) == total_deg:
                    feat = np.ones(len(X))
                    for j in range(d):
                        if combo[j] > 0:
                            feat = feat * (X[:, j] ** combo[j])
                    cols.append(feat)
        return np.column_stack(cols)
    
    def ridge_fit(Phi, y, alpha=1e-4):
        """Ridge regression."""
        I = np.eye(Phi.shape[1]) * alpha
        w = np.linalg.solve(Phi.T @ Phi + I, Phi.T @ y)
        return w
    
    def predict_with_features(X, Phi_train, w):
        """Predict using precomputed feature weights."""
        return Phi_train @ w  # placeholder, actual prediction below
    
    # Strategy: try multiple feature bases, use cross-validation to pick best
    strategies = [
        ("cheb5", lambda X: gen_chebyshev_features(X, degree=5)),
        ("cheb6", lambda X: gen_chebyshev_features(X, degree=6)),
        ("poly3", lambda X: gen_polynomial_features(X, degree=3)),
        ("poly4", lambda X: gen_polynomial_features(X, degree=4)),
        ("fourier", lambda X: gen_fourier_features(X, max_freq=4)),
        ("rbf", lambda X: gen_rbf_features(X)),
    ]
    
    best_score = np.inf
    best_predict = predict_const
    
    # Time-aware loop: try strategies in order, stop when time is low
    for name, feat_func in strategies:
        if time_left() < 30.0:
            break
        
        try:
            Phi_train = feat_func(X_train)
            n_features = Phi_train.shape[1]
            
            # Use ridge regression with small regularization
            alpha = 1e-4 * max(1, n_features)
            w = ridge_fit(Phi_train, y_train, alpha)
            
            # Compute training MSE as proxy
            pred_train = Phi_train @ w
            train_mse = np.mean((pred_train - y_train) ** 2)
            
            # Create a simple predict function
            def make_predict(feat_func, w):
                def predict(X):
                    Phi = feat_func(X)
                    return Phi @ w
                return predict
            
            pred_fn = make_predict(feat_func, w)
            
            # Use a hold-out validation split for model selection
            n_val = max(20, n // 5)
            idx = np.random.permutation(n)
            val_idx = idx[:n_val]
            tr_idx = idx[n_val:]
            
            Phi_val = feat_func(X_train[val_idx])
            Phi_tr = feat_func(X_train[tr_idx])
            
            alpha_val = 1e-4 * max(1, n_features)
            w_val = ridge_fit(Phi_tr, y_train[tr_idx], alpha_val)
            
            pred_val = Phi_val @ w_val
            val_mse = np.mean((pred_val - y_train[val_idx]) ** 2)
            
            if val_mse < best_score:
                best_score = val_mse
                best_predict = pred_fn
        
        except Exception:
            continue
    
    # If we have time left, try to improve with ensemble of top methods
    if time_left() > 60.0:
        # Try averaging predictions from top 2-3 models
        # For simplicity, just return the best single model found
        pass
    
    return best_predict
# EVOLVE-BLOCK-END
