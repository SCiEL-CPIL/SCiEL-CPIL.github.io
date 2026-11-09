# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 320.0  # 10s margin for 330s kill
    
    n, d = X_train.shape
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_mean = np.mean(y_train)
        return lambda X: np.full(X.shape[0], y_mean)
    
    def time_ok():
        return time.time() < deadline
    
    def build_poly_features(X, degree):
        """Build polynomial features up to given degree with unique monomials."""
        cols = [np.ones(len(X))]
        # Degree 1
        for j in range(d):
            cols.append(X[:, j])
        # Higher degrees - use unique combinations with repetition
        for deg in range(2, degree + 1):
            for idx in product(range(d), repeat=deg):
                sorted_idx = tuple(sorted(idx))
                # Check if this combination was already added by looking at count
                term = np.ones(len(X))
                for k in range(deg):
                    term = term * X[:, sorted_idx[k]]
                # Only add if not duplicate (check via coefficient pattern)
                cols.append(term)
        # Remove duplicates by checking column uniqueness
        arr = np.column_stack(cols)
        # Keep unique columns
        _, idx = np.unique(arr, axis=1, return_index=True)
        return arr[:, np.sort(idx)]
    
    def build_trig_features(X, freqs=None):
        """Build trigonometric features with multiple frequencies."""
        if freqs is None:
            freqs = [1, 2, 3]
        cols = [np.ones(len(X))]
        for j in range(d):
            for f in freqs:
                cols.append(np.sin(f * X[:, j]))
                cols.append(np.cos(f * X[:, j]))
        return np.column_stack(cols)
    
    def build_exp_features(X):
        """Build exponential and log features."""
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(np.exp(X[:, j] / 5.0))
            cols.append(np.exp(-X[:, j] / 5.0))
            cols.append(np.log1p(X[:, j]))
            cols.append(np.abs(X[:, j]))
            cols.append(X[:, j] ** 2)
            cols.append(X[:, j] ** 0.5)
        return np.column_stack(cols)
    
    def build_interaction_features(X):
        """Build pairwise interaction features."""
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(X[:, j])
        for i in range(d):
            for j in range(i+1, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(np.sin(X[:, i] * X[:, j]))
                cols.append(np.cos(X[:, i] * X[:, j]))
                cols.append(np.abs(X[:, i] - X[:, j]))
        return np.column_stack(cols)
    
    def build_rbf_features(X, centers, bandwidths):
        """Build RBF features."""
        cols = [np.ones(len(X))]
        for c in centers:
            for b in bandwidths:
                dist_sq = np.sum((X - c) ** 2, axis=1)
                cols.append(np.exp(-b * dist_sq))
        return np.column_stack(cols)
    
    def fit_and_score(X, y, features):
        """Fit linear model and return training MSE."""
        try:
            # Add small ridge for stability
            reg = 1e-8
            w, residuals, rank, sv = np.linalg.lstsq(features, y, rcond=None)
            pred = features @ w
            mse = np.mean((pred - y) ** 2)
            return mse, w
        except:
            return np.inf, None
    
    best_mse = np.inf
    best_predictor = None
    
    # Strategy: try multiple feature sets, pick best by training MSE
    # Since data is noiseless, training MSE should be very small for the right model
    
    # 1. Try polynomial features of various degrees
    for degree in [2, 3, 4, 5, 6]:
        if not time_ok():
            break
        try:
            feat = build_poly_features(X_train, degree)
            if feat.shape[1] > 500:  # Too many features
                continue
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_poly_features, dg=degree, ww=w: f(X, dg) @ ww
        except:
            pass
    
    # 2. Try trigonometric features with different frequencies
    for freqs in [[1, 2], [1, 2, 3], [1, 2, 3, 4]]:
        if not time_ok():
            break
        try:
            feat = build_trig_features(X_train, freqs)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_trig_features, fr=freqs, ww=w: f(X, fr) @ ww
        except:
            pass
    
    # 3. Try exponential/log features
    if time_ok():
        try:
            feat = build_exp_features(X_train)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_exp_features, ww=w: f(X) @ ww
        except:
            pass
    
    # 4. Try interaction features
    if time_ok():
        try:
            feat = build_interaction_features(X_train)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_interaction_features, ww=w: f(X) @ ww
        except:
            pass
    
    # 5. Try RBF features with well-distributed centers
    if time_ok():
        # Use evenly spaced points as centers
        n_centers = min(20, n)
        if n >= n_centers:
            # Spread centers across the input space
            indices = np.linspace(0, n-1, n_centers, dtype=int)
            centers = X_train[indices]
            bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
            try:
                feat = build_rbf_features(X_train, centers, bandwidths)
                mse, w = fit_and_score(X_train, y_train, feat)
                if mse < best_mse:
                    best_mse = mse
                    best_predictor = lambda X, f=build_rbf_features, cs=centers, bs=bandwidths, ww=w: f(X, cs, bs) @ ww
            except:
                pass
    
    # 6. Try combined feature sets
    if time_ok():
        try:
            feat1 = build_poly_features(X_train, 3)
            feat2 = build_trig_features(X_train, [1, 2])
            feat = np.column_stack([feat1, feat2])
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f1=build_poly_features, f2=build_trig_features, dg=3, fr=[1,2], ww=w: np.column_stack([f1(X, dg), f2(X, fr)]) @ ww
        except:
            pass
    
    if time_ok():
        try:
            feat1 = build_poly_features(X_train, 2)
            feat2 = build_exp_features(X_train)
            feat = np.column_stack([feat1, feat2])
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f1=build_poly_features, f2=build_exp_features, dg=2, ww=w: np.column_stack([f1(X, dg), f2(X)]) @ ww
        except:
            pass
    
    # 7. Try a neural-network-like approach with tanh activation
    if time_ok():
        try:
            # Simple single hidden layer with tanh
            n_hidden = 30
            # Initialize with fixed random weights for reproducibility
            rng = np.random.RandomState(42)
            W1 = rng.randn(d, n_hidden) * np.sqrt(2.0 / d)
            b1 = np.zeros(n_hidden)
            W2 = rng.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
            b2 = 0.0
            
            # Compute hidden layer
            Z1 = X_train @ W1 + b1
            H = np.tanh(Z1)
            Z2 = H @ W2 + b2
            
            # Use this as a feature and fit linear on top
            feat = np.column_stack([np.ones(n), X_train, H])
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                def predict_nn(X, W1=W1, b1=b1, ww=w):
                    Z1 = X @ W1 + b1
                    H = np.tanh(Z1)
                    feat = np.column_stack([np.ones(X.shape[0]), X, H])
                    return feat @ ww
                best_predictor = predict_nn
        except:
            pass
    
    # If no good model found, fall back to simple linear
    if best_predictor is None:
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X_train]), y_train, rcond=None)
        best_predictor = lambda X: np.column_stack([np.ones(X.shape[0]), X]) @ w
    
    return best_predictor
# EVOLVE-BLOCK-END
