# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # 5s margin for 330s kill
    
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
        for j in range(d):
            cols.append(X[:, j])
        seen = set()
        for deg in range(2, degree + 1):
            for idx in product(range(d), repeat=deg):
                sorted_idx = tuple(sorted(idx))
                if sorted_idx in seen:
                    continue
                seen.add(sorted_idx)
                term = np.ones(len(X))
                for k in range(deg):
                    term = term * X[:, sorted_idx[k]]
                cols.append(term)
        return np.column_stack(cols)
    
    def build_rich_features(X, poly_deg=4, trig_freqs=None, include_exp=True):
        """Build a rich feature set combining polynomial, trig, and exp features."""
        if trig_freqs is None:
            trig_freqs = [1, 2, 3, 4, 5]
        
        cols = [np.ones(len(X))]
        
        # Polynomial features
        poly_feat = build_poly_features(X, poly_deg)
        cols.extend([poly_feat[:, j] for j in range(1, poly_feat.shape[1])])
        
        # Trigonometric features
        for j in range(d):
            for f in trig_freqs:
                cols.append(np.sin(f * X[:, j]))
                cols.append(np.cos(f * X[:, j]))
        
        # Exponential and other nonlinear features
        if include_exp:
            for j in range(d):
                cols.append(np.exp(X[:, j] / 10.0))
                cols.append(np.exp(-X[:, j] / 10.0))
                cols.append(np.log1p(X[:, j]))
                cols.append(np.abs(X[:, j]))
                cols.append(X[:, j] ** 0.5)
        
        return np.column_stack(cols)
    
    def build_rbf_features(X, centers, bandwidths):
        """Build RBF features."""
        cols = [np.ones(len(X))]
        for c in centers:
            for b in bandwidths:
                dist_sq = np.sum((X - c) ** 2, axis=1)
                cols.append(np.exp(-b * dist_sq))
        return np.column_stack(cols)
    
    def fit_linear(features, y, alpha=1e-6):
        """Fit ridge regression and return (mse, weights)."""
        try:
            n_feat = features.shape[1]
            A = features.T @ features + alpha * np.eye(n_feat)
            b = features.T @ y
            w = np.linalg.solve(A, b)
            pred = features @ w
            mse = np.mean((pred - y) ** 2)
            return mse, w
        except:
            return np.inf, None
    
    def train_nn(X, y, hidden_sizes, n_epochs=2000, lr=0.01, seed=42):
        """Train a small neural network via gradient descent."""
        rng = np.random.RandomState(seed)
        n = X.shape[0]
        
        # Initialize weights
        layers = [d] + list(hidden_sizes) + [1]
        W = [rng.randn(layers[i], layers[i+1]) * np.sqrt(2.0/layers[i]) for i in range(len(layers)-1)]
        b = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]
        
        def forward(x):
            activations = [x]
            for i in range(len(W)):
                z = activations[-1] @ W[i] + b[i]
                if i < len(W) - 1:
                    a = np.tanh(z)
                else:
                    a = z
                activations.append(a)
            return activations
        
        def backward(activations, target):
            m = len(W)
            dz = [None] * m
            da = [None] * m
            dW = [None] * m
            db = [None] * m
            
            # Output layer error
            da[m-1] = activations[m] - target  # MSE derivative
            dz[m-1] = da[m-1]  # linear output
            
            dW[m-1] = activations[m-1].T @ dz[m-1] / n
            db[m-1] = np.sum(dz[m-1], axis=0)
            
            for i in range(m-2, -1, -1):
                dz[i] = da[i] * (1 - activations[i+1]**2)  # tanh derivative
                dW[i] = activations[i].T @ dz[i] / n
                db[i] = np.sum(dz[i], axis=0)
                if i > 0:
                    da[i-1] = dz[i] @ W[i].T
            
            return dW, db
        
        best_w = None
        best_mse = np.inf
        
        for epoch in range(n_epochs):
            if not time_ok():
                break
            
            acts = forward(X)
            pred = acts[-1]
            mse = np.mean((pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_w = [w.copy() for w in W] + [b_.copy() for b_ in b]
            
            dW, db = backward(acts, y)
            for i in range(len(W)):
                W[i] -= lr * dW[i]
                b[i] -= lr * db[i]
        
        if best_w is None:
            return np.inf, None
        
        def predict(X_new, ww=best_w):
            W_r = ww[:len(hidden_sizes)+1]
            b_r = ww[len(hidden_sizes)+1:]
            a = X_new
            for i in range(len(W_r)):
                z = a @ W_r[i] + b_r[i]
                if i < len(W_r) - 1:
                    a = np.tanh(z)
                else:
                    a = z
            return a.flatten()
        
        return best_mse, predict
    
    best_mse = np.inf
    best_predictor = None
    
    # Strategy 1: Try rich feature basis with ridge regression
    configs = [
        {'poly_deg': 2, 'trig_freqs': [1, 2, 3], 'include_exp': True},
        {'poly_deg': 3, 'trig_freqs': [1, 2, 3, 4], 'include_exp': True},
        {'poly_deg': 4, 'trig_freqs': [1, 2, 3, 4, 5], 'include_exp': True},
        {'poly_deg': 3, 'trig_freqs': [1, 2, 3, 4, 5, 6, 7, 8], 'include_exp': False},
        {'poly_deg': 2, 'trig_freqs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'include_exp': True},
    ]
    
    for cfg in configs:
        if not time_ok():
            break
        try:
            feat = build_rich_features(X_train, **cfg)
            if feat.shape[1] > 3000:
                continue
            mse, w = fit_linear(feat, y_train, alpha=1e-6)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_rich_features, c=cfg, ww=w: f(X, **c) @ ww
        except:
            pass
    
    # Strategy 2: RBF features with multiple center sets
    if time_ok():
        n_centers_list = [10, 20, 30]
        bandwidths_list = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        
        for nc in n_centers_list:
            if not time_ok():
                break
            if n < nc:
                continue
            indices = np.linspace(0, n-1, nc, dtype=int)
            centers = X_train[indices]
            try:
                feat = build_rbf_features(X_train, centers, bandwidths_list)
                if feat.shape[1] > 3000:
                    continue
                mse, w = fit_linear(feat, y_train, alpha=1e-6)
                if mse < best_mse:
                    best_mse = mse
                    best_predictor = lambda X, f=build_rbf_features, cs=centers, bs=bandwidths_list, ww=w: f(X, cs, bs) @ ww
            except:
                pass
    
    # Strategy 3: Neural networks with various architectures
    nn_configs = [
        [20],
        [30],
        [40],
        [50],
        [20, 20],
        [30, 30],
        [40, 20],
        [20, 40],
        [30, 15],
        [50, 25],
    ]
    
    for hidden in nn_configs:
        if not time_ok():
            break
        try:
            mse, predictor = train_nn(X_train, y_train, hidden, n_epochs=3000, lr=0.01, seed=42)
            if mse < best_mse:
                best_mse = mse
                best_predictor = predictor
        except:
            pass
    
    # Strategy 4: Combine best feature set with NN hidden layer
    if time_ok():
        try:
            feat_base = build_rich_features(X_train, poly_deg=3, trig_freqs=[1, 2, 3], include_exp=True)
            rng = np.random.RandomState(42)
            n_hidden = 30
            W1 = rng.randn(d, n_hidden) * np.sqrt(2.0 / d)
            H = np.tanh(X_train @ W1)
            feat = np.column_stack([feat_base, H])
            mse, w = fit_linear(feat, y_train, alpha=1e-6)
            if mse < best_mse:
                best_mse = mse
                def predict_combined(X, fb=build_rich_features, W1=W1, ww=w):
                    H_new = np.tanh(X @ W1)
                    feat_new = np.column_stack([fb(X, poly_deg=3, trig_freqs=[1,2,3], include_exp=True), H_new])
                    return feat_new @ ww
                best_predictor = predict_combined
        except:
            pass
    
    # Fallback: simple linear
    if best_predictor is None:
        feat = np.column_stack([np.ones(n), X_train])
        w, *_ = np.linalg.lstsq(feat, y_train, rcond=None)
        best_predictor = lambda X: np.column_stack([np.ones(X.shape[0]), X]) @ w
    
    return best_predictor
# EVOLVE-BLOCK-END
