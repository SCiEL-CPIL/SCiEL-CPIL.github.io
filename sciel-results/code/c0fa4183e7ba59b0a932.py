import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0
    
    d = X_train.shape[1]
    n = len(X_train)
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    y_norm = (y_train - y_mean) / np.sqrt(y_var)
    
    best_score = np.inf
    best_predictor = None
    
    def evaluate(model, X_test=None, y_test=None):
        """Return normalized MSE on training data (proxy for test)."""
        pred = model(X_train)
        mse = np.mean((pred - y_train)**2)
        return mse / y_var
    
    def make_predictor(model_fn):
        """Wrap model to return proper predictions."""
        return lambda X: model_fn(X)
    
    # Strategy 1: Polynomial regression (degrees 2-5)
    for deg in [2, 3, 4, 5]:
        if time.time() > deadline - 5:
            break
        def poly_features(X, d=d, deg=deg):
            cols = [np.ones(len(X))]
            # All monomials up to degree deg
            for total_deg in range(1, deg + 1):
                for indices in product(range(d), repeat=total_deg):
                    # Sort to avoid duplicates
                    if sorted(indices) != indices:
                        continue
                    feat = np.ones(len(X))
                    for idx in indices:
                        feat = feat * X[:, idx]
                    cols.append(feat)
            return np.column_stack(cols)
        
        try:
            F = poly_features(X_train)
            w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            pred = F @ w
            score = np.mean((pred - y_train)**2) / y_var
            if score < best_score:
                best_score = score
                best_predictor = (lambda X, F_fn=poly_features, w=w: F_fn(X) @ w)
        except:
            pass
    
    # Strategy 2: GP Regression
    if time.time() < deadline - 10:
        try:
            from scipy.spatial.distance import cdist
            def gp_predict(X_pred, X_tr, y_tr, length_scale, sigma_f, noise):
                K_tr = sigma_f**2 * np.exp(-cdist(X_tr, X_tr)**2 / (2 * length_scale**2)) + noise * np.eye(n)
                K_pred = sigma_f**2 * np.exp(-cdist(X_pred, X_tr)**2 / (2 * length_scale**2))
                L = np.linalg.cholesky(K_tr)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr - y_mean))
                mu = y_mean + K_pred @ alpha
                return mu
            
            for ls in [0.5, 1.0, 2.0, 5.0]:
                for sf in [1.0, 5.0, 10.0]:
                    if time.time() > deadline - 5:
                        break
                    try:
                        pred_tr = gp_predict(X_train, X_train, y_train, ls, sf, 1e-8)
                        score = np.mean((pred_tr - y_train)**2) / y_var
                        if score < best_score:
                            best_score = score
                            X_tr_copy = X_train.copy()
                            y_tr_copy = y_train.copy()
                            ls_copy, sf_copy = ls, sf
                            best_predictor = (lambda X, X_t=X_tr_copy, y_t=y_tr_copy, l=ls_copy, s=sf_copy: gp_predict(X, X_t, y_t, l, s, 1e-8))
                    except:
                        pass
        except ImportError:
            pass
    
    # Strategy 3: Symbolic expression search
    if time.time() < deadline - 15:
        # Generate candidate features from basic functions
        base_funcs = [
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: np.exp(-x**2 / 10),
            lambda x: np.log1p(x),
            lambda x: 1.0 / (1.0 + np.exp(-x)),
        ]
        
        # Generate single-variable features and pairwise products
        candidate_features = []
        candidate_names = []
        
        # Constant
        candidate_features.append(np.ones(n))
        candidate_names.append('1')
        
        # Single variable transforms
        for j in range(d):
            for fi, f in enumerate(base_funcs):
                try:
                    feat = f(X_train[:, j])
                    if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                        candidate_features.append(feat)
                        candidate_names.append(f'{f.__name__ if hasattr(f,"__name__") else "f"}(x{j})')
                except:
                    pass
        
        # Pairwise products (only if d >= 2)
        if d >= 2:
            for i in range(d):
                for j in range(i+1, d):
                    for fi, f1 in enumerate(base_funcs[:3]):  # limit to first 3 for speed
                        for fj, f2 in enumerate(base_funcs[:3]):
                            try:
                                feat1 = f1(X_train[:, i])
                                feat2 = f2(X_train[:, j])
                                feat = feat1 * feat2
                                if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                                    candidate_features.append(feat)
                                    candidate_names.append(f'{f1.__name__ if hasattr(f1,"__name__") else "f"}(x{i})*{f2.__name__ if hasattr(f2,"__name__") else "f"}(x{j})')
                            except:
                                pass
        
        # Also try sums of pairs
        if d >= 2:
            for i in range(d):
                for j in range(i+1, d):
                    for fi, f1 in enumerate(base_funcs[:2]):
                        for fj, f2 in enumerate(base_funcs[:2]):
                            try:
                                feat1 = f1(X_train[:, i])
                                feat2 = f2(X_train[:, j])
                                feat = feat1 + feat2
                                if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                                    candidate_features.append(feat)
                                    candidate_names.append(f'{f1.__name__ if hasattr(f1,"__name__") else "f"}(x{i})+{f2.__name__ if hasattr(f2,"__name__") else "f"}(x{j})')
                            except:
                                pass
        
        if len(candidate_features) > 0:
            F_cand = np.column_stack(candidate_features)
            n_feat = F_cand.shape[1]
            
            # Try different regularization strengths
            for alpha in [0, 0.01, 0.1, 1.0, 10.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    if alpha == 0:
                        w, *_ = np.linalg.lstsq(F_cand, y_train, rcond=None)
                    else:
                        # Ridge regression
                        A = F_cand.T @ F_cand + alpha * np.eye(n_feat)
                        b = F_cand.T @ y_train
                        w = np.linalg.solve(A, b)
                    pred = F_cand @ w
                    score = np.mean((pred - y_train)**2) / y_var
                    if score < best_score:
                        best_score = score
                        # Store for prediction
                        F_cand_copy = F_cand.copy()
                        w_copy = w.copy()
                        d_copy = d
                        candidate_features_copy = [f.copy() for f in candidate_features]
                        
                        def sym_predict(X, F_c=F_cand_copy, w_c=w_copy, d_c=d_c, cands=candidate_features_copy, names=candidate_names):
                            # Rebuild features for X
                            n_new = len(X)
                            new_feats = [np.ones(n_new)]
                            for j in range(d_c):
                                for fi, f in enumerate(base_funcs):
                                    try:
                                        feat = f(X[:, j])
                                        if np.all(np.isfinite(feat)):
                                            new_feats.append(feat)
                                        else:
                                            new_feats.append(np.zeros(n_new))
                                    except:
                                        new_feats.append(np.zeros(n_new))
                            if d_c >= 2:
                                for i in range(d_c):
                                    for j in range(i+1, d_c):
                                        for fi, f1 in enumerate(base_funcs[:3]):
                                            for fj, f2 in enumerate(base_funcs[:3]):
                                                try:
                                                    feat1 = f1(X[:, i])
                                                    feat2 = f2(X[:, j])
                                                    feat = feat1 * feat2
                                                    if np.all(np.isfinite(feat)):
                                                        new_feats.append(feat)
                                                    else:
                                                        new_feats.append(np.zeros(n_new))
                                                except:
                                                    new_feats.append(np.zeros(n_new))
                                for i in range(d_c):
                                    for j in range(i+1, d_c):
                                        for fi, f1 in enumerate(base_funcs[:2]):
                                            for fj, f2 in enumerate(base_funcs[:2]):
                                                try:
                                                    feat1 = f1(X[:, i])
                                                    feat2 = f2(X[:, j])
                                                    feat = feat1 + feat2
                                                    if np.all(np.isfinite(feat)):
                                                        new_feats.append(feat)
                                                    else:
                                                        new_feats.append(np.zeros(n_new))
                                                except:
                                                    new_feats.append(np.zeros(n_new))
                            F_new = np.column_stack(new_feats)
                            return F_new @ w_c
                        
                        best_predictor = sym_predict
                except:
                    pass
    
    # Strategy 4: Small neural network
    if time.time() < deadline - 20:
        try:
            from sklearn.neural_network import MLPRegressor
            for hidden in [(16,), (32,), (16, 16)]:
                if time.time() > deadline - 5:
                    break
                try:
                    mlp = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000, random_state=42, 
                                        early_stopping=False, activation='relu', solver='adam',
                                        learning_rate_init=0.01)
                    mlp.fit(X_train, y_train)
                    pred = mlp.predict(X_train)
                    score = np.mean((pred - y_train)**2) / y_var
                    if score < best_score:
                        best_score = score
                        best_predictor = mlp.predict
                except:
                    pass
        except ImportError:
            pass
    
    # Fallback: constant predictor
    if best_predictor is None:
        best_predictor = lambda X: np.full(len(X), y_mean)
    
    return best_predictor
