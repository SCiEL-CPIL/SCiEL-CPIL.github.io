"""Structural change: sparse kernel regression with multi-kernel CV ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_TIME_START = [0.0]
_TOTAL_DEADLINE = 310.0

def _remaining():
    return _TOTAL_DEADLINE - (time.time() - _TIME_START[0])

def _rbf_kernel(X, Y, gamma):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-gamma * d2)

def _poly_kernel(X, Y, degree, gamma, coef0):
    return (gamma * X @ Y.T + coef0) ** degree

def _matern_kernel(X, Y, l, nu=2.5):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    r = np.sqrt(d2 + 1e-12)
    if nu == 2.5:
        return (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
    elif nu == 1.5:
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r / l)
    else:
        return np.exp(-r / l)

def _fit_kernel_ridge(X, y, K_fn, lam):
    """Fit kernel ridge regression. K_fn(X, X) gives the kernel matrix."""
    K = K_fn(X, X)
    n = K.shape[0]
    K_reg = K + lam * np.eye(n)
    try:
        alpha = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K_reg, y, rcond=None)[0]
    
    def predict(Xt):
        K_test = K_fn(Xt, X)
        return K_test @ alpha
    return predict

def _fit_nystrom_kernel(X, y, K_fn, n_landmarks, lam):
    """Nyström approximate kernel regression."""
    n = X.shape[0]
    rng = np.random.default_rng(42)
    
    if n_landmarks >= n:
        landmark_idx = np.arange(n)
    else:
        # Farthest-point sampling
        landmarks = [rng.integers(0, n)]
        d2_first = np.sum((X - X[landmarks[0]]) ** 2, axis=1)
        d2_first = np.maximum(d2_first, 0.0)
        min_dist = d2_first.copy()
        
        for _ in range(n_landmarks - 1):
            idx = np.argmax(min_dist)
            landmarks.append(idx)
            d2_new = np.sum((X - X[idx]) ** 2, axis=1)
            d2_new = np.maximum(d2_new, 0.0)
            min_dist = np.minimum(min_dist, d2_new)
        
        landmark_idx = np.array(landmarks)
    
    X_land = X[landmark_idx]
    K_land_land = K_fn(X_land, X_land)
    K_train_land = K_fn(X, X_land)
    
    M = len(landmark_idx)
    A = K_train_land.T @ K_train_land + lam * np.eye(M)
    b = K_train_land.T @ y
    
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    
    def predict(Xt):
        K_test_land = K_fn(Xt, X_land)
        return K_test_land @ w
    return predict

def _cv_score(predict_fn, X, y, n_folds=5):
    """K-fold cross-validation MSE."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    fold_size = n // n_folds
    se = 0.0
    count = 0
    for i in range(n_folds):
        val_idx = idx[i * fold_size:(i + 1) * fold_size]
        tr_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
        if len(tr_idx) < 10:
            continue
        try:
            pred_val = predict_fn(X[val_idx])
            se += np.sum((pred_val - y[val_idx]) ** 2)
            count += len(val_idx)
        except Exception:
            continue
    if count == 0:
        return np.inf
    return se / count

def _build_kernel_specs(X, y):
    """Build a diverse set of kernel specifications."""
    n, d = X.shape
    specs = []
    
    # RBF kernels with various bandwidths
    x_range = np.max(np.ptp(X, axis=0)) + 1e-6
    for gamma in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        K_fn = lambda Xa, Xb, g=gamma: _rbf_kernel(Xa, Xb, g)
        specs.append(('rbf', K_fn, gamma))
    
    # Polynomial kernels
    for deg in (2, 3):
        for gamma in (0.5, 1.0, 2.0):
            K_fn = lambda Xa, Xb, dg=deg, g=gamma: _poly_kernel(Xa, Xb, dg, g, 1.0)
            specs.append((f'poly{deg}', K_fn, gamma))
    
    # Matern kernels
    for l in (0.5 * x_range, x_range, 2.0 * x_range):
        for nu in (1.5, 2.5):
            K_fn = lambda Xa, Xb, ll=l, nv=nu: _matern_kernel(Xa, Xb, ll, nv)
            specs.append((f'matern{nu}', K_fn, l))
    
    return specs

def fit(X_train, y_train):
    _TIME_START[0] = time.time()
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    yvar = np.var(y) + 1e-12
    
    if yvar < 1e-12:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    candidates = []
    
    # Build kernel specs
    kernel_specs = _build_kernel_specs(X, y)
    
    # Fit kernel ridge regression with various regularization strengths
    # Use full kernel for small n, Nyström for larger n
    use_nystrom = n > 150
    n_landmarks = min(80, n)
    
    for name, K_fn, param in kernel_specs:
        if _remaining() < 5:
            break
        
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            if _remaining() < 3:
                break
            try:
                if use_nystrom:
                    p = _fit_nystrom_kernel(X, y, K_fn, n_landmarks, lam)
                else:
                    p = _fit_kernel_ridge(X, y, K_fn, lam)
                candidates.append((f'{name}_{param}_l{lam}', p))
            except Exception:
                continue
    
    # Also add simple linear/polynomial baselines
    if _remaining() > 5:
        try:
            # Linear
            X_lin = np.column_stack([np.ones(n), X])
            w = np.linalg.lstsq(X_lin, y, rcond=None)[0]
            p = lambda Xt: np.column_stack([np.ones(len(Xt)), Xt]) @ w
            candidates.append(('linear', p))
        except Exception:
            pass
        
        try:
            # Quadratic
            X_quad = np.column_stack([np.ones(n)])
            for i in range(d):
                X_quad = np.column_stack([X_quad, X[:, i]])
            for i in range(d):
                for j in range(i, d):
                    X_quad = np.column_stack([X_quad, X[:, i] * X[:, j]])
            w = np.linalg.lstsq(X_quad, y, rcond=None)[0]
            def _quad_pred(Xt):
                Xtq = np.column_stack([np.ones(len(Xt))])
                for i in range(d):
                    Xtq = np.column_stack([Xtq, Xt[:, i]])
                for i in range(d):
                    for j in range(i, d):
                        Xtq = np.column_stack([Xtq, Xt[:, i] * Xt[:, j]])
                return Xtq @ w
            candidates.append(('quad', _quad_pred))
        except Exception:
            pass
    
    if not candidates:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Score candidates with CV
    scored = []
    for name, p in candidates:
        if _remaining() < 3:
            break
        try:
            score = _cv_score(p, X, y, n_folds=5)
            if np.isfinite(score):
                scored.append((name, p, score))
        except Exception:
            continue
    
    if not scored:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Sort by CV score
    scored.sort(key=lambda x: x[2])
    
    # Ensemble top-K with inverse-score weighting
    K_top = min(5, len(scored))
    top = scored[:K_top]
    
    scores = np.array([s[2] for s in top])
    # Soft weighting: better models get more weight
    weights = 1.0 / (scores + 1e-8)
    weights = weights / weights.sum()
    
    predictors = [s[1] for s in top]
    
    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        if Xt.ndim == 1:
            Xt = Xt.reshape(1, -1)
        out = np.zeros(len(Xt))
        for p, w in zip(predictors, weights):
            try:
                out += w * np.asarray(p(Xt), dtype=float).ravel()
            except Exception:
                continue
        out = np.where(np.isfinite(out), out, np.nanmean(y))
        return out
    
    return predict
# EVOLVE-BLOCK-END
