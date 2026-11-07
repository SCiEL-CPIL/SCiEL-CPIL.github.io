# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import differential_evolution
from scipy.interpolate import make_interp_spline
from scipy.special import comb

warnings.filterwarnings("ignore")

_TOTAL_DEADLINE = time.time() + 325.0  # safety margin under 330s


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _nrmse(pred, y):
    """Normalized MSE: mean((pred-y)^2) / var(y)"""
    var_y = np.var(y)
    if var_y < 1e-15:
        return 0.0
    return np.mean((pred - y) ** 2) / var_y


def _poly_features(X, degree):
    """Polynomial features up to given degree for 1D input."""
    x = X[:, 0] if X.ndim == 2 else X
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def _trig_features(X, n_harm=6):
    """Trigonometric features for 1D input."""
    x = X[:, 0] if X.ndim == 2 else X
    cols = [np.ones_like(x)]
    for k in range(1, n_harm + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _exp_features(X, n_terms=5):
    """Exponential features: exp(k*x) for various k."""
    x = X[:, 0] if X.ndim == 2 else X
    cols = [np.ones_like(x)]
    for k in range(-n_terms, n_terms + 1):
        if k == 0:
            continue
        # Scale to avoid overflow
        cols.append(np.exp(0.3 * k * x))
    return np.column_stack(cols)


def _log_features(X):
    """Log features for positive input."""
    x = X[:, 0] if X.ndim == 2 else X
    cols = [np.ones_like(x)]
    cols.append(np.log(np.maximum(x, 1e-10)))
    cols.append(np.log(np.maximum(x, 1e-10)) ** 2)
    cols.append(x)
    cols.append(x ** 2)
    cols.append(x ** 3)
    return np.column_stack(cols)


def _mixed_features(X, poly_deg=5, n_harm=4):
    """Mixed polynomial + trigonometric features."""
    x = X[:, 0] if X.ndim == 2 else X
    cols = [np.ones_like(x)]
    for d in range(1, poly_deg + 1):
        cols.append(x ** d)
    for k in range(1, n_harm + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _rbf_features(X, centers, bandwidths):
    """Gaussian RBF features."""
    x = X[:, 0] if X.ndim == 2 else X
    cols = [np.ones_like(x)]
    for c, b in zip(centers, bandwidths):
        cols.append(np.exp(-((x - c) ** 2) / (2 * b ** 2)))
    return np.column_stack(cols)


def _fit_ols(features, y, max_iter=1000):
    """Fit OLS with ridge regularization if needed."""
    try:
        w, *_ = np.linalg.lstsq(features, y, rcond=None)
        return w
    except Exception:
        # Add small ridge
        lam = 1e-6
        A = features.T @ features + lam * np.eye(features.shape[1])
        b = features.T @ y
        try:
            w = np.linalg.solve(A, b)
            return w
        except Exception:
            return np.zeros(features.shape[1])


def _fit_ols_ridge(features, y, alpha=1e-4):
    """Ridge regression."""
    A = features.T @ features + alpha * np.eye(features.shape[1])
    b = features.T @ y
    try:
        w = np.linalg.solve(A, b)
        return w
    except Exception:
        return np.linalg.pinv(A) @ b


def _spline_fit(x, y, k=4):
    """Fit a cubic spline."""
    x_sorted = np.sort(x)
    y_sorted = y[np.argsort(x)]
    try:
        # Use a subset of points for knot placement
        n_knots = min(len(x_sorted), 20)
        if n_knots < 2 * k + 1:
            n_knots = 2 * k + 1
        knots = np.linspace(x_sorted[0], x_sorted[-1], n_knots)
        spl = make_interp_spline(knots, y_sorted[knots is not None] if False else y_sorted[:len(knots)], k=min(k, len(knots) - 1))
        return spl
    except Exception:
        return None


def _fit_mlp(X, y, hidden_sizes=(32, 32), lr=0.01, epochs=500, seed=42):
    """Simple MLP using gradient descent."""
    rng = np.random.RandomState(seed)
    x = X[:, 0] if X.ndim == 2 else X
    # Normalize
    x_mean, x_std = x.mean(), x.std()
    if x_std < 1e-10:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    y_mean, y_std = y.mean(), y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    n = len(x_norm)
    layers = [1] + list(hidden_sizes) + [1]
    weights = [rng.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(len(layers)-1)]
    biases = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]
    
    def forward(X_):
        a = X_
        acts = [a]
        for i in range(len(weights)):
            z = a @ weights[i] + biases[i]
            if i < len(weights) - 1:
                a = np.tanh(z)
            else:
                a = z
            acts.append(a)
        return acts
    
    def backward(acts, target):
        m = acts[-1].shape[0]
        dz = (acts[-1] - target) / m
        grads_w = [None] * len(weights)
        grads_b = [None] * len(biases)
        for i in range(len(weights) - 1, -1, -1):
            grads_w[i] = acts[i].T @ dz
            grads_b[i] = np.sum(dz, axis=0)
            if i > 0:
                dz = (dz @ weights[i].T) * (1 - acts[i+1]**2)
        return grads_w, grads_b
    
    for epoch in range(epochs):
        if _remaining() < 5:
            break
        acts = forward(x_norm.reshape(-1, 1))
        grads_w, grads_b = backward(acts, y_norm.reshape(-1, 1))
        for i in range(len(weights)):
            weights[i] -= lr * grads_w[i]
            biases[i] -= lr * grads_b[i]
    
    def predict_mlp(X_):
        x_ = X_[:, 0] if X_.ndim == 2 else X_
        x_n = (x_ - x_mean) / x_std
        acts = forward(x_n.reshape(-1, 1))
        return (acts[-1].flatten() * y_std + y_mean)
    
    return predict_mlp


def _try_symbolic_regression(X, y, time_budget=30):
    """Try to find a simple symbolic expression via grid search over operators."""
    x = X[:, 0] if X.ndim == 2 else X
    best_score = np.inf
    best_pred = y.copy()
    
    # Try various simple forms
    # 1: a*x + b
    # 2: a*x^2 + b*x + c
    # 3: a*sin(x) + b
    # 4: a*exp(b*x) + c
    # 5: a*log(x) + b (for positive x)
    # 6: a*x^b + c
    # 7: a*sin(b*x + c) + d
    # 8: a*exp(-b*(x-c)^2) + d (Gaussian)
    
    n = len(x)
    
    # Linear
    A = np.column_stack([np.ones(n), x])
    try:
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        score = _nrmse(pred, y)
        if score < best_score:
            best_score = score
            best_pred = pred
    except:
        pass
    
    # Quadratic
    A = np.column_stack([np.ones(n), x, x**2])
    try:
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        score = _nrmse(pred, y)
        if score < best_score:
            best_score = score
            best_pred = pred
    except:
        pass
    
    # Cubic
    A = np.column_stack([np.ones(n), x, x**2, x**3])
    try:
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        score = _nrmse(pred, y)
        if score < best_score:
            best_score = score
            best_pred = pred
    except:
        pass
    
    # sin(x)
    A = np.column_stack([np.ones(n), x, np.sin(x), np.cos(x)])
    try:
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        score = _nrmse(pred, y)
        if score < best_score:
            best_score = score
            best_pred = pred
    except:
        pass
    
    # Multiple harmonics
    A = np.column_stack([np.ones(n)])
    for k in range(1, 5):
        A = np.column_stack([A, np.sin(k*x), np.cos(k*x)])
    try:
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        score = _nrmse(pred, y)
        if score < best_score:
            best_score = score
            best_pred = pred
    except:
        pass
    
    # exp forms
    if np.min(x) > 0.05:
        # log(x)
        A = np.column_stack([np.ones(n), np.log(x), x, x**2])
        try:
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ w
            score = _nrmse(pred, y)
            if score < best_score:
                best_score = score
                best_pred = pred
        except:
            pass
    
    # exp(b*x) + c via linearization: y - c = a*exp(b*x)
    # Try a few b values
    for b_val in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1.0, 2.0]:
        try:
            ex = np.exp(b_val * x)
            A = np.column_stack([np.ones(n), ex, x])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ w
            score = _nrmse(pred, y)
            if score < best_score:
                best_score = score
                best_pred = pred
        except:
            pass
    
    # Gaussian: a*exp(-b*(x-c)^2) + d
    # Grid search over c and b
    x_min, x_max = x.min(), x.max()
    for c_val in np.linspace(x_min, x_max, 8):
        for b_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                g = np.exp(-b_val * (x - c_val)**2)
                A = np.column_stack([np.ones(n), g, x])
                w, *_ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ w
                score = _nrmse(pred, y)
                if score < best_score:
                    best_score = score
                    best_pred = pred
            except:
                pass
    
    return best_pred, best_score


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n, d = X_train.shape
    x = X_train[:, 0]
    
    best_pred = y_train.copy()
    best_score = np.inf
    best_predict_fn = lambda X: best_pred  # fallback
    
    # Method 1: Polynomial OLS with various degrees
    for deg in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        if _remaining() < 5:
            break
        try:
            F = _poly_features(X_train, deg)
            w = _fit_ols_ridge(F, y_train, alpha=1e-6)
            pred = F @ w
            score = _nrmse(pred, y_train)
            if score < best_score:
                best_score = score
                best_pred = pred
                F_fn = lambda X, d=deg: _poly_features(X, d)
                best_predict_fn = lambda X, Ff=F_fn, ww=w: Ff(X) @ ww
        except Exception:
            pass
    
    # Method 2: Trigonometric
    for nh in [3, 5, 7, 10]:
        if _remaining() < 5:
            break
        try:
            F = _trig_features(X_train, nh)
            w = _fit_ols_ridge(F, y_train, alpha=1e-6)
            pred = F @ w
            score = _nrmse(pred, y_train)
            if score < best_score:
                best_score = score
                best_pred = pred
                F_fn = lambda X, n=nh: _trig_features(X, n)
                best_predict_fn = lambda X, Ff=F_fn, ww=w: Ff(X) @ ww
        except Exception:
            pass
    
    # Method 3: Mixed features
    for pd, nh in [(4, 3), (5, 4), (3, 5), (6, 3)]:
        if _remaining() < 5:
            break
        try:
            F = _mixed_features(X_train, pd, nh)
            w = _fit_ols_ridge(F, y_train, alpha=1e-6)
            pred = F @ w
            score = _nrmse(pred, y_train)
            if score < best_score:
                best_score = score
                best_pred = pred
                F_fn = lambda X, p=pd, n=nh: _mixed_features(X, p, n)
                best_predict_fn = lambda X, Ff=F_fn, ww=w: Ff(X) @ ww
        except Exception:
            pass
    
    # Method 4: RBF with grid of centers
    if _remaining() > 10:
        x_min, x_max = x.min(), x.max()
        for n_centers in [5, 8, 12, 15]:
            centers = np.linspace(x_min, x_max, n_centers)
            for bw in [0.5, 1.0, 2.0]:
                if _remaining() < 5:
                    break
                try:
                    F = _rbf_features(X_train, centers, [bw]*n_centers)
                    w = _fit_ols_ridge(F, y_train, alpha=1e-4)
                    pred = F @ w
                    score = _nrmse(pred, y_train)
                    if score < best_score:
                        best_score = score
                        best_pred = pred
                        c_copy = centers.copy()
                        b_copy = bw
                        F_fn = lambda X, c=c_copy, b=b_copy: _rbf_features(X, c, [b]*len(c))
                        best_predict_fn = lambda X, Ff=F_fn, ww=w: Ff(X) @ ww
                except Exception:
                    pass
    
    # Method 5: Spline interpolation
    if _remaining() > 5:
        try:
            x_sorted = np.sort(x)
            y_sorted = y_train[np.argsort(x)]
            n_knots = min(n, 30)
            if n_knots >= 5:
                knots = np.linspace(x_sorted[0], x_sorted[-1], n_knots)
                # Interpolate at knots
                y_at_knots = np.interp(knots, x_sorted, y_sorted)
                spl = make_interp_spline(knots, y_at_knots, k=3)
                
                def predict_spline(X):
                    xx = X[:, 0] if X.ndim == 2 else X
                    return spl(xx)
                
                pred = predict_spline(X_train)
                score = _nrmse(pred, y_train)
                if score < best_score:
                    best_score = score
                    best_pred = pred
                    best_predict_fn = predict_spline
        except Exception:
            pass
    
    # Method 6: MLP
    if _remaining() > 15:
        try:
            pred_fn = _fit_mlp(X_train, y_train, hidden_sizes=(32, 32), lr=0.01, epochs=300, seed=42)
            pred = pred_fn(X_train)
            score = _nrmse(pred, y_train)
            if score < best_score:
                best_score = score
                best_pred = pred
                best_predict_fn = pred_fn
        except Exception:
            pass
    
    # Method 7: Symbolic regression attempts
    if _remaining() > 10:
        try:
            sym_pred, sym_score = _try_symbolic_regression(X_train, y_train, time_budget=10)
            if sym_score < best_score:
                best_score = sym_score
                best_pred = sym_pred
                # Build a predict function based on the best symbolic form
                # For now, use the training predictions as a lookup
                x_train_sorted = np.sort(x)
                y_sorted = y_train[np.argsort(x)]
                from scipy.interpolate import interp1d
                f_interp = interp1d(x_train_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
                
                def predict_sym(X):
                    xx = X[:, 0] if X.ndim == 2 else X
                    return f_interp(xx)
                
                # Verify on training
                pred_check = predict_sym(X_train)
                score_check = _nrmse(pred_check, y_train)
                if score_check < best_score * 1.1:  # allow small tolerance
                    best_predict_fn = predict_sym
        except Exception:
            pass
    
    # Final safety: ensure predict returns finite values
    def safe_predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = best_predict_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
            return y
        except Exception:
            # Fallback: return mean of training y
            return np.full(len(X), np.mean(y_train), dtype=np.float64)
    
    return safe_predict
# EVOLVE-BLOCK-END
