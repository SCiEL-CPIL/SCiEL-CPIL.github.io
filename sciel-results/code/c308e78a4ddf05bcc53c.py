# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline, CubicSpline
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

_TOTAL_DEADLINE = time.time() + 310.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _fit_poly(X, y, degree):
    """Direct polynomial fit."""
    try:
        x = X[:, 0] if X.shape[1] == 1 else X[:, 0]
        coeffs = np.polyfit(x, y, degree)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            return np.polyval(coeffs, xp)
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_spline(X, y, num_knots=None):
    """Fit a cubic spline via interpolation on sorted points."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if num_knots is None:
            num_knots = min(n, max(10, n // 2))
        if n < num_knots + 4:
            num_knots = max(4, n - 4)
        knot_idx = np.linspace(0, n - 1, num_knots, dtype=int)
        knot_x = xs[knot_idx]
        knot_y = ys[knot_idx]
        s = make_interp_spline(knot_x, knot_y, k=3)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_gp(X, y, length_scale=None, noise=1e-6):
    """Gaussian Process with RBF kernel."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        if length_scale is None:
            # Estimate length scale from median distance
            dists = cdist(Xn, Xn)
            median_dist = np.median(dists[dists > 0])
            length_scale = max(median_dist, 0.1)
        
        def kernel(X1, X2, ls):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (ls**2)
            return np.exp(-0.5 * sq_dist)
        
        K = kernel(Xn, Xn, length_scale) + noise * np.eye(n)
        try:
            cf = cho_factor(K)
            alpha = cho_solve(cf, y)
        except:
            K += 1e-4 * np.eye(n)
            cf = cho_factor(K)
            alpha = cho_solve(cf, y)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn, length_scale)
            preds = Ks @ alpha
            return preds.flatten()
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_rbf(X, y, num_centers=30, width=None):
    """RBF with fixed centers, optimized width."""
    try:
        n, d = X.shape
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        centers = np.linspace(x_min, x_max, num_centers)
        
        if width is None:
            # Try multiple widths
            best_rmse = np.inf
            best_width = 1.0
            for w in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                F = np.column_stack([np.exp(-0.5 * ((x - c) / w) ** 2) for c in centers])
                try:
                    w_coeff, *_ = np.linalg.lstsq(F, y, rcond=None)
                    pred = F @ w_coeff
                    rmse = np.sqrt(np.mean((pred - y) ** 2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_width = w
                except:
                    continue
            width = best_width
        
        F = np.column_stack([np.exp(-0.5 * ((x - c) / width) ** 2) for c in centers])
        w_coeff, *_ = np.linalg.lstsq(F, y, rcond=None)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            Fp = np.column_stack([np.exp(-0.5 * ((xp - c) / width) ** 2) for c in centers])
            return Fp @ w_coeff
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_mlp(X, y, hidden_sizes=(128, 128), lr=0.005, epochs=2000, seed=42, batch_size=32):
    """MLP with Adam optimizer and mini-batch training."""
    try:
        d = X.shape[1]
        n = X.shape[0]
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma
        
        np.random.seed(seed)
        layers = [d] + list(hidden_sizes) + [1]
        W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(len(layers)-1)]
        b = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]
        
        mW = [np.zeros_like(w) for w in W]
        vW = [np.zeros_like(w) for w in W]
        mb = [np.zeros_like(bb) for bb in b]
        vb = [np.zeros_like(bb) for bb in b]
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        def forward(Xb):
            a = Xb
            acts = [a]
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                if i < len(W) - 1:
                    a = np.tanh(z)
                else:
                    a = z
                acts.append(a)
            return acts
        
        for epoch in range(epochs):
            if _remaining() < 3.0:
                break
            # Mini-batch
            if batch_size < n:
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    if _remaining() < 3.0:
                        break
                    batch_idx = idx[start:start+batch_size]
                    Xb = Xn[batch_idx]
                    yb = yn[batch_idx]
                    bs = len(batch_idx)
                    
                    t += 1
                    acts = forward(Xb)
                    pred = acts[-1]
                    dz = (pred - yb) / bs
                    
                    grads_W = [None] * len(W)
                    grads_b = [None] * len(b)
                    for i in range(len(W) - 1, -1, -1):
                        if i == len(W) - 1:
                            dz_i = dz
                        else:
                            dz_i = dz @ W[i].T * (1 - acts[i+1]**2)
                        grads_W[i] = acts[i].T @ dz_i
                        grads_b[i] = dz_i.sum(axis=0)
                    
                    for i in range(len(W)):
                        mW[i] = beta1 * mW[i] + (1 - beta1) * grads_W[i]
                        vW[i] = beta2 * vW[i] + (1 - beta2) * grads_W[i]**2
                        mW_hat = mW[i] / (1 - beta1**t)
                        vW_hat = vW[i] / (1 - beta2**t)
                        W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                        
                        mb[i] = beta1 * mb[i] + (1 - beta1) * grads_b[i]
                        vb[i] = beta2 * vb[i] + (1 - beta2) * grads_b[i]**2
                        mb_hat = mb[i] / (1 - beta1**t)
                        vb_hat = vb[i] / (1 - beta2**t)
                        b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
            else:
                # Full batch
                t += 1
                acts = forward(Xn)
                pred = acts[-1]
                dz = (pred - yn) / n
                
                grads_W = [None] * len(W)
                grads_b = [None] * len(b)
                for i in range(len(W) - 1, -1, -1):
                    if i == len(W) - 1:
                        dz_i = dz
                    else:
                        dz_i = dz @ W[i].T * (1 - acts[i+1]**2)
                    grads_W[i] = acts[i].T @ dz_i
                    grads_b[i] = dz_i.sum(axis=0)
                
                for i in range(len(W)):
                    mW[i] = beta1 * mW[i] + (1 - beta1) * grads_W[i]
                    vW[i] = beta2 * vW[i] + (1 - beta2) * grads_W[i]**2
                    mW_hat = mW[i] / (1 - beta1**t)
                    vW_hat = vW[i] / (1 - beta2**t)
                    W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                    
                    mb[i] = beta1 * mb[i] + (1 - beta1) * grads_b[i]
                    vb[i] = beta2 * vb[i] + (1 - beta2) * grads_b[i]**2
                    mb_hat = mb[i] / (1 - beta1**t)
                    vb_hat = vb[i] / (1 - beta2**t)
                    b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            a = Xpn
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                a = np.tanh(z) if i < len(W) - 1 else z
            return a.flatten() * y_sigma + y_mu
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_fourier(X, y, num_terms=15):
    """Fourier series fit."""
    try:
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        period = x_max - x_min + 1e-8
        
        cols = [np.ones_like(x)]
        for k in range(1, num_terms + 1):
            cols.append(np.sin(2 * np.pi * k * x / period))
            cols.append(np.cos(2 * np.pi * k * x / period))
        
        F = np.column_stack(cols)
        coeffs, *_ = np.linalg.lstsq(F, y, rcond=None)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            cols = [np.ones_like(xp)]
            for k in range(1, num_terms + 1):
                cols.append(np.sin(2 * np.pi * k * xp / period))
                cols.append(np.cos(2 * np.pi * k * xp / period))
            Fp = np.column_stack(cols)
            return Fp @ coeffs
        return _safe_predict(predict)
    except Exception:
        return None


def _try_method(method_fn, X, y):
    """Try a fitting method, return (predict_fn, rmse) or (None, inf)."""
    try:
        predict = method_fn(X, y)
        if predict is None:
            return None, np.inf
        pred = predict(X)
        pred = np.asarray(pred, dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        if not np.isfinite(rmse):
            return None, np.inf
        return predict, rmse
    except Exception:
        return None, np.inf


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # Try various methods
    methods = []
    
    # Polynomial fits
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        methods.append(("poly_%d" % deg, lambda X, y, d=deg: _fit_poly(X, y, d)))
    
    # Spline fits
    for nk in [15, 25, 40, 60, 80, 100]:
        methods.append(("spline_%d" % nk, lambda X, y, nk=nk: _fit_spline(X, y, nk)))
    
    # Gaussian Process with various length scales
    for ls in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
        methods.append(("gp_ls_%s" % ls, lambda X, y, ls=ls: _fit_gp(X, y, length_scale=ls)))
    
    # RBF with various centers and widths
    for nc in [15, 25, 40, 60]:
        for w in [0.1, 0.3, 0.5, 1.0, 2.0]:
            methods.append(("rbf_%d_w%s" % (nc, w), lambda X, y, nc=nc, w=w: _fit_rbf(X, y, nc, w)))
    
    # Fourier
    for nt in [5, 10, 15, 20]:
        methods.append(("fourier_%d" % nt, lambda X, y, nt=nt: _fit_fourier(X, y, nt)))
    
    # MLP with various architectures
    for hs in [(64, 64), (128, 128), (256, 128), (128, 256)]:
        for lr in [0.001, 0.005, 0.01]:
            methods.append(("mlp_%s_lr%s" % (hs, lr), lambda X, y, hs=hs, lr=lr: _fit_mlp(X, y, hs, lr, 1500, 42, 32)))
    
    # Sort methods by expected speed (fast first)
    # Polynomials and splines are fast, GP/RBF medium, MLP slow
    method_order = {
        "poly": 0,
        "spline": 1,
        "fourier": 2,
        "rbf": 3,
        "gp": 4,
        "mlp": 5,
    }
    
    def sort_key(item):
        name = item[0]
        for prefix, priority in method_order.items():
            if name.startswith(prefix):
                return priority
        return 10
    
    methods.sort(key=sort_key)
    
    for name, method in methods:
        if _remaining() < 5.0:
            break
        predict, rmse = _try_method(method, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict
    
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
