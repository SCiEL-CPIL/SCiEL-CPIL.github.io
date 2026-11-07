# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import UnivariateSpline, make_interp_spline, PchipInterpolator
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

_TOTAL_DEADLINE = time.time() + 320.0


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
        x = X[:, 0]
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


def _fit_spline_interp(X, y, num_knots=None):
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


def _fit_smooth_spline(X, y, s_factor=1.0):
    """Fit a smooth cubic spline using UnivariateSpline."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        
        var_y = np.var(ys)
        s = s_factor * var_y * n / 100
        
        try:
            sp = UnivariateSpline(xs, ys, s=s, k=3)
        except:
            sp = UnivariateSpline(xs, ys, s=var_y * n, k=3)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = sp(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_gp(X, y, length_scale=None, noise=1e-8):
    """Gaussian Process with RBF kernel."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        if length_scale is None:
            dists = np.sqrt(np.sum((Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :])**2, axis=2))
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


def _fit_rbf(X, y, num_centers=25, width=None):
    """RBF with fixed centers, optimized width."""
    try:
        n, d = X.shape
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        centers = np.linspace(x_min, x_max, num_centers)
        
        if width is None:
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


def _fit_mlp(X, y, hidden_sizes=(64, 64), lr=0.01, epochs=3000, seed=42, batch_size=32):
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
            if _remaining() < 5.0:
                break
            if batch_size < n:
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    if _remaining() < 5.0:
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


def _fit_fourier(X, y, num_terms=10):
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


def _fit_pchip(X, y):
    """PCHIP (monotone cubic) interpolation - no overshoot."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        pchip = PchipInterpolator(xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = pchip(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_deriv_spline(X, y, s_factor=1.0):
    """Smooth spline with derivative penalty: fit values AND match finite-difference slopes."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        
        # Compute finite-difference derivatives
        dx = np.diff(xs)
        dy = np.diff(ys)
        # Central differences where possible, forward/backward at edges
        derivs = np.zeros(n)
        if n > 2:
            derivs[1:-1] = (ys[2:] - ys[:-2]) / (xs[2:] - xs[:-2])
            derivs[0] = (ys[1] - ys[0]) / dx[0]
            derivs[-1] = (ys[-1] - ys[-2]) / dx[-1]
        
        var_y = np.var(ys)
        var_d = np.var(derivs) + 1e-12
        
        # Fit spline to values with smoothing, then check derivative match
        s_val = s_factor * var_y * n / 100.0
        try:
            sp = UnivariateSpline(xs, ys, s=s_val, k=3)
        except:
            sp = UnivariateSpline(xs, ys, s=var_y * n, k=3)
        
        # Evaluate how well derivatives match
        sp_derivs = sp.derivative()(xs)
        deriv_error = np.mean((sp_derivs - derivs) ** 2) / var_d
        
        # If derivative error is large, try a stiffer spline
        if deriv_error > 0.5:
            for s_try in [s_val * 0.1, s_val * 0.01, s_val * 0.001]:
                try:
                    sp2 = UnivariateSpline(xs, ys, s=max(s_try, 1e-10), k=3)
                    sp2_derivs = sp2.derivative()(xs)
                    de2 = np.mean((sp2_derivs - derivs) ** 2) / var_d
                    val_err2 = np.mean((sp2(xs) - ys) ** 2) / var_y
                    if de2 < deriv_error and val_err2 < 2.0:
                        sp = sp2
                        break
                except:
                    continue
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = sp(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_deriv_poly(X, y, degree=8):
    """Polynomial fit that also matches derivative constraints via augmented system."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        
        # Compute derivatives
        derivs = np.zeros(n)
        if n > 2:
            derivs[1:-1] = (ys[2:] - ys[:-2]) / (xs[2:] - xs[:-2])
            derivs[0] = (ys[1] - ys[0]) / (xs[1] - xs[0])
            derivs[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        
        # Build augmented system: [V; V'] [coeffs] ≈ [y; w*derivs]
        # V is Vandermonde, V' is its derivative
        V = np.vander(xs, N=degree + 1, increasing=True)
        # Derivative of polynomial: if p(x) = sum c_i x^i, p'(x) = sum i*c_i*x^(i-1)
        Vp = np.zeros_like(V)
        for i in range(1, degree + 1):
            Vp[:, i - 1] = i * xs ** (i - 1)
        
        # Weight: balance value fit and derivative fit
        var_y = np.var(ys)
        var_d = np.var(derivs) + 1e-12
        w = 0.5 * np.sqrt(var_y / var_d)
        
        A = np.vstack([V, w * Vp])
        b = np.concatenate([ys, w * derivs])
        
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            return np.polyval(coeffs, xp)
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_radial_basis_learned(X, y, num_centers=20):
    """RBF with learned centers and width via optimization."""
    try:
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        centers = np.linspace(x_min, x_max, num_centers)
        
        def build_features(xp, centers, width):
            return np.column_stack([np.exp(-0.5 * ((xp - c) / width) ** 2) for c in centers])
        
        def objective(params):
            c = params[:num_centers]
            w = params[num_centers]
            w = max(w, 0.01)
            F = build_features(x, c, w)
            try:
                coeff, *_ = np.linalg.lstsq(F, y, rcond=None)
                pred = F @ coeff
                return np.mean((pred - y) ** 2)
            except:
                return 1e10
        
        best_rmse = np.inf
        best_params = None
        
        for w_init in [0.2, 0.5, 1.0, 2.0, 3.0]:
            params0 = np.concatenate([centers, [w_init]])
            try:
                res = minimize(objective, params0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 0.01, 'fatol': 1e-6})
                if res.fun < best_rmse:
                    best_rmse = res.fun
                    best_params = res.x
            except:
                continue
        
        if best_params is None:
            return None
        
        best_centers = best_params[:num_centers]
        best_width = max(best_params[num_centers], 0.01)
        
        F = build_features(x, best_centers, best_width)
        coeff, *_ = np.linalg.lstsq(F, y, rcond=None)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            Fp = build_features(xp, best_centers, best_width)
            return Fp @ coeff
        return _safe_predict(predict)
    except Exception:
        return None


def _try_method(method_fn, X, y):
    """Try a fitting method, return (predict_fn, train_rmse) or (None, inf)."""
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

    # Hold-out validation split for model selection
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    val_frac = min(0.2, max(0.1, 40.0 / n))
    val_size = int(n * val_frac)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    
    var_y_val = np.var(y_val)
    if var_y_val < 1e-12:
        var_y_val = 1.0
    
    best_predict = None
    best_val_nmse = np.inf

    # Baseline: constant
    const = np.mean(y_tr)
    const_pred = np.full(X_val.shape[0], const)
    const_val_nmse = np.mean((const_pred - y_val) ** 2) / var_y_val
    best_val_nmse = const_val_nmse
    best_predict = _safe_predict(lambda Xp: np.full(np.asarray(Xp).shape[0], const))

    # Track top models for ensemble
    top_models = []  # list of (val_nmse, method)
    
    methods = []
    
    # Polynomial fits (fast)
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        methods.append(("poly_%d" % deg, lambda X, y, d=deg: _fit_poly(X, y, d)))
    
    # Derivative-constrained polynomial (fast, strong smoothness prior)
    for deg in [4, 6, 8, 10, 12]:
        methods.append(("deriv_poly_%d" % deg, lambda X, y, d=deg: _fit_deriv_poly(X, y, d)))
    
    # Smooth splines (fast)
    for sf in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        methods.append(("smooth_spline_%s" % sf, lambda X, y, sf=sf: _fit_smooth_spline(X, y, sf)))
    
    # Derivative-augmented spline (fast, captures derivative structure)
    for sf in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        methods.append(("deriv_spline_%s" % sf, lambda X, y, sf=sf: _fit_deriv_spline(X, y, sf)))
    
    # PCHIP interpolation (fast, monotone - no overshoot)
    methods.append(("pchip", lambda X, y: _fit_pchip(X, y)))
    
    # Interpolation splines (fast)
    for nk in [10, 15, 25, 40, 60]:
        methods.append(("spline_interp_%d" % nk, lambda X, y, nk=nk: _fit_spline_interp(X, y, nk)))
    
    # Fourier (fast)
    for nt in [3, 5, 8, 12, 15, 20]:
        methods.append(("fourier_%d" % nt, lambda X, y, nt=nt: _fit_fourier(X, y, nt)))
    
    # RBF fixed (medium) - reduced set
    for nc in [10, 20, 40]:
        for w in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            methods.append(("rbf_%d_w%s" % (nc, w), lambda X, y, nc=nc, w=w: _fit_rbf(X, y, nc, w)))
    
    # GP (medium) - fewer configs
    for ls in [0.2, 0.5, 1.0, 2.0, 3.0]:
        methods.append(("gp_ls_%s" % ls, lambda X, y, ls=ls: _fit_gp(X, y, length_scale=ls)))
    
    # Sort by expected speed
    method_order = {
        "poly": 0,
        "deriv_poly": 0,
        "smooth_spline": 1,
        "deriv_spline": 1,
        "pchip": 1,
        "spline_interp": 2,
        "fourier": 3,
        "rbf_": 4,
        "gp": 5,
    }
    
    def sort_key(item):
        name = item[0]
        for prefix, priority in method_order.items():
            if name.startswith(prefix):
                return priority
        return 10
    
    methods.sort(key=sort_key)
    
    for name, method in methods:
        if _remaining() < 10.0:
            break
        # Fit on the training subset
        try:
            predict = method(X_tr, y_tr)
        except Exception:
            predict = None
        if predict is None:
            continue
        
        # Evaluate on the validation subset
        try:
            val_pred = predict(X_val)
            val_pred = np.asarray(val_pred, dtype=np.float64)
            if not np.all(np.isfinite(val_pred)):
                continue
            val_nmse = np.mean((val_pred - y_val) ** 2) / var_y_val
        except Exception:
            continue
        
        if not np.isfinite(val_nmse):
            continue
        
        # Track top 5 for ensemble
        if len(top_models) < 5:
            top_models.append((val_nmse, method))
            top_models.sort(key=lambda t: t[0])
        elif val_nmse < top_models[-1][0]:
            top_models[-1] = (val_nmse, method)
            top_models.sort(key=lambda t: t[0])
        
        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            # Refit on the full dataset for the best model
            try:
                full_predict = method(X_train, y_train)
                if full_predict is not None:
                    full_pred = full_predict(X_train)
                    full_pred = np.asarray(full_pred, dtype=np.float64)
                    if np.all(np.isfinite(full_pred)):
                        best_predict = full_predict
                else:
                    best_predict = predict
            except Exception:
                best_predict = predict
    
    # Ensemble: average top models if we have at least 2 good ones
    if len(top_models) >= 2 and _remaining() > 15.0:
        # Only ensemble if top models are all reasonably good
        if top_models[-1][0] < best_val_nmse * 2.0 + 0.01:
            ensemble_preds_val = []
            ensemble_methods = []
            for nmse, method in top_models[:4]:
                try:
                    full_p = method(X_train, y_train)
                    if full_p is not None:
                        fp = full_p(X_train)
                        fp = np.asarray(fp, dtype=np.float64)
                        if np.all(np.isfinite(fp)):
                            vp = full_p(X_val)
                            vp = np.asarray(vp, dtype=np.float64)
                            if np.all(np.isfinite(vp)):
                                ensemble_preds_val.append(vp)
                                ensemble_methods.append(full_p)
                except Exception:
                    continue
            
            if len(ensemble_preds_val) >= 2:
                # Weighted average (inverse NMSE weights)
                weights = np.array([1.0 / (nmse + 1e-10) for nmse, _ in top_models[:len(ensemble_preds_val)]])
                weights /= weights.sum()
                
                ens_val_pred = np.zeros(X_val.shape[0])
                for w_i, p in zip(weights, ensemble_preds_val):
                    ens_val_pred += w_i * p
                
                ens_val_nmse = np.mean((ens_val_pred - y_val) ** 2) / var_y_val
                
                if ens_val_nmse < best_val_nmse:
                    # Build ensemble predict
                    ens_methods = ensemble_methods[:]
                    ens_weights = weights[:]
                    def ensemble_predict(Xp):
                        Xp = np.asarray(Xp, dtype=np.float64)
                        if Xp.ndim == 1:
                            Xp = Xp.reshape(1, -1)
                        result = np.zeros(Xp.shape[0])
                        for w_i, m in zip(ens_weights, ens_methods):
                            try:
                                p = m(Xp)
                                p = np.asarray(p, dtype=np.float64)
                                if np.all(np.isfinite(p)):
                                    result += w_i * p
                            except Exception:
                                pass
                        return result
                    best_predict = _safe_predict(ensemble_predict)
    
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(np.asarray(Xp).shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
