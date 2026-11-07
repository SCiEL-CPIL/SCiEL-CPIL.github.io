# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import least_squares, minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import CubicSpline

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


def _fit_chebyshev(x, y, degree, lam=0.0):
    """Fit using Chebyshev polynomials with optional L2 regularization."""
    try:
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-12:
            return lambda xp: np.full_like(xp, np.mean(y))
        
        # Map to [-1, 1]
        xn = (2 * x - (x_min + x_max)) / (x_max - x_min)
        
        # Build Chebyshev basis matrix
        T = np.zeros((len(x), degree + 1))
        T[:, 0] = 1.0
        T[:, 1] = xn
        for k in range(2, degree + 1):
            T[:, k] = 2 * xn * T[:, k-1] - T[:, k-2]
        
        # Solve least squares (with optional regularization)
        if lam > 0:
            A = T.T @ T + lam * np.eye(degree + 1)
            b = T.T @ y
            coeffs = np.linalg.solve(A, b)
        else:
            coeffs, *_ = np.linalg.lstsq(T, y, rcond=None)
        
        def predict_fn(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xpn = (2 * xp - (x_min + x_max)) / (x_max - x_min)
            Tp = np.zeros((len(xp), degree + 1))
            Tp[:, 0] = 1.0
            Tp[:, 1] = xpn
            for k in range(2, degree + 1):
                Tp[:, k] = 2 * xpn * Tp[:, k-1] - Tp[:, k-2]
            return Tp @ coeffs
        
        return predict_fn
    except Exception:
        return None


def _fit_rational(x, y, num_deg=2, den_deg=2, max_iter=50, tol=1e-10):
    """Fit y ≈ P(x)/Q(x) using nonlinear least squares."""
    try:
        x_min, x_max = x.min(), x.max()
        rng = np.random.RandomState(42)
        
        # Normalize x to [-1, 1] for numerical stability
        xn = (2 * x - (x_min + x_max)) / (x_max - x_min)
        
        # Initialize coefficients
        p0 = rng.randn(num_deg + 1) * 0.1
        q0 = np.zeros(den_deg + 1)
        q0[0] = 1.0  # Normalize Q(0) = 1
        q0[1:] = rng.randn(den_deg) * 0.01
        
        def model(params, x_eval):
            p = params[:num_deg+1]
            q = params[num_deg+1:]
            P = np.zeros_like(x_eval)
            Q = np.zeros_like(x_eval)
            for i in range(num_deg, -1, -1):
                P = P * x_eval + p[i]
            for i in range(den_deg, -1, -1):
                Q = Q * x_eval + q[i]
            Q_safe = np.where(np.abs(Q) < 1e-10, 1e-10, Q)
            return P / Q_safe
        
        def residuals(params):
            pred = model(params, xn)
            return pred - y
        
        result = least_squares(residuals, np.concatenate([p0, q0]), 
                               method='lm', max_nfev=1000)
        
        if not result.success and result.fun is None:
            return None
        
        best_params = result.x
        best_p = best_params[:num_deg+1]
        best_q = best_params[num_deg+1:]
        
        # Check if denominator has roots in the training range
        q_roots = np.roots(best_q)
        real_roots = q_roots[np.abs(q_roots.imag) < 1e-8].real
        for root in real_roots:
            if x_min - 0.1 <= root <= x_max + 0.1:
                return None
        
        def predict_fn(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xpn = (2 * xp - (x_min + x_max)) / (x_max - x_min)
            P = np.zeros_like(xpn)
            Q = np.zeros_like(xpn)
            for i in range(num_deg, -1, -1):
                P = P * xpn + best_p[i]
            for i in range(den_deg, -1, -1):
                Q = Q * xpn + best_q[i]
            Q_safe = np.where(np.abs(Q) < 1e-10, 1e-10, Q)
            return P / Q_safe
        
        return predict_fn
    except Exception:
        return None


def _fit_spline(x, y, n_knots=20, s=0.0):
    """Fit a smoothing spline."""
    try:
        # Sort by x
        idx = np.argsort(x)
        xs = x[idx]
        ys = y[idx]
        
        # Create knot positions
        if n_knots > len(x):
            n_knots = len(x)
        
        # Use quantile-based knots for better distribution
        knot_positions = np.quantile(xs, np.linspace(0, 1, n_knots))
        
        # Fit spline
        try:
            spl = CubicSpline(xs, ys, bc_type='not-a-knot')
        except:
            return None
        
        def predict_fn(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            return spl(xp)
        
        return predict_fn
    except Exception:
        return None


def _fit_fourier(x, y, n_terms=10):
    """Fit using Fourier series."""
    try:
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min + 1e-8
        xn = (x - x_min) / x_range
        
        # Build Fourier basis
        F = np.zeros((len(x), 2 * n_terms + 1))
        F[:, 0] = 1.0
        for k in range(1, n_terms + 1):
            F[:, 2*k-1] = np.cos(2 * np.pi * k * xn)
            F[:, 2*k] = np.sin(2 * np.pi * k * xn)
        
        coeffs, *_ = np.linalg.lstsq(F, y, rcond=None)
        
        def predict_fn(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xpn = (xp - x_min) / x_range
            Fp = np.zeros((len(xp), 2 * n_terms + 1))
            Fp[:, 0] = 1.0
            for k in range(1, n_terms + 1):
                Fp[:, 2*k-1] = np.cos(2 * np.pi * k * xpn)
                Fp[:, 2*k] = np.sin(2 * np.pi * k * xpn)
            return Fp @ coeffs
        
        return predict_fn
    except Exception:
        return None


def _fit_gp(x, y, length_scale=None, noise=1e-6):
    """Gaussian Process with RBF kernel."""
    try:
        n = len(x)
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min + 1e-8
        xn = (x - x_min) / x_range
        
        if length_scale is None:
            dists = np.abs(xn[:, np.newaxis] - xn[np.newaxis, :])
            median_dist = np.median(dists[dists > 0])
            length_scale = max(median_dist, 0.05)
        
        def kernel(X1, X2, ls):
            diff = X1[:, np.newaxis] - X2[np.newaxis, :]
            sq_dist = (diff**2) / (ls**2)
            return np.exp(-0.5 * sq_dist)
        
        K = kernel(xn, xn, length_scale) + noise * np.eye(n)
        try:
            cf = cho_factor(K)
            alpha = cho_solve(cf, y)
        except:
            K += 1e-4 * np.eye(n)
            cf = cho_factor(K)
            alpha = cho_solve(cf, y)
        
        def predict_fn(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xpn = (xp - x_min) / x_range
            Ks = kernel(xpn, xn, length_scale)
            preds = Ks @ alpha
            return preds.flatten()
        
        return predict_fn
    except Exception:
        return None


def _fit_mlp(x, y, hidden_sizes=(128, 128), lr=0.005, epochs=2000, seed=42, batch_size=32):
    """MLP with Adam optimizer."""
    try:
        n = len(x)
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min + 1e-8
        xn = (x - x_min) / x_range
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma
        
        np.random.seed(seed)
        layers = [1] + list(hidden_sizes) + [1]
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
            if batch_size < n:
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    if _remaining() < 3.0:
                        break
                    batch_idx = idx[start:start+batch_size]
                    Xb = xn[batch_idx]
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
        
        def predict_fn(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xpn = (xp - x_min) / x_range
            a = xpn
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                a = np.tanh(z) if i < len(W) - 1 else z
            return a.flatten() * y_sigma + y_mu
        
        return predict_fn
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
    x = X_train[:, 0]
    y = y_train

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # Try various methods
    methods = []
    
    # Chebyshev fits (stable, global)
    for deg in [3, 5, 7, 10, 15, 20, 30]:
        for lam in [0.0, 1e-4, 1e-3]:
            methods.append(("cheb_%d_lam%s" % (deg, lam), 
                           lambda X, y, d=deg, l=lam: _fit_chebyshev(X[:,0], y, d, l)))
    
    # Rational function fits (structural change)
    for pd in [1, 2, 3]:
        for qd in [1, 2, 3]:
            if pd + qd <= 6:
                methods.append(("rational_%d_%d" % (pd, qd), 
                               lambda X, y, p=pd, q=qd: _fit_rational(X[:,0], y, p, q)))
    
    # Spline fits
    for nk in [10, 20, 30, 50]:
        methods.append(("spline_%d" % nk, lambda X, y, nk=nk: _fit_spline(X[:,0], y, nk)))
    
    # Fourier fits
    for nt in [5, 10, 15, 20]:
        methods.append(("fourier_%d" % nt, lambda X, y, nt=nt: _fit_fourier(X[:,0], y, nt)))
    
    # Gaussian Process
    for ls in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        methods.append(("gp_ls_%s" % ls, lambda X, y, ls=ls: _fit_gp(X[:,0], y, ls)))
    
    # MLP
    for hs in [(64, 64), (128, 128), (256, 128)]:
        for lr in [0.001, 0.005, 0.01]:
            methods.append(("mlp_%s_lr%s" % (hs, lr), 
                           lambda X, y, hs=hs, lr=lr: _fit_mlp(X[:,0], y, hs, lr, 1500, 42, 32)))
    
    # Sort methods by expected speed
    method_order = {
        "cheb": 0,
        "rational": 1,
        "spline": 2,
        "fourier": 3,
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
            best_predict = _safe_predict(predict)
    
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
