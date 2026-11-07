# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline, PchipInterpolator, CubicSpline, UnivariateSpline
from scipy.optimize import minimize_scalar, differential_evolution, minimize

_TOTAL_DEADLINE = time.time() + 325.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
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


def _poly_features(X, degree, scale=None):
    x = X[:, 0]
    if scale is None:
        scale = (x.min() + x.max()) / 2.0
    x_shifted = x - scale
    cols = [np.ones(X.shape[0])]
    for p in range(1, degree + 1):
        cols.append(x_shifted ** p)
    return np.column_stack(cols)


def _trig_features(X, max_k=10):
    x = X[:, 0]
    cols = [np.ones(X.shape[0])]
    for k in range(1, max_k + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _exp_features(X):
    x = X[:, 0]
    cols = [np.ones(X.shape[0])]
    cols.append(np.exp(x))
    cols.append(np.exp(-x))
    cols.append(np.exp(-x**2))
    cols.append(np.exp(-x**2 / 2))
    cols.append(np.sqrt(np.abs(x) + 1e-9))
    cols.append(np.log(np.abs(x) + 1e-9))
    cols.append(1.0 / (np.abs(x) + 1e-6))
    cols.append(x * np.exp(-x))
    cols.append(x * np.exp(x))
    cols.append(np.tanh(x))
    cols.append(np.arctan(x))
    return np.column_stack(cols)


def _mixed_features(X):
    x = X[:, 0]
    cols = [np.ones(X.shape[0])]
    for p in range(1, 8):
        cols.append(x ** p)
    for k in range(1, 8):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    cols.append(np.tanh(x))
    cols.append(np.exp(-x ** 2))
    cols.append(np.exp(x))
    cols.append(np.exp(-x))
    cols.append(np.log(np.abs(x) + 1e-9))
    cols.append(1.0 / (np.abs(x) + 1e-6))
    cols.append(np.sqrt(np.abs(x) + 1e-9))
    return np.column_stack(cols)


def _fit_ols(F, y, ridge_alpha=1e-10):
    try:
        n, p = F.shape
        A = F.copy()
        A[:, 0] *= np.sqrt(1 + ridge_alpha)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-10):
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        w = _fit_ols(F, y, ridge_alpha)
        if w is None:
            return None, np.inf
        pred = F @ w
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        predict = _safe_predict(lambda Xp: feat_fn(Xp) @ w)
        return predict, rmse
    except Exception:
        return None, np.inf


def _fit_spline_interp(X, y, num_knots=20, method='cubic'):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 10:
            return None
        num_knots = min(num_knots, n)
        if num_knots < 5:
            num_knots = 5
        knot_idx = np.linspace(0, n - 1, num_knots, dtype=int)
        knot_x = xs[knot_idx]
        knot_y = ys[knot_idx]
        
        if method == 'pchip':
            s = PchipInterpolator(knot_x, knot_y)
        else:
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


def _fit_smooth_spline(X, y):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 10:
            return None, np.inf
        
        best_predict = None
        best_rmse = np.inf
        
        for s in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
            try:
                spl = UnivariateSpline(xs, ys, s=s)
                pred = spl(xs)
                rmse = np.sqrt(np.mean((pred - ys) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_predict = _safe_predict(lambda Xp, spl=spl: np.nan_to_num(spl(Xp[:, 0]), nan=0.0, posinf=1e12, neginf=-1e12))
            except Exception:
                continue
        
        return best_predict, best_rmse
    except Exception:
        return None, np.inf


def _fit_kernel_ridge(X, y, length_scale=1.0, alpha=1e-6):
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
        K = np.exp(-0.5 * sq_dist) + alpha * np.eye(n)
        
        w = np.linalg.solve(K, y)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            diff = Xpn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            Ks = np.exp(-0.5 * sq_dist)
            preds = Ks @ w
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _rbf_features(X, centers, widths):
    cols = [np.ones(X.shape[0])]
    for c in centers:
        for w in widths:
            dist = np.abs(X[:, 0] - c)
            cols.append(np.exp(-0.5 * (dist / w) ** 2))
    return np.column_stack(cols)


def _fit_rbf_optimized(X, y, num_centers=20):
    try:
        n, d = X.shape
        if n < num_centers + 4:
            num_centers = max(5, n - 4)
        
        x = X[:, 0]
        centers = np.quantile(x, np.linspace(0.02, 0.98, num_centers))
        
        def objective(params):
            if _remaining() < 2.0:
                return 1e12
            c = params[:num_centers]
            w = np.exp(params[num_centers:])
            F = _rbf_features(X, c, w)
            if not np.all(np.isfinite(F)):
                return 1e12
            try:
                ww, *_ = np.linalg.lstsq(F, y, rcond=None)
                pred = F @ ww
                mse = np.mean((pred - y)**2)
                return mse if np.isfinite(mse) else 1e12
            except Exception:
                return 1e12
        
        x_range = x.max() - x.min()
        init_widths = np.full(num_centers, x_range / (2 * num_centers))
        init_params = np.concatenate([centers, np.log(init_widths)])
        
        result = minimize(objective, init_params, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-8})
        
        if result.fun < 1e10 and np.isfinite(result.fun):
            best_params = result.x
            c = best_params[:num_centers]
            w = np.exp(best_params[num_centers:])
            F = _rbf_features(X, c, w)
            ww, *_ = np.linalg.lstsq(F, y, rcond=None)
            
            def predict(Xp):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                Fp = _rbf_features(Xp, c, w)
                return Fp @ ww
            return _safe_predict(predict)
    except Exception:
        return None


def _fit_mlp(X, y, hidden_sizes=(128, 64), lr=0.01, epochs=500, seed=42):
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
        return predict
    except Exception:
        return None


def _fit_fourier(X, y, max_harmonics=20):
    try:
        x = X[:, 0]
        best_rmse = np.inf
        best_predict = None
        
        for freq_mult in [0.5, 1.0, 2.0, np.pi, 2*np.pi, 0.25, 4.0, 0.1, 0.75, 1.5, 3.0, 5.0]:
            fundamental = freq_mult
            cols = [np.ones(X.shape[0])]
            for k in range(1, max_harmonics + 1):
                cols.append(np.sin(k * fundamental * x))
                cols.append(np.cos(k * fundamental * x))
            F = np.column_stack(cols)
            if not np.all(np.isfinite(F)):
                continue
            w = _fit_ols(F, y)
            if w is None:
                continue
            pred = F @ w
            if not np.all(np.isfinite(pred)):
                continue
            rmse = np.sqrt(np.mean((pred - y) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                fund_cached = fundamental
                max_harm_cached = max_harmonics
                w_cached = w
                best_predict = _safe_predict(lambda Xp, f=fund_cached, m=max_harm_cached, w=w_cached: 
                    np.column_stack([np.ones(Xp.shape[0])] + [np.sin(k*f*Xp[:,0]) for k in range(1, m+1)] + [np.cos(k*f*Xp[:,0]) for k in range(1, m+1)]) @ w)
        
        if best_predict is not None and best_rmse < np.inf:
            return best_predict, best_rmse
    except Exception:
        pass
    return None, np.inf


def _fit_neural_ode(X, y, hidden_dim=32, lr=0.005, epochs=300, num_steps=20, seed=42):
    """Vectorized Neural ODE: learn dy/dx = f(x, y, theta) via Euler integration."""
    try:
        n = X.shape[0]
        x = X[:, 0]
        
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min + 1e-8
        x_norm = (x - x_min) / x_range
        
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        y_norm = (y - y_mu) / y_sigma
        
        order = np.argsort(x_norm)
        x_sorted = x_norm[order]
        y_sorted = y_norm[order]
        
        np.random.seed(seed)
        W1 = np.random.randn(hidden_dim, 3) * 0.5
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(1, hidden_dim) * 0.3
        b2 = np.zeros(1)
        
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t_adam = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        dt = 1.0 / (n - 1) if n > 1 else 1.0
        
        def integrate(W1, b1, W2, b2, x_vals, y_start):
            """Vectorized Euler integration from t=0 to each x_val."""
            n_pts = len(x_vals)
            y_traj = np.zeros((n_pts,))
            y_traj[0] = y_start
            
            # Build input matrix: [x, y, 1] for each step
            # We integrate sequentially but vectorize the network eval
            for i in range(1, n_pts):
                inp = np.array([x_vals[i-1], y_traj[i-1], 1.0])
                h = np.tanh(W1 @ inp + b1)
                dydt = (W2 @ h + b2).flatten()[0]
                y_traj[i] = y_traj[i-1] + dydt * dt
            
            return y_traj
        
        for epoch in range(epochs):
            if _remaining() < 3.0:
                break
            t_adam += 1
            
            # Forward: integrate from first point
            y_pred = integrate(W1, b1, W2, b2, x_sorted, y_sorted[0])
            
            loss = np.mean((y_pred - y_sorted) ** 2)
            if not np.isfinite(loss):
                break
            
            # Backprop through Euler steps (unrolled)
            grads_W1 = np.zeros_like(W1)
            grads_b1 = np.zeros_like(b1)
            grads_W2 = np.zeros_like(W2)
            grads_b2 = np.zeros_like(b2)
            
            # dL/dy_traj[i] = 2*(y_pred[i] - y_sorted[i]) / n
            dy_traj = 2.0 * (y_pred - y_sorted) / n
            
            # Backprop through Euler steps
            dy_next = dy_traj[-1]
            for i in range(n - 1, 0, -1):
                inp = np.array([x_sorted[i-1], y_traj[i-1], 1.0])
                h = np.tanh(W1 @ inp + b1)
                dh = 1 - h**2
                
                # dy_traj[i] = dy_traj[i-1] + (W2 @ h + b2) * dt
                # dL/dy_traj[i-1] += dL/dy_traj[i] * dt * W2 @ dh * W1
                dW2 = np.outer(dy_next * dt, h)
                db2 = np.array([dy_next * dt])
                dh_dL = (W2.T @ (dy_next * dt)).reshape(-1, 1) * dh
                
                dW1 = np.outer(dh_dL.flatten(), inp)
                db1 = dh_dL.flatten()
                
                grads_W2 += dW2
                grads_b2 += db2
                grads_W1 += dW1
                grads_b1 += db1
                
                # dL/dy_traj[i-1] from direct term + backprop
                dy_next = dy_traj[i-1] + (dh_dL.flatten() @ W1[0]) * dt
                # Actually: dL/dy_traj[i-1] gets contribution from dL/dy_traj[i] * dt * W2 @ dh * W1[1]
                # W1[1] is the second row (coefficient of y in input)
                dy_next = dy_traj[i-1] + (dh_dL.flatten() @ W1[1]) * dt
            
            # Add contribution from initial condition
            # dy_traj[0] = y_start, so dL/dy_start = dy_next (accumulated)
            # But y_start is fixed, so we don't backprop further
            
            # Adam update
            for pi, p in enumerate(params):
                g = [grads_W1, grads_b1, grads_W2, grads_b2][pi]
                m[pi] = beta1 * m[pi] + (1 - beta1) * g
                v[pi] = beta2 * v[pi] + (1 - beta2) * g**2
                m_hat = m[pi] / (1 - beta1**t_adam)
                v_hat = v[pi] / (1 - beta2**t_adam)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Final prediction function
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xp_norm = (xp - x_min) / x_range
            
            preds = np.zeros(Xp.shape[0])
            for i in range(Xp.shape[0]):
                # Find position in sorted training x
                idx_float = np.searchsorted(x_sorted, xp_norm[i])
                idx_float = np.clip(idx_float, 1, n - 1)
                idx = int(idx_float)
                # Interpolate between idx-1 and idx
                frac = (xp_norm[i] - x_sorted[idx-1]) / (x_sorted[idx] - x_sorted[idx-1] + 1e-10)
                frac = np.clip(frac, 0, 1)
                # Integrate to get y at this point
                # Use the trained ODE
                y_start = y_sorted[0]
                y_val = y_start
                for j in range(1, idx):
                    inp = np.array([x_sorted[j-1], y_val, 1.0])
                    h = np.tanh(W1 @ inp + b1)
                    dydt = (W2 @ h + b2).flatten()[0]
                    y_val = y_val + dydt * dt
                # Partial step for fractional part
                inp = np.array([x_sorted[idx-1], y_val, 1.0])
                h = np.tanh(W1 @ inp + b1)
                dydt = (W2 @ h + b2).flatten()[0]
                y_val = y_val + dydt * dt * frac
                
                preds[i] = y_val * y_sigma + y_mu
            
            return np.nan_to_num(preds, nan=y_mu, posinf=1e12, neginf=-1e12)
        
        pred_train = predict(X)
        rmse = np.sqrt(np.mean((pred_train - y) ** 2))
        if not np.isfinite(rmse):
            return None, np.inf
        return _safe_predict(predict), rmse
    except Exception:
        return None, np.inf


def _fit_gaussian_process_optimized(X, y):
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_std = y.std() + 1e-8
        y_norm = (y - y_mu) / y_std
        
        def kernel(X1, X2, length_scale, variance):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            return variance * np.exp(-0.5 * sq_dist)
        
        def neg_log_likelihood(params):
            length_scale = np.exp(params[0])
            variance = np.exp(params[1])
            noise = np.exp(params[2])
            
            K = kernel(Xn, Xn, length_scale, variance) + noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                mse_term = y_norm @ alpha
                logdet = 2 * np.sum(np.log(np.diag(L)))
                nll = 0.5 * (n * np.log(2 * np.pi) + logdet + mse_term)
                return nll
            except Exception:
                return 1e10
        
        best_params = None
        best_nll = np.inf
        for init in [np.array([0.0, 0.0, -5.0]), np.array([-0.5, 0.5, -5.0]), np.array([0.5, 0.5, -5.0]), np.array([-1.0, 1.0, -6.0])]:
            result = minimize(neg_log_likelihood, init, method='Nelder-Mead', options={'maxiter': 150})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        
        if best_params is None:
            return None, np.inf
        
        length_scale = np.exp(best_params[0])
        variance = np.exp(best_params[1])
        noise = np.exp(best_params[2])
        
        K = kernel(Xn, Xn, length_scale, variance) + noise * np.eye(n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn, length_scale, variance)
            preds = (Ks @ alpha).flatten() * y_std + y_mu
            return np.nan_to_num(preds, nan=y_mu, posinf=1e12, neginf=-1e12)
        
        pred_train = predict(X)
        rmse = np.sqrt(np.mean((pred_train - y) ** 2))
        return _safe_predict(predict), rmse
    except Exception:
        return None, np.inf


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    # Hold-out validation split for model selection
    np.random.seed(12345)
    indices = np.random.permutation(n)
    val_frac = 0.2
    n_val = max(20, int(n * val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_fit, y_fit = X_train[train_idx], y_train[train_idx]
    
    best_predict = None
    best_val_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_fit)
    const_pred_val = np.full(len(y_val), const)
    const_val_rmse = np.sqrt(np.mean((const_pred_val - y_val) ** 2))
    best_val_rmse = const_val_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # Try polynomial degrees
    for deg in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]:
        if _remaining() < 5.0:
            break
        predict, train_rmse = _try_fit_family(lambda Xp, d=deg: _poly_features(Xp, d), X_fit, y_fit)
        if predict is not None and train_rmse < np.inf:
            val_pred = predict(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = predict

    # Try trigonometric
    if _remaining() > 5.0:
        for max_k in [5, 8, 10, 15, 20, 25, 30]:
            if _remaining() < 3.0:
                break
            predict, train_rmse = _try_fit_family(lambda Xp, k=max_k: _trig_features(Xp, k), X_fit, y_fit)
            if predict is not None and train_rmse < np.inf:
                val_pred = predict(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_predict = predict

    # Try exp/log
    if _remaining() > 5.0:
        predict, train_rmse = _try_fit_family(_exp_features, X_fit, y_fit)
        if predict is not None and train_rmse < np.inf:
            val_pred = predict(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = predict

    # Try mixed
    if _remaining() > 5.0:
        predict, train_rmse = _try_fit_family(_mixed_features, X_fit, y_fit)
        if predict is not None and train_rmse < np.inf:
            val_pred = predict(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = predict

    # Try Fourier with different fundamental frequencies
    if _remaining() > 5.0 and d == 1:
        fourier_predict, fourier_rmse = _fit_fourier(X_fit, y_fit, max_harmonics=25)
        if fourier_predict is not None and fourier_rmse < np.inf:
            val_pred = fourier_predict(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = fourier_predict

    # Try smooth splines
    if _remaining() > 5.0 and d == 1:
        ss_predict, ss_rmse = _fit_smooth_spline(X_fit, y_fit)
        if ss_predict is not None and ss_rmse < np.inf:
            val_pred = ss_predict(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = ss_predict

    # Try splines with different knot counts
    if _remaining() > 5.0 and d == 1:
        for nk in [10, 15, 20, 25, 30, 40, 50, 60]:
            if _remaining() < 3.0:
                break
            for meth in ['cubic', 'pchip']:
                sp_predict = _fit_spline_interp(X_fit, y_fit, num_knots=nk, method=meth)
                if sp_predict is not None:
                    val_pred = sp_predict(X_val)
                    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_predict = sp_predict

    # Try Kernel Ridge Regression
    if _remaining() > 10.0:
        for ls in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
            if _remaining() < 3.0:
                break
            kr_predict = _fit_kernel_ridge(X_fit, y_fit, length_scale=ls)
            if kr_predict is not None:
                val_pred = kr_predict(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_predict = kr_predict

    # Try Gaussian Process
    if _remaining() > 15.0:
        gp_predict, gp_rmse = _fit_gaussian_process_optimized(X_fit, y_fit)
        if gp_predict is not None and gp_rmse < np.inf:
            val_pred = gp_predict(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = gp_predict

    # Try RBF with optimization
    if _remaining() > 20.0:
        for nc in [15, 20, 25, 30]:
            if _remaining() < 8.0:
                break
            rbf_predict = _fit_rbf_optimized(X_fit, y_fit, num_centers=nc)
            if rbf_predict is not None:
                val_pred = rbf_predict(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_predict = rbf_predict

    # Try Neural ODE
    if _remaining() > 30.0 and d == 1:
        node_predict, node_rmse = _fit_neural_ode(X_fit, y_fit, hidden_dim=32, epochs=200, num_steps=20)
        if node_predict is not None and node_rmse < np.inf:
            val_pred = node_predict(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = node_predict

    # Try MLP
    if _remaining() > 30.0:
        for hidden in [(64, 64), (128, 64), (128, 128), (256, 128)]:
            if _remaining() < 10.0:
                break
            mlp_predict = _fit_mlp(X_fit, y_fit, hidden_sizes=hidden, epochs=400)
            if mlp_predict is not None:
                val_pred = mlp_predict(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_predict = mlp_predict

    # If we have time, refit the best model on ALL data
    # This is a refinement step: the model family was selected via validation,
    # but we can retrain on all data for better accuracy
    if _remaining() > 15.0:
        # Re-run the top candidates on full data
        candidates = []
        
        # Polynomial
        for deg in [3, 4, 5, 6, 8, 10]:
            if _remaining() < 5.0:
                break
            predict_full, _ = _try_fit_family(lambda Xp, d=deg: _poly_features(Xp, d), X_train, y_train)
            if predict_full is not None:
                candidates.append(predict_full)
        
        # Smooth spline
        if _remaining() > 5.0 and d == 1:
            ss_full, _ = _fit_smooth_spline(X_train, y_train)
            if ss_full is not None:
                candidates.append(ss_full)
        
        # Kernel ridge
        if _remaining() > 5.0:
            for ls in [0.3, 0.5, 1.0, 2.0]:
                if _remaining() < 3.0:
                    break
                kr_full = _fit_kernel_ridge(X_train, y_train, length_scale=ls)
                if kr_full is not None:
                    candidates.append(kr_full)
        
        # GP
        if _remaining() > 10.0:
            gp_full, _ = _fit_gaussian_process_optimized(X_train, y_train)
            if gp_full is not None:
                candidates.append(gp_full)
        
        # Evaluate candidates on validation set
        for cand in candidates:
            val_pred = cand(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = cand

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
