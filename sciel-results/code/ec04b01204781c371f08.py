# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import make_interp_spline, UnivariateSpline
from scipy.integrate import solve_ivp

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


def _nrmse(pred, y):
    var = np.var(y)
    if var < 1e-12:
        return 0.0 if np.allclose(pred, y) else 1e10
    return float(np.mean((pred - y)**2) / var)


def _poly_features(X, degree):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                for p in range(1, degree + 1):
                    for q in range(1, degree + 1 - p + 1):
                        if p + q <= degree:
                            cols.append((X[:, i] ** p) * (X[:, j] ** q))
    return np.column_stack(cols)


def _trig_features(X, max_k=6):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, max_k + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(x * np.sin(x))
        cols.append(x * np.cos(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        safe_x = np.clip(x, -50, 50)
        cols.append(np.exp(safe_x))
        cols.append(np.exp(-safe_x))
        cols.append(np.sqrt(np.abs(x) + 1e-12))
        cols.append(np.log(np.abs(x) + 1e-12))
        cols.append(x * np.exp(-safe_x))
        cols.append(x * np.exp(safe_x))
        cols.append(np.sinh(safe_x))
        cols.append(np.cosh(safe_x))
        cols.append(1.0 / (x + 1e-12))
        cols.append(x / (x**2 + 1e-12))
    return np.column_stack(cols)


def _gaussian_rbf_features(X, centers, widths):
    n = X.shape[0]
    cols = [np.ones(n)]
    for c, w in zip(centers, widths):
        dist = np.sum((X - c)**2, axis=1)
        cols.append(np.exp(-dist / (2 * w**2)))
    return np.column_stack(cols)


def _mixed_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(x)
        cols.append(x**2)
        cols.append(x**3)
        cols.append(x**4)
        cols.append(x**5)
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2*x))
        cols.append(np.cos(2*x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.exp(-x**2))
        safe = np.clip(x, -50, 50)
        cols.append(np.exp(safe))
        cols.append(np.exp(-safe))
        cols.append(np.log(np.abs(x) + 1e-12))
        cols.append(np.sqrt(np.abs(x) + 1e-12))
        cols.append(x * np.exp(-safe))
    return np.column_stack(cols)


def _fit_ols(F, y, ridge_alpha=1e-8):
    try:
        n, p = F.shape
        if p == 0:
            return np.array([np.mean(y)])
        F_reg = F.copy()
        if p > 1:
            F_reg[:, 1:] = F_reg[:, 1:] * np.sqrt(n)
            F_reg[:, 0] = F_reg[:, 0] * np.sqrt(n)
            F_reg[:, 1:] += ridge_alpha * np.eye(p)[1:] * np.sqrt(n)
        else:
            F_reg[:, 0] = F_reg[:, 0] * np.sqrt(n)
            F_reg[:, 0] += ridge_alpha * np.sqrt(n)
        w, *_ = np.linalg.lstsq(F_reg, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _fit_ols_ridge(F, y, alpha):
    try:
        n, p = F.shape
        if p == 0:
            return np.array([np.mean(y)])
        A = F.T @ F + alpha * np.eye(p)
        A[0, 0] -= alpha
        b = F.T @ y
        w = np.linalg.solve(A, b)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _fit_spline(X, y, num_knots=20):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 8:
            return None
        if num_knots > n - 4:
            num_knots = n - 4
        unique_mask = np.concatenate(([True], np.diff(xs) > 1e-12))
        xs_u, ys_u = xs[unique_mask], ys[unique_mask]
        if len(xs_u) < 8:
            return None
        knot_idx = np.linspace(0, len(xs_u) - 1, num_knots, dtype=int)
        knot_x = xs_u[knot_idx]
        knot_y = ys_u[knot_idx]
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


def _fit_univariate_spline(X, y, s=None):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 8:
            return None
        if s is None:
            s = 0.01 * np.var(ys) * n
        sp = UnivariateSpline(xs, ys, s=s, k=3)
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


def _fit_mlp(X, y, hidden_sizes=(64, 32), lr=0.01, epochs=800, seed=42):
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


def _fit_gaussian_process(X, y, noise=1e-4):
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_std = y.std() + 1e-8
        yn = (y - y_mu) / y_std

        def kernel(X1, X2, ls):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (ls**2)
            return np.exp(-0.5 * sq_dist)

        def objective(params):
            ls = params[0]
            if ls <= 0:
                return 1e6
            K = kernel(Xn, Xn, ls) + noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
                lml = -0.5 * yn @ alpha - np.sum(np.log(np.diag(L)))
                return -float(lml)
            except Exception:
                return 1e6

        res = minimize(objective, [1.0], method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4})
        best_ls = max(0.01, float(res.x[0]))

        K = kernel(Xn, Xn, best_ls) + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        except Exception:
            return None

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn, best_ls)
            preds = Ks @ alpha
            return preds.flatten() * y_std + y_mu
        return predict
    except Exception:
        return None


def _fit_neural_ode(X, y, num_nodes=100, lr=0.01, epochs=300, seed=42):
    """
    Neural ODE approach: learn a function f(x) such that dy/dx = f(x),
    with f parameterized by a small neural network.
    This is a structural change from basis expansions.
    """
    try:
        if X.shape[1] != 1:
            return None
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 10:
            return None

        x_min, x_max = xs[0], xs[-1]
        x_range = x_max - x_min
        if x_range < 1e-8:
            return None

        # Normalize x to [0, 1]
        xn = (xs - x_min) / x_range
        y_mu = ys.mean()
        y_sigma = ys.std() + 1e-8
        yn = (ys - y_mu) / y_sigma

        # Sample points for ODE integration
        t_eval = np.linspace(0, 1, num_nodes)
        x_eval = x_min + t_eval * x_range

        # Neural network for f(x): tanh activation, 2 hidden layers
        np.random.seed(seed)
        h1, h2 = 16, 16
        W1 = np.random.randn(1, h1) * 0.5
        b1 = np.zeros(h1)
        W2 = np.random.randn(h1, h2) * 0.5
        b2 = np.zeros(h2)
        W3 = np.random.randn(h2, 1) * 0.5
        b3 = np.zeros(1)

        def ode_func(t, state):
            # state = [y]
            x_val = x_min + t * x_range
            xv = np.array([x_val])
            a1 = np.tanh(xv @ W1 + b1)
            a2 = np.tanh(a1 @ W2 + b2)
            a3 = a2 @ W3 + b3
            return a3.flatten()

        params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2, W3.flatten(), b3])
        n1, nb1, n2, nb2, n3, nb3 = W1.size, b1.size, W2.size, b2.size, W3.size, b3.size

        def reconstruct(params):
            W1_r = params[0:n1].reshape(W1.shape)
            b1_r = params[n1:n1+nb1]
            W2_r = params[n1+nb1:n1+nb1+n2].reshape(W2.shape)
            b2_r = params[n1+nb1+n2:n1+nb1+n2+nb2]
            W3_r = params[n1+nb1+n2+nb2:n1+nb1+n2+nb2+n3].reshape(W3.shape)
            b3_r = params[n1+nb1+n2+nb2+n3:]
            return W1_r, b1_r, W2_r, b2_r, W3_r, b3_r

        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t_adam = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def compute_loss(params):
            W1_r, b1_r, W2_r, b2_r, W3_r, b3_r = reconstruct(params)
            def ode_func_local(t, state):
                x_val = x_min + t * x_range
                xv = np.array([x_val])
                a1 = np.tanh(xv @ W1_r + b1_r)
                a2 = np.tanh(a1 @ W2_r + b2_r)
                a3 = a2 @ W3_r + b3_r
                return a3.flatten()

            sol = solve_ivp(ode_func_local, [0, 1], [0.0], t_eval=t_eval, max_step=0.01)
            if not sol.success:
                return 1e10
            y_pred = sol.y[0]
            # Interpolate to match training points
            y_at_train = np.interp(xn, t_eval, y_pred)
            loss = np.mean((y_at_train - yn)**2)
            return float(loss)

        # Simple gradient-free optimization with random restarts
        best_params = params.copy()
        best_loss = compute_loss(params)
        step_size = 0.1

        for epoch in range(epochs):
            if _remaining() < 5.0:
                break
            # Coordinate descent style
            for i in range(len(params)):
                if _remaining() < 2.0:
                    break
                for direction in [-1, 1]:
                    trial = params.copy()
                    trial[i] += direction * step_size
                    loss = compute_loss(trial)
                    if loss < best_loss:
                        params = trial
                        best_loss = loss
                        best_params = trial.copy()
            step_size *= 0.95

        # Final prediction
        W1_r, b1_r, W2_r, b2_r, W3_r, b3_r = reconstruct(best_params)
        def ode_func_final(t, state):
            x_val = x_min + t * x_range
            xv = np.array([x_val])
            a1 = np.tanh(xv @ W1_r + b1_r)
            a2 = np.tanh(a1 @ W2_r + b2_r)
            a3 = a2 @ W3_r + b3_r
            return a3.flatten()

        # Pre-compute solution on a fine grid for fast prediction
        t_fine = np.linspace(0, 1, 1000)
        sol_fine = solve_ivp(ode_func_final, [0, 1], [0.0], t_eval=t_fine, max_step=0.005)
        if not sol_fine.success:
            return None
        t_grid = t_fine
        y_grid = sol_fine.y[0]

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xn_p = (xp - x_min) / x_range
            yn_p = np.interp(xn_p, t_grid, y_grid)
            return yn_p * y_sigma + y_mu
        return predict
    except Exception:
        return None


def _fit_rbf(X, y, num_centers=20):
    """RBF regression with optimized centers and widths."""
    try:
        n, d = X.shape
        if d != 1:
            return None
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 10:
            return None

        # Choose centers evenly spaced
        center_idx = np.linspace(0, n-1, num_centers, dtype=int)
        centers = xs[center_idx]
        # Initial widths based on spacing
        if num_centers > 1:
            widths = np.full(num_centers, (xs[-1] - xs[0]) / num_centers)
        else:
            widths = np.array([(xs[-1] - xs[0]) / 2])

        # Build RBF features
        F = _gaussian_rbf_features(X, centers.reshape(-1, 1), widths)
        w = _fit_ols_ridge(F, y, 1e-6)
        if w is None:
            return None

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Fp = _gaussian_rbf_features(Xp, centers.reshape(-1, 1), widths)
            return Fp @ w
        return predict
    except Exception:
        return None


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    # Hold-out validation split
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    split_idx = int(n * 0.8)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    best_val_nrmse = np.inf
    best_model_type = None
    best_model_params = None

    # 1. Polynomial Fits
    for deg in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
        if _remaining() < 5.0:
            break
        try:
            F_tr = _poly_features(X_tr, deg)
            w = _fit_ols(F_tr, y_tr)
            if w is None:
                continue
            pred_val = _poly_features(X_val, deg) @ w
            nrmse = _nrmse(pred_val, y_val)
            if nrmse < best_val_nrmse:
                best_val_nrmse = nrmse
                best_model_type = 'poly'
                best_model_params = deg
        except Exception:
            pass

    # 2. Trig Fits
    if _remaining() > 5.0:
        for max_k in [3, 5, 6, 8]:
            try:
                F_tr = _trig_features(X_tr, max_k)
                w = _fit_ols(F_tr, y_tr)
                if w is None:
                    continue
                pred_val = _trig_features(X_val, max_k) @ w
                nrmse = _nrmse(pred_val, y_val)
                if nrmse < best_val_nrmse:
                    best_val_nrmse = nrmse
                    best_model_type = 'trig'
                    best_model_params = max_k
            except Exception:
                pass

    # 3. Exp Fits
    if _remaining() > 5.0:
        try:
            F_tr = _exp_features(X_tr)
            w = _fit_ols(F_tr, y_tr)
            if w is not None:
                pred_val = _exp_features(X_val) @ w
                nrmse = _nrmse(pred_val, y_val)
                if nrmse < best_val_nrmse:
                    best_val_nrmse = nrmse
                    best_model_type = 'exp'
                    best_model_params = None
        except Exception:
            pass

    # 4. Mixed Fits
    if _remaining() > 5.0:
        try:
            F_tr = _mixed_features(X_tr)
            w = _fit_ols(F_tr, y_tr)
            if w is not None:
                pred_val = _mixed_features(X_val) @ w
                nrmse = _nrmse(pred_val, y_val)
                if nrmse < best_val_nrmse:
                    best_val_nrmse = nrmse
                    best_model_type = 'mixed'
                    best_model_params = None
        except Exception:
            pass

    # 5. Spline (interpolating)
    if _remaining() > 5.0 and d == 1:
        for nk in [10, 15, 20, 30]:
            try:
                sp_pred = _fit_spline(X_tr, y_tr, nk)
                if sp_pred is not None:
                    pred_val = sp_pred(X_val)
                    nrmse = _nrmse(pred_val, y_val)
                    if nrmse < best_val_nrmse:
                        best_val_nrmse = nrmse
                        best_model_type = 'spline'
                        best_model_params = nk
            except Exception:
                pass

    # 6. Univariate Spline (smoothed)
    if _remaining() > 5.0 and d == 1:
        for s_factor in [0.001, 0.01, 0.1, 1.0, 10.0]:
            try:
                s_val = s_factor * np.var(y_tr) * len(y_tr)
                sp_pred = _fit_univariate_spline(X_tr, y_tr, s_val)
                if sp_pred is not None:
                    pred_val = sp_pred(X_val)
                    nrmse = _nrmse(pred_val, y_val)
                    if nrmse < best_val_nrmse:
                        best_val_nrmse = nrmse
                        best_model_type = 'uni_spline'
                        best_model_params = s_val
            except Exception:
                pass

    # 7. RBF
    if _remaining() > 5.0 and d == 1:
        for nc in [10, 15, 20, 30]:
            try:
                rbf_pred = _fit_rbf(X_tr, y_tr, nc)
                if rbf_pred is not None:
                    pred_val = rbf_pred(X_val)
                    nrmse = _nrmse(pred_val, y_val)
                    if nrmse < best_val_nrmse:
                        best_val_nrmse = nrmse
                        best_model_type = 'rbf'
                        best_model_params = nc
            except Exception:
                pass

    # 8. Gaussian Process
    if _remaining() > 15.0:
        try:
            gp_pred = _fit_gaussian_process(X_tr, y_tr)
            if gp_pred is not None:
                pred_val = gp_pred(X_val)
                nrmse = _nrmse(pred_val, y_val)
                if nrmse < best_val_nrmse:
                    best_val_nrmse = nrmse
                    best_model_type = 'gp'
                    best_model_params = None
        except Exception:
            pass

    # 9. Neural ODE (structural change)
    if _remaining() > 30.0 and d == 1:
        try:
            no_pred = _fit_neural_ode(X_tr, y_tr)
            if no_pred is not None:
                pred_val = no_pred(X_val)
                nrmse = _nrmse(pred_val, y_val)
                if nrmse < best_val_nrmse:
                    best_val_nrmse = nrmse
                    best_model_type = 'neural_ode'
                    best_model_params = None
        except Exception:
            pass

    # 10. MLP
    if _remaining() > 20.0:
        for hs in [(32, 32), (64, 32), (64, 64)]:
            try:
                mlp_pred = _fit_mlp(X_tr, y_tr, hidden_sizes=hs)
                if mlp_pred is not None:
                    pred_val = mlp_pred(X_val)
                    nrmse = _nrmse(pred_val, y_val)
                    if nrmse < best_val_nrmse:
                        best_val_nrmse = nrmse
                        best_model_type = 'mlp'
                        best_model_params = hs
            except Exception:
                pass

    # Fallback to constant
    if best_model_type is None:
        best_model_type = 'const'
        best_model_params = np.mean(y_train)

    # Refit on FULL data with the best model type
    final_predict = None

    if best_model_type == 'poly':
        deg = best_model_params
        F_full = _poly_features(X_train, deg)
        w = _fit_ols(F_full, y_train)
        if w is not None:
            final_predict = _safe_predict(lambda Xp: _poly_features(Xp, deg) @ w)

    elif best_model_type == 'trig':
        max_k = best_model_params
        F_full = _trig_features(X_train, max_k)
        w = _fit_ols(F_full, y_train)
        if w is not None:
            final_predict = _safe_predict(lambda Xp: _trig_features(Xp, max_k) @ w)

    elif best_model_type == 'exp':
        F_full = _exp_features(X_train)
        w = _fit_ols(F_full, y_train)
        if w is not None:
            final_predict = _safe_predict(lambda Xp: _exp_features(Xp) @ w)

    elif best_model_type == 'mixed':
        F_full = _mixed_features(X_train)
        w = _fit_ols(F_full, y_train)
        if w is not None:
            final_predict = _safe_predict(lambda Xp: _mixed_features(Xp) @ w)

    elif best_model_type == 'spline':
        nk = best_model_params
        final_predict = _fit_spline(X_train, y_train, nk)

    elif best_model_type == 'uni_spline':
        s_val = best_model_params
        final_predict = _fit_univariate_spline(X_train, y_train, s_val)

    elif best_model_type == 'rbf':
        nc = best_model_params
        final_predict = _fit_rbf(X_train, y_train, nc)

    elif best_model_type == 'gp':
        final_predict = _fit_gaussian_process(X_train, y_train)

    elif best_model_type == 'neural_ode':
        final_predict = _fit_neural_ode(X_train, y_train)

    elif best_model_type == 'mlp':
        hs = best_model_params
        final_predict = _fit_mlp(X_train, y_train, hidden_sizes=hs)

    elif best_model_type == 'const':
        const_val = best_model_params
        final_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const_val))

    # Final safety check
    if final_predict is None:
        const_val = np.mean(y_train)
        final_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const_val))

    return final_predict
# EVOLVE-BLOCK-END
