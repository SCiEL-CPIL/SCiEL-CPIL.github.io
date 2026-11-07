# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline

_TOTAL_DEADLINE = time.time() + 315.0  # safety margin under 330s


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


def _poly_features(X, degree):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    return np.column_stack(cols)


def _trig_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, 8):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        # Avoid division by zero issues in features if x is close to 0
        cols.append(1.0 / (x + 1e-9))
        cols.append(x * np.exp(-x))
        cols.append(x * np.exp(x))
    return np.column_stack(cols)


def _mixed_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(x)
        cols.append(x ** 2)
        cols.append(x ** 3)
        cols.append(x ** 4)
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2 * x))
        cols.append(np.cos(2 * x))
        cols.append(np.tanh(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.log(np.abs(x) + 1e-9))
    return np.column_stack(cols)


def _fit_ols(F, y, ridge_alpha=1e-6):
    try:
        n, p = F.shape
        F_reg = F.copy()
        # Regularize non-intercept columns more strongly
        if p > 1:
            F_reg[:, 1:] += ridge_alpha * np.sqrt(n)
        else:
            F_reg[:, 0] += ridge_alpha * np.sqrt(n)
        w, *_ = np.linalg.lstsq(F_reg, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _fit_spline(X, y, num_knots=15):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < num_knots + 4:
            num_knots = max(4, n - 4)
        
        # Ensure unique knots for spline
        unique_mask = np.concatenate(([True], np.diff(xs) > 1e-12))
        xs_unique = xs[unique_mask]
        ys_unique = ys[unique_mask]
        
        if len(xs_unique) < 4:
            return None

        knot_idx = np.linspace(0, len(xs_unique) - 1, num_knots, dtype=int)
        knot_x = xs_unique[knot_idx]
        knot_y = ys_unique[knot_idx]
        
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


def _fit_mlp(X, y, hidden_sizes=(32, 32), lr=0.005, epochs=500, seed=42):
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
    """Simple Gaussian Process regression with optimized length scale."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        # Normalize y for numerical stability in GP
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
        
        # Optimize length scale
        # Initial guess: 1.0 (normalized space)
        res = minimize(objective, [1.0], method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-3})
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


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    # Hold-out validation split
    # Shuffle indices
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    split_idx = int(n * 0.8)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    best_val_rmse = np.inf
    best_model_type = None
    best_model_params = None

    # 1. Try Polynomial Fits
    for deg in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        if _remaining() < 5.0:
            break
        try:
            F_tr = _poly_features(X_tr, deg)
            w = _fit_ols(F_tr, y_tr)
            if w is None: continue
            pred_val = _poly_features(X_val, deg) @ w
            rmse = np.sqrt(np.mean((pred_val - y_val)**2))
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_model_type = 'poly'
                best_model_params = deg
        except Exception:
            pass

    # 2. Try Trig Fits
    if _remaining() > 5.0:
        try:
            F_tr = _trig_features(X_tr)
            w = _fit_ols(F_tr, y_tr)
            if w is not None:
                pred_val = _trig_features(X_val) @ w
                rmse = np.sqrt(np.mean((pred_val - y_val)**2))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_model_type = 'trig'
                    best_model_params = None
        except Exception:
            pass

    # 3. Try Exp Fits
    if _remaining() > 5.0:
        try:
            F_tr = _exp_features(X_tr)
            w = _fit_ols(F_tr, y_tr)
            if w is not None:
                pred_val = _exp_features(X_val) @ w
                rmse = np.sqrt(np.mean((pred_val - y_val)**2))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_model_type = 'exp'
                    best_model_params = None
        except Exception:
            pass

    # 4. Try Mixed Fits
    if _remaining() > 5.0:
        try:
            F_tr = _mixed_features(X_tr)
            w = _fit_ols(F_tr, y_tr)
            if w is not None:
                pred_val = _mixed_features(X_val) @ w
                rmse = np.sqrt(np.mean((pred_val - y_val)**2))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_model_type = 'mixed'
                    best_model_params = None
        except Exception:
            pass

    # 5. Try Spline
    if _remaining() > 5.0 and d == 1:
        try:
            sp_pred = _fit_spline(X_tr, y_tr)
            if sp_pred is not None:
                pred_val = sp_pred(X_val)
                rmse = np.sqrt(np.mean((pred_val - y_val)**2))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_model_type = 'spline'
                    best_model_params = None
        except Exception:
            pass

    # 6. Try Gaussian Process
    if _remaining() > 10.0:
        try:
            gp_pred = _fit_gaussian_process(X_tr, y_tr)
            if gp_pred is not None:
                pred_val = gp_pred(X_val)
                rmse = np.sqrt(np.mean((pred_val - y_val)**2))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_model_type = 'gp'
                    best_model_params = None
        except Exception:
            pass

    # 7. Try MLP
    if _remaining() > 15.0:
        try:
            mlp_pred = _fit_mlp(X_tr, y_tr)
            if mlp_pred is not None:
                pred_val = mlp_pred(X_val)
                rmse = np.sqrt(np.mean((pred_val - y_val)**2))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_model_type = 'mlp'
                    best_model_params = None
        except Exception:
            pass

    # Fallback to constant if nothing worked well
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
        F_full = _trig_features(X_train)
        w = _fit_ols(F_full, y_train)
        if w is not None:
            final_predict = _safe_predict(lambda Xp: _trig_features(Xp) @ w)
            
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
        final_predict = _fit_spline(X_train, y_train)
        
    elif best_model_type == 'gp':
        final_predict = _fit_gaussian_process(X_train, y_train)
        
    elif best_model_type == 'mlp':
        final_predict = _fit_mlp(X_train, y_train)
        
    elif best_model_type == 'const':
        const_val = best_model_params
        final_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const_val))

    # Final safety check
    if final_predict is None:
        const_val = np.mean(y_train)
        final_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const_val))

    return final_predict
# EVOLVE-BLOCK-END
