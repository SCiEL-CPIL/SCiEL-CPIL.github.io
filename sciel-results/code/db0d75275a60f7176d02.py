# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize_scalar

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    x = X[:, 0]  # 1D input
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x[0] - 0.5 if n > 0 else 0.0
        x_max = x[0] + 0.5 if n > 0 else 1.0
    
    # Scale x to [0,1]
    x_scaled = (x - x_min) / x_range
    
    candidates = []  # list of (name, predict_func, cv_mse)
    
    def elapsed():
        return time.time() - start_time
    
    def cv_mse(pred_func, n_folds=5):
        """Compute cross-validation MSE for model selection"""
        if n < 10:
            # Too few points, just use training MSE
            try:
                yp = np.asarray(pred_func(X), dtype=float).flatten()
                if yp.shape[0] != n or not np.all(np.isfinite(yp)):
                    return np.inf
                return float(np.mean((yp - y) ** 2))
            except Exception:
                return np.inf
        
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        total_se = 0.0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            try:
                yp = np.asarray(pred_func(X[val_idx]), dtype=float).flatten()
                if yp.shape[0] != len(val_idx) or not np.all(np.isfinite(yp)):
                    return np.inf
                se = float(np.sum((yp - y[val_idx]) ** 2))
                if not np.isfinite(se):
                    return np.inf
                total_se += se
            except Exception:
                return np.inf
        return total_se / n
    
    def add_candidate(name, pred_func):
        if elapsed() > deadline:
            return
        try:
            yp = np.asarray(pred_func(X), dtype=float).flatten()
            if yp.shape[0] != n or not np.all(np.isfinite(yp)):
                return
            cv = cv_mse(pred_func)
            if np.isfinite(cv):
                candidates.append((name, pred_func, cv))
        except Exception:
            pass
    
    # --- Strategy 1: Polynomial regression (varying degrees) on scaled x ---
    for deg in range(1, 15):
        if elapsed() > deadline:
            break
        V = np.vander(x_scaled, N=deg + 1, increasing=True)
        try:
            w, *_ = np.linalg.lstsq(V, y, rcond=None)
            def pred_poly(Xp, deg=deg, w=w, x_min=x_min, x_range=x_range):
                Xp = np.asarray(Xp, dtype=float).reshape(-1)
                Xs = (Xp - x_min) / x_range
                Vp = np.vander(Xs, N=deg + 1, increasing=True)
                return Vp @ w
            add_candidate(f"poly_{deg}", pred_poly)
        except Exception:
            pass
    
    # --- Strategy 2: Fourier basis with adaptive frequencies ---
    max_nf = 8
    # Generate more diverse frequency candidates
    omegas = []
    for base in [0.25, 0.5, 1.0, 2.0, 4.0, np.pi, 2 * np.pi]:
        for mult in [0.5, 1.0, 2.0]:
            omegas.append(base * mult)
    # Add frequencies based on data range
    for div in [2, 3, 4, 5, 6, 8, 10, 12]:
        omegas.append(np.pi / (x_range / div))
        omegas.append(2 * np.pi / (x_range / div))
    omegas = list(set(omegas))
    omegas = [om for om in omegas if 0.01 < om < 50]
    
    for nf in range(1, max_nf + 1):
        if elapsed() > deadline:
            break
        for omega in omegas[:8]:  # Limit number of omegas for speed
            if elapsed() > deadline:
                break
            cols = [np.ones(n)]
            for k in range(1, nf + 1):
                cols.append(np.sin(k * omega * x))
                cols.append(np.cos(k * omega * x))
            V = np.column_stack(cols)
            try:
                w, *_ = np.linalg.lstsq(V, y, rcond=None)
                def pred_fourier(Xp, nf=nf, omega=omega, w=w):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    cols = [np.ones(len(Xp))]
                    for k in range(1, nf + 1):
                        cols.append(np.sin(k * omega * Xp))
                        cols.append(np.cos(k * omega * Xp))
                    Vp = np.column_stack(cols)
                    return Vp @ w
                add_candidate(f"four_nf{nf}_om{omega:.4f}", pred_fourier)
            except Exception:
                pass
    
    # --- Strategy 3: Rich basis with greedy selection ---
    basis_funcs = {}
    basis_funcs['1'] = lambda x: np.ones_like(x)
    basis_funcs['x'] = lambda x: x
    basis_funcs['x2'] = lambda x: x ** 2
    basis_funcs['x3'] = lambda x: x ** 3
    basis_funcs['x4'] = lambda x: x ** 4
    basis_funcs['x5'] = lambda x: x ** 5
    basis_funcs['x6'] = lambda x: x ** 6
    basis_funcs['abs_x'] = lambda x: np.abs(x)
    basis_funcs['sqrt_abs'] = lambda x: np.sqrt(np.abs(x))
    basis_funcs['log1p_abs'] = lambda x: np.log1p(np.abs(x))
    basis_funcs['tanh'] = lambda x: np.tanh(x)
    basis_funcs['tanh_2x'] = lambda x: np.tanh(2 * x)
    basis_funcs['tanh_05x'] = lambda x: np.tanh(0.5 * x)
    basis_funcs['exp_05x'] = lambda x: np.exp(0.5 * x)
    basis_funcs['exp_neg05x'] = lambda x: np.exp(-0.5 * x)
    basis_funcs['exp_x'] = lambda x: np.exp(x)
    basis_funcs['exp_negx'] = lambda x: np.exp(-x)
    basis_funcs['exp_01x'] = lambda x: np.exp(0.1 * x)
    basis_funcs['exp_neg01x'] = lambda x: np.exp(-0.1 * x)
    basis_funcs['x_over_1x2'] = lambda x: x / (1 + x ** 2)
    basis_funcs['x_over_1abx'] = lambda x: x / (1 + np.abs(x))
    basis_funcs['sinc'] = lambda x: np.sinc(x)
    basis_funcs['sin_x'] = lambda x: np.sin(x)
    basis_funcs['cos_x'] = lambda x: np.cos(x)
    basis_funcs['sin_2x'] = lambda x: np.sin(2 * x)
    basis_funcs['cos_2x'] = lambda x: np.cos(2 * x)
    basis_funcs['sin_05x'] = lambda x: np.sin(0.5 * x)
    basis_funcs['cos_05x'] = lambda x: np.cos(0.5 * x)
    basis_funcs['x_exp_negx2'] = lambda x: x * np.exp(-x ** 2)
    basis_funcs['exp_negx2'] = lambda x: np.exp(-x ** 2)
    basis_funcs['x2_exp_negx2'] = lambda x: x ** 2 * np.exp(-x ** 2)
    
    # Evaluate on training data
    basis_names = list(basis_funcs.keys())
    B = np.zeros((n, len(basis_names)))
    for j, name in enumerate(basis_names):
        try:
            B[:, j] = basis_funcs[name](x)
        except Exception:
            pass
    B = np.where(np.isfinite(B), B, 0.0)
    
    if elapsed() < deadline:
        # Greedy forward selection
        selected = []
        remaining = list(range(len(basis_names)))
        for step in range(min(10, len(basis_names))):
            if elapsed() > deadline:
                break
            best_j = None
            best_rss = np.inf
            for j in remaining:
                try:
                    if selected:
                        V = np.column_stack([B[:, i] for i in selected] + [B[:, j]])
                    else:
                        V = B[:, j:j+1]
                    w, *_ = np.linalg.lstsq(V, y, rcond=None)
                    rss = float(np.sum((V @ w - y) ** 2))
                    if np.isfinite(rss) and rss < best_rss:
                        best_rss = rss
                        best_j = j
                except Exception:
                    pass
            if best_j is None:
                break
            selected.append(best_j)
            remaining.remove(best_j)
            
            sel_names = [basis_names[i] for i in selected]
            try:
                V = np.column_stack([B[:, i] for i in selected])
                w, *_ = np.linalg.lstsq(V, y, rcond=None)
                def pred_sel(Xp, sel_names=sel_names, w=w):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    cols = []
                    for nm in sel_names:
                        c = basis_funcs[nm](Xp)
                        c = np.where(np.isfinite(c), c, 0.0)
                        cols.append(c)
                    Vp = np.column_stack(cols)
                    return Vp @ w
                add_candidate(f"sel_{'_'.join(sel_names)}", pred_sel)
            except Exception:
                pass
    
    # --- Strategy 4: Gaussian Process with Matern 5/2 ---
    if elapsed() < deadline:
        def matern52_kernel(X1, X2, ls):
            X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
            X2 = np.asarray(X2, dtype=float).reshape(-1, 1)
            D = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
            sqrt3d = np.sqrt(3.0) * D / ls
            return (1 + sqrt3d + sqrt3d ** 2 / 3.0) * np.exp(-sqrt3d)
        
        y_var = float(np.var(y))
        if y_var < 1e-12:
            y_var = 1.0
        
        for ls in [x_range / 3.0, x_range / 5.0, x_range / 8.0, x_range / 15.0, x_range / 20.0, 0.2, 0.3, 0.5, 1.0, 2.0]:
            if ls <= 0 or elapsed() > deadline:
                continue
            try:
                K_kernel = matern52_kernel(x, x, ls)
                K = y_var * K_kernel + 1e-8 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                
                def pred_gp(Xp, x=x, L=L, alpha=alpha, ls=ls, y_var=y_var, matern52_kernel=matern52_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Ks = matern52_kernel(Xp, x, ls)
                    return y_var * (Ks @ alpha)
                
                add_candidate(f"gp_ls{ls:.4f}", pred_gp)
            except Exception:
                pass
    
    # --- Strategy 5: RBF kernel regression (fixed width) ---
    if elapsed() < deadline:
        def rbf_kernel(X1, X2, width):
            X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
            X2 = np.asarray(X2, dtype=float).reshape(-1, 1)
            D2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
            return np.exp(-D2 / (2 * width ** 2))
        
        for width in [x_range / 4.0, x_range / 6.0, x_range / 10.0, x_range / 20.0, x_range / 30.0, 0.1, 0.2, 0.5, 1.0]:
            if width <= 0 or elapsed() > deadline:
                continue
            try:
                K = rbf_kernel(x, x, width)
                K += 1e-8 * np.eye(n)
                w, *_ = np.linalg.lstsq(K, y, rcond=None)
                
                def pred_rbf(Xp, x=x, w=w, width=width, rbf_kernel=rbf_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Ks = rbf_kernel(Xp, x, width)
                    return Ks @ w
                
                add_candidate(f"rbf_w{width:.4f}", pred_rbf)
            except Exception:
                pass
    
    # --- Strategy 6: Cubic Spline Interpolation ---
    if elapsed() < deadline:
        try:
            # Sort by x for interpolation
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            cs = CubicSpline(x_sorted, y_sorted)
            def pred_spline(Xp):
                Xp = np.asarray(Xp, dtype=float).reshape(-1)
                return cs(Xp)
            add_candidate("spline_cubic", pred_spline)
        except Exception:
            pass
    
    # --- Strategy 7: Linear interpolation ---
    if elapsed() < deadline:
        try:
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
            def pred_interp(Xp):
                Xp = np.asarray(Xp, dtype=float).reshape(-1)
                return interp_func(Xp)
            add_candidate("interp_linear", pred_interp)
        except Exception:
            pass
    
    # Select best candidate using CV MSE
    if not candidates:
        # Fallback: linear regression
        A = np.column_stack([np.ones(n), x])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def pred_lin(Xp, w=w):
            Xp = np.asarray(Xp, dtype=float).reshape(-1)
            return np.column_stack([np.ones(len(Xp)), Xp]) @ w
        return pred_lin
    
    candidates.sort(key=lambda t: t[2])
    best_name, best_pred, best_cv = candidates[0]
    
    def final_predict(Xp):
        out = best_pred(Xp)
        out = np.asarray(out, dtype=float).flatten()
        out = np.where(np.isfinite(out), out, 0.0)
        return out
    
    return final_predict
# EVOLVE-BLOCK-END
