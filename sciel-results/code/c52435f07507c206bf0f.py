"""Rewritten program for symbolic regression over 4 hidden 1D equations."""
import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator, Rbf
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # safety margin before 330s wall-clock
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Handle constant y
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-14:
        c = y_mean
        return lambda X: np.full(len(np.asarray(X)), c)
    
    if d != 1:
        # Multi-dimensional fallback: RBF interpolation
        return _fit_rbf(X, y, start_time, DEADLINE)
    
    x = X[:, 0].copy()
    y = y.copy()
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates
    unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) < n:
        y_sum = np.zeros(len(unique_x))
        np.add.at(y_sum, inverse, y_sorted)
        y_unique = y_sum / counts
    else:
        y_unique = y_sorted
        unique_x = x_sorted
    
    x_u = unique_x
    y_u = y_unique
    n_u = len(x_u)
    
    # Strategy 1: Cubic Spline with not-a-knot
    if n_u >= 4:
        try:
            cs = CubicSpline(x_u, y_u, bc_type='not-a-knot')
            y_pred = cs(x_u)
            sse = np.sum((y_u - y_pred)**2)
            sst = np.sum((y_u - np.mean(y_u))**2)
            r2 = 1 - sse / sst if sst > 0 else 1.0
            if r2 > 0.9999:
                def predict_cs(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    xp = X_pred[:, 0]
                    yp = cs(xp)
                    return np.nan_to_num(yp, nan=y_mean, posinf=np.max(y_u), neginf=np.min(y_u))
                return predict_cs
        except Exception:
            pass
    
    # Strategy 2: PCHIP (monotone cubic, no oscillation)
    if n_u >= 3:
        try:
            pchip = PchipInterpolator(x_u, y_u)
            y_pred = pchip(x_u)
            sse = np.sum((y_u - y_pred)**2)
            sst = np.sum((y_u - np.mean(y_u))**2)
            r2 = 1 - sse / sst if sst > 0 else 1.0
            if r2 > 0.9999:
                def predict_pchip(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    xp = X_pred[:, 0]
                    yp = pchip(xp)
                    return np.nan_to_num(yp, nan=y_mean, posinf=np.max(y_u), neginf=np.min(y_u))
                return predict_pchip
        except Exception:
            pass
    
    # Strategy 3: Gaussian Process with Matern 5/2 kernel (exact interpolation for noiseless data)
    if n_u >= 4 and time_left() > 30:
        try:
            gp_result = _fit_gp_matern(x_u, y_u, start_time, DEADLINE)
            if gp_result is not None:
                return gp_result
        except Exception:
            pass
    
    # Strategy 4: RBF interpolation with multiple kernels
    if n_u >= 4 and time_left() > 20:
        try:
            rbf_result = _fit_rbf_1d(x_u, y_u, start_time, DEADLINE)
            if rbf_result is not None:
                return rbf_result
        except Exception:
            pass
    
    # Strategy 5: Adaptive piecewise polynomial
    if n_u >= 8 and time_left() > 20:
        try:
            pw_result = _fit_piecewise_poly(x_u, y_u, start_time, DEADLINE)
            if pw_result is not None:
                return pw_result
        except Exception:
            pass
    
    # Strategy 6: Basis function regression
    if time_left() > 15:
        try:
            basis_result = _fit_basis(x_u, y_u, start_time, DEADLINE)
            if basis_result is not None:
                return basis_result
        except Exception:
            pass
    
    # Fallback: linear interpolation
    def predict_linear(X_pred):
        X_pred = np.asarray(X_pred, dtype=np.float64)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(-1, 1)
        xp = X_pred[:, 0]
        yp = np.interp(xp, x_u, y_u)
        return yp
    
    return predict_linear


def _fit_gp_matern(x, y, start_time, DEADLINE):
    """Gaussian Process with Matern 5/2 kernel, exact interpolation (zero noise)."""
    n = len(x)
    if n < 4 or n > 500:
        return None
    
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        return None
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        return None
    
    # Normalize
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    
    # Precompute pairwise distances
    D = np.abs(x_norm[:, None] - x_norm[None, :])
    
    def matern52(K, l):
        """Matern 5/2 kernel."""
        z = K / l
        sqrt3z = np.sqrt(3.0) * z
        return (1.0 + sqrt3z + 3.0*z**2/3.0) * np.exp(-sqrt3z)
    
    def log_likelihood(params):
        if time.time() - start_time > DEADLINE - 5:
            return 0  # signal timeout
        l = np.exp(params[0])
        sigma2 = np.exp(params[1])
        
        K = matern52(D, l)
        K[np.diag_indices(n)] += 1e-12  # tiny jitter for numerical stability
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 0
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        ll = -0.5 * np.dot(y_norm, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
        return ll
    
    # Optimize length scale and signal variance
    best_ll = -np.inf
    best_params = (0.0, 0.0)
    
    # Try multiple initial values
    for l0 in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        for s0 in [0.5, 1.0, 2.0]:
            res = minimize_scalar(
                lambda p: -log_likelihood([p, np.log(s0)]),
                bounds=(np.log(0.01), np.log(50.0)),
                method='bounded',
                options={'xatol': 1e-6}
            )
            if time.time() - start_time > DEADLINE - 5:
                return None
            if res.success and -res.fun > best_ll:
                best_ll = -res.fun
                best_params = (res.x, np.log(s0))
    
    if best_ll == -np.inf:
        return None
    
    l_opt = np.exp(best_params[0])
    sigma2_opt = np.exp(best_params[1])
    
    # Build GP
    K = matern52(D, l_opt)
    K[np.diag_indices(n)] += 1e-12
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    
    def predict_gp(X_pred):
        X_pred = np.asarray(X_pred, dtype=np.float64)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(-1, 1)
        xp = (X_pred[:, 0] - x_mean) / x_std
        
        # K(x_new, x_train)
        Ks = np.abs(xp[:, None] - x_norm[None, :])
        Ks = matern52(Ks, l_opt)
        
        mu_norm = Ks @ alpha
        mu = mu_norm * y_std + y_mean
        
        return np.nan_to_num(mu, nan=y_mean, posinf=np.max(y), neginf=np.min(y))
    
    return predict_gp


def _fit_rbf_1d(x, y, start_time, DEADLINE):
    """RBF interpolation with multiple kernel types."""
    n = len(x)
    if n < 4:
        return None
    
    y_mean = np.mean(y)
    
    kernels = ['gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline']
    epsilons = [0.1, 0.3, 1.0, 3.0, 10.0]
    
    best_r2 = -np.inf
    best_predict = None
    
    for kernel in kernels:
        for eps in epsilons:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                rbf = Rbf(x, y, function=kernel, epsilon=eps)
                y_pred = rbf(x)
                sse = np.sum((y - y_pred)**2)
                sst = np.sum((y - y_mean)**2)
                r2 = 1 - sse / sst if sst > 0 else 1.0
                if r2 > best_r2:
                    best_r2 = r2
                    if r2 > 0.9999:
                        def predict_rbf(X_pred, _rbf=rbf):
                            X_pred = np.asarray(X_pred, dtype=np.float64)
                            if X_pred.ndim == 1:
                                X_pred = X_pred.reshape(-1, 1)
                            yp = _rbf(X_pred[:, 0])
                            return np.nan_to_num(yp, nan=y_mean, posinf=np.max(y), neginf=np.min(y))
                        return predict_rbf
            except Exception:
                continue
        if time.time() - start_time > DEADLINE - 5:
            break
    
    if best_r2 > 0.99:
        # Rebuild with best params - we need to track which was best
        # Since we returned early on >0.9999, here we just do a quick re-search
        for kernel in kernels:
            for eps in epsilons:
                try:
                    rbf = Rbf(x, y, function=kernel, epsilon=eps)
                    y_pred = rbf(x)
                    sse = np.sum((y - y_pred)**2)
                    sst = np.sum((y - y_mean)**2)
                    r2 = 1 - sse / sst if sst > 0 else 1.0
                    if abs(r2 - best_r2) < 1e-9:
                        def predict_rbf(X_pred, _rbf=rbf):
                            X_pred = np.asarray(X_pred, dtype=np.float64)
                            if X_pred.ndim == 1:
                                X_pred = X_pred.reshape(-1, 1)
                            yp = _rbf(X_pred[:, 0])
                            return np.nan_to_num(yp, nan=y_mean, posinf=np.max(y), neginf=np.min(y))
                        return predict_rbf
                except Exception:
                    continue
    
    return None


def _fit_piecewise_poly(x, y, start_time, DEADLINE):
    """Adaptive piecewise polynomial with varying segment counts."""
    n = len(x)
    if n < 8:
        return None
    
    y_mean = np.mean(y)
    best_r2 = -np.inf
    best_predict = None
    
    # Try different numbers of segments
    for n_seg in [4, 6, 8, 12, 16, 20]:
        if time.time() - start_time > DEADLINE - 5:
            break
        if n_seg > n // 3:
            continue
        
        # Create equal-width bins
        x_min, x_max = x[0], x[-1]
        edges = np.linspace(x_min, x_max, n_seg + 1)
        
        # Assign points to segments
        seg_idx = np.searchsorted(edges[1:-1], x)
        
        # Fit polynomial to each segment
        coeffs_list = []
        valid = True
        for s in range(n_seg):
            mask = seg_idx == s
            if np.sum(mask) < 4:
                valid = False
                break
            xs = x[mask]
            ys = y[mask]
            try:
                c = np.polyfit(xs, ys, 3)
                coeffs_list.append(c)
            except Exception:
                valid = False
                break
        
        if not valid:
            continue
        
        # Evaluate
        y_pred = np.zeros(n)
        for s in range(n_seg):
            mask = seg_idx == s
            y_pred[mask] = np.polyval(coeffs_list[s], x[mask])
        
        sse = np.sum((y - y_pred)**2)
        sst = np.sum((y - y_mean)**2)
        r2 = 1 - sse / sst if sst > 0 else 1.0
        
        if r2 > best_r2:
            best_r2 = r2
            if r2 > 0.9999:
                edges_copy = edges.copy()
                coeffs_copy = [c.copy() for c in coeffs_list]
                
                def predict_pw(X_pred, _edges=edges_copy, _coeffs=coeffs_copy, _y_mean=y_mean, _y_max=np.max(y), _y_min=np.min(y)):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    xp = X_pred[:, 0]
                    yp = np.zeros(len(xp))
                    # Clip to range
                    xp_clipped = np.clip(xp, _edges[0], _edges[-1])
                    sidx = np.searchsorted(_edges[1:-1], xp_clipped)
                    for s in range(len(_coeffs)):
                        mask = sidx == s
                        if np.any(mask):
                            yp[mask] = np.polyval(_coeffs[s], xp_clipped[mask])
                    return np.nan_to_num(yp, nan=_y_mean, posinf=_y_max, neginf=_y_min)
                return predict_pw
    
    return None


def _fit_basis(x, y, start_time, DEADLINE):
    """Basis function regression with a rich set of features."""
    n = len(x)
    y_mean = np.mean(y)
    y_centered = y - y_mean
    y_norm = np.linalg.norm(y_centered)
    if y_norm < 1e-12:
        return None
    
    # Build basis
    cols = [np.ones(n)]
    
    # Polynomials up to degree 12
    for p in range(1, 13):
        cols.append(x ** p)
    
    # Trigonometric
    for k in [1, 2, 3, 4, 5]:
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    
    # Exponential
    for a in [0.5, 1.0, 2.0]:
        cols.append(np.exp(a * x))
        cols.append(np.exp(-a * x))
    
    # Log
    cols.append(np.log(np.abs(x) + 1e-12))
    cols.append(np.log1p(np.abs(x)))
    
    # Inverse
    cols.append(1.0 / (1.0 + x**2))
    cols.append(1.0 / (1.0 + np.abs(x)))
    
    # Sigmoid / tanh
    cols.append(1.0 / (1.0 + np.exp(-x)))
    cols.append(np.tanh(x))
    cols.append(np.tanh(2 * x))
    
    # Sqrt
    cols.append(np.sqrt(np.abs(x) + 1e-12))
    
    # x * trig
    cols.append(x * np.sin(x))
    cols.append(x * np.cos(x))
    
    # x * exp
    cols.append(x * np.exp(-x**2 / 2))
    
    B = np.column_stack(cols)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    
    B_centered = B - np.mean(B, axis=0)
    
    # Single feature search
    best_r2 = -np.inf
    best_idx = -1
    for j in range(B.shape[1]):
        b_norm = np.linalg.norm(B_centered[:, j])
        if b_norm < 1e-12:
            continue
        corr = np.dot(B_centered[:, j], y_centered) / (b_norm * y_norm)
        r2 = corr * corr
        if r2 > best_r2:
            best_r2 = r2
            best_idx = j
    
    if best_r2 > 0.9999:
        X_aug = np.column_stack([np.ones(n), B[:, best_idx]])
        w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        
        def predict_single(X_pred, _idx=best_idx, _w=w):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            xp = X_pred[:, 0]
            Bp = _build_basis(xp)
            Bp = np.nan_to_num(Bp, nan=0.0, posinf=0.0, neginf=0.0)
            X_aug_p = np.column_stack([np.ones(len(xp)), Bp[:, _idx]])
            return X_aug_p @ _w
        
        return predict_single
    
    # Pair search among top features
    if best_r2 < 0.999 and time_left_global(start_time, DEADLINE) > 10:
        r2_scores = []
        for j in range(B.shape[1]):
            b_norm = np.linalg.norm(B_centered[:, j])
            if b_norm < 1e-12:
                r2_scores.append((0, j))
            else:
                corr = np.dot(B_centered[:, j], y_centered) / (b_norm * y_norm)
                r2_scores.append((corr*corr, j))
        
        r2_scores.sort(reverse=True)
        top_k = min(12, len(r2_scores))
        top_indices = [idx for _, idx in r2_scores[:top_k]]
        
        best_pair_r2 = best_r2
        best_pair = (best_idx, -1)
        
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                idx1, idx2 = top_indices[i], top_indices[j]
                X_pair = np.column_stack([np.ones(n), B[:, idx1], B[:, idx2]])
                try:
                    w_pair, *_ = np.linalg.lstsq(X_pair, y, rcond=None)
                    y_hat = X_pair @ w_pair
                    sse = np.sum((y - y_hat)**2)
                    r2_pair = 1 - sse / np.sum(y_centered**2)
                    if r2_pair > best_pair_r2:
                        best_pair_r2 = r2_pair
                        best_pair = (idx1, idx2)
                        if r2_pair > 0.9999:
                            break
                except Exception:
                    pass
                if time.time() - start_time > DEADLINE - 5:
                    break
            if time.time() - start_time > DEADLINE - 5:
                break
            if best_pair_r2 > 0.9999:
                break
        
        if best_pair_r2 > 0.9999:
            idx1, idx2 = best_pair
            X_f = np.column_stack([np.ones(n), B[:, idx1], B[:, idx2]])
            w, *_ = np.linalg.lstsq(X_f, y, rcond=None)
            
            def predict_pair(X_pred, _i1=idx1, _i2=idx2, _w=w):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                xp = X_pred[:, 0]
                Bp = _build_basis(xp)
                Bp = np.nan_to_num(Bp, nan=0.0, posinf=0.0, neginf=0.0)
                X_f_p = np.column_stack([np.ones(len(xp)), Bp[:, _i1], Bp[:, _i2]])
                return X_f_p @ _w
            
            return predict_pair
    
    return None


def _build_basis(xp):
    """Build the same basis as in _fit_basis for prediction."""
    n = len(xp)
    cols = [np.ones(n)]
    for p in range(1, 13):
        cols.append(xp ** p)
    for k in [1, 2, 3, 4, 5]:
        cols.append(np.sin(k * xp))
        cols.append(np.cos(k * xp))
    for a in [0.5, 1.0, 2.0]:
        cols.append(np.exp(a * xp))
        cols.append(np.exp(-a * xp))
    cols.append(np.log(np.abs(xp) + 1e-12))
    cols.append(np.log1p(np.abs(xp)))
    cols.append(1.0 / (1.0 + xp**2))
    cols.append(1.0 / (1.0 + np.abs(xp)))
    cols.append(1.0 / (1.0 + np.exp(-xp)))
    cols.append(np.tanh(xp))
    cols.append(np.tanh(2 * xp))
    cols.append(np.sqrt(np.abs(xp) + 1e-12))
    cols.append(xp * np.sin(xp))
    cols.append(xp * np.cos(xp))
    cols.append(xp * np.exp(-xp**2 / 2))
    return np.column_stack(cols)


def time_left_global(start_time, DEADLINE):
    return DEADLINE - (time.time() - start_time)


def _fit_rbf(X, y, start_time, DEADLINE):
    """RBF fallback for multi-dimensional data."""
    n, d = X.shape
    y_mean = np.mean(y)
    
    kernels = ['gaussian', 'multiquadric', 'inverse_multiquadric']
    epsilons = [1.0, 3.0, 10.0, 30.0]
    
    best_r2 = -np.inf
    best_predict = None
    
    for kernel in kernels:
        for eps in epsilons:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                rbf = Rbf(X, y, function=kernel, epsilon=eps)
                y_pred = rbf(X)
                sse = np.sum((y - y_pred)**2)
                sst = np.sum((y - y_mean)**2)
                r2 = 1 - sse / sst if sst > 0 else 1.0
                if r2 > best_r2:
                    best_r2 = r2
                    if r2 > 0.999:
                        def predict_rbf(X_pred, _rbf=rbf):
                            X_pred = np.asarray(X_pred, dtype=np.float64)
                            if X_pred.ndim == 1:
                                X_pred = X_pred.reshape(1, -1)
                            yp = _rbf(X_pred)
                            return np.nan_to_num(yp, nan=y_mean, posinf=np.max(y), neginf=np.min(y))
                        return predict_rbf
            except Exception:
                continue
    
    # Fallback: nearest neighbor
    def predict_nn(X_pred):
        X_pred = np.asarray(X_pred, dtype=np.float64)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1)
        D = cdist(X_pred, X)
        idx = np.argmin(D, axis=1)
        return y[idx]
    
    return predict_nn
