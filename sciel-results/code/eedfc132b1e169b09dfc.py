"""Symbolic regression via diverse basis search with cross-validation."""
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def fit(X_train, y_train):
    """
    Fit a regression model and return a predict callable.
    Uses cross-validation to select the best model from many candidates.
    """
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0]

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    # Simple k-fold cross-validation score
    def cv_score(build_predictor, k=5):
        """Estimate test error via k-fold CV on training data."""
        if elapsed() > deadline - 5:
            return np.inf
        try:
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            folds = np.array_split(indices, k)
            errors = []
            for i in range(k):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                try:
                    pred_func = build_predictor(X[tr_idx], y[tr_idx])
                    if pred_func is None:
                        return np.inf
                    pred_val = pred_func(X[val_idx])
                    pred_val = np.asarray(pred_val, dtype=np.float64)
                    if not np.all(np.isfinite(pred_val)):
                        return np.inf
                    mse = np.mean((pred_val - y[val_idx]) ** 2)
                    errors.append(mse)
                except Exception:
                    return np.inf
            return np.mean(errors)
        except Exception:
            return np.inf

    best_cv = np.inf
    best_predictor = None

    def try_model(build_predictor_full):
        """Try a model builder that takes (X, y) and returns a predictor."""
        nonlocal best_cv, best_predictor
        if remaining() < 3:
            return
        try:
            cv = cv_score(build_predictor_full)
            if cv < best_cv:
                best_cv = cv
                # Rebuild with all data
                pred_func = build_predictor_full(X, y)
                if pred_func is not None:
                    # Verify it works on full training set
                    pred_check = pred_func(X)
                    pred_check = np.asarray(pred_check, dtype=np.float64)
                    if np.all(np.isfinite(pred_check)):
                        best_predictor = pred_func
        except Exception:
            pass

    # ===== Model builders =====

    # 1) Polynomial of degree d
    def build_poly(deg):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                A = np.vander(x_fit, deg + 1, increasing=True)
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    Ai = np.vander(xi, deg + 1, increasing=True)
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 2) Fourier series
    def build_fourier(max_k):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                n_fit = len(x_fit)
                cols = [np.ones(n_fit)]
                for k in range(1, max_k + 1):
                    cols.append(np.sin(k * x_fit))
                    cols.append(np.cos(k * x_fit))
                A = np.column_stack(cols)
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    cols_i = [np.ones(len(xi))]
                    for k in range(1, max_k + 1):
                        cols_i.append(np.sin(k * xi))
                        cols_i.append(np.cos(k * xi))
                    Ai = np.column_stack(cols_i)
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 3) Transformed features: f(x) with linear fit
    def build_transformed(transform_func, name):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                Xf = transform_func(x_fit)
                if not np.all(np.isfinite(Xf)):
                    return None
                # Fit linear in transformed space: a + b*f(x) + c*f(x)^2
                A = np.column_stack([np.ones(len(x_fit)), Xf, Xf ** 2])
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    xf = transform_func(xi)
                    if not np.all(np.isfinite(xf)):
                        return np.zeros(len(xi))
                    Ai = np.column_stack([np.ones(len(xi)), xf, xf ** 2])
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 4) RBF network
    def build_rbf(n_centers, bandwidth):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                centers = np.quantile(x_fit, np.linspace(0.05, 0.95, n_centers))
                Xc = x_fit.reshape(-1, 1)
                C = centers.reshape(1, -1)
                dist2 = (Xc - C) ** 2
                A = np.exp(-dist2 / (2 * bandwidth ** 2))
                A = np.column_stack([np.ones(len(x_fit)), A])
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0].reshape(-1, 1)
                    dist2_i = (xi - C) ** 2
                    Ai = np.exp(-dist2_i / (2 * bandwidth ** 2))
                    Ai = np.column_stack([np.ones(len(xi)), Ai])
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 5) Piecewise linear (spline)
    def build_spline(n_knots):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                knots = np.quantile(x_fit, np.linspace(0, 1, n_knots + 1))
                cols = [np.ones(len(x_fit)), x_fit]
                for knot in knots[1:-1]:
                    cols.append(np.maximum(0, x_fit - knot))
                A = np.column_stack(cols)
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    cols_i = [np.ones(len(xi)), xi]
                    for knot in knots[1:-1]:
                        cols_i.append(np.maximum(0, xi - knot))
                    Ai = np.column_stack(cols_i)
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 6) Exponential / log / reciprocal forms
    def build_exp_form(a_range, b_range):
        """Fit y = a * exp(b*x) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                # Nonlinear fit: y = a*exp(b*x) + c
                from scipy.optimize import curve_fit
                def model(x, a, b, c):
                    return a * np.exp(np.clip(b * x, -20, 20)) + c
                # Initial guess
                y_mean = np.mean(y_fit)
                y_range = np.max(y_fit) - np.min(y_fit)
                p0 = [y_range / 2, 0.1, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=1000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c = popt
                    def predict(X_in):
                        xi = X_in[:, 0]
                        return a * np.exp(np.clip(b * xi, -20, 20)) + c
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    def build_log_form():
        """Fit y = a*log(x+b) + c for x>0"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                # Only valid if x > 0 mostly
                if np.min(x_fit) < 0:
                    return None
                from scipy.optimize import curve_fit
                def model(x, a, b, c):
                    arg = x + b
                    if np.min(arg) <= 0:
                        return np.full(len(x), np.nan)
                    return a * np.log(arg) + c
                y_mean = np.mean(y_fit)
                p0 = [1.0, 0.1, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=1000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c = popt
                    def predict(X_in):
                        xi = X_in[:, 0]
                        arg = xi + b
                        mask = arg > 0
                        res = np.zeros(len(xi))
                        res[mask] = a * np.log(arg[mask]) + c
                        if not np.all(mask):
                            res[~mask] = a * np.log(max(b + 0.1, 0.1)) + c
                        return res
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    def build_reciprocal_form():
        """Fit y = a/(x+b) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                from scipy.optimize import curve_fit
                def model(x, a, b, c):
                    denom = x + b
                    mask = np.abs(denom) > 1e-10
                    res = np.full(len(x), c)
                    res[mask] = a / denom[mask] + c
                    return res
                y_mean = np.mean(y_fit)
                p0 = [1.0, 0.5, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=1000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c = popt
                    def predict(X_in):
                        xi = X_in[:, 0]
                        denom = xi + b
                        mask = np.abs(denom) > 1e-10
                        res = np.full(len(xi), c)
                        res[mask] = a / denom[mask] + c
                        return res
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    def build_quadratic_form():
        """Fit y = a*x^2 + b*x + c (same as poly degree 2, but explicit)"""
        return build_poly(2)

    def build_cubic_form():
        return build_poly(3)

    def build_sin_cos_single():
        """y = a*sin(x) + b*cos(x) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                A = np.column_stack([np.ones(len(x_fit)), np.sin(x_fit), np.cos(x_fit)])
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    Ai = np.column_stack([np.ones(len(xi)), np.sin(xi), np.cos(xi)])
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    def build_tanh_form():
        """y = a*tanh(b*x) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                from scipy.optimize import curve_fit
                def model(x, a, b, c):
                    return a * np.tanh(b * x) + c
                y_range = np.max(y_fit) - np.min(y_fit)
                y_mean = np.mean(y_fit)
                p0 = [y_range / 2, 1.0, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=1000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c = popt
                    def predict(X_in):
                        xi = X_in[:, 0]
                        return a * np.tanh(b * xi) + c
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    def build_gaussian_bump():
        """y = a * exp(-((x-b)/c)^2) + d"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                from scipy.optimize import curve_fit
                def model(x, a, b, c, d):
                    return a * np.exp(-((x - b) / c) ** 2) + d
                y_range = np.max(y_fit) - np.min(y_fit)
                y_mean = np.mean(y_fit)
                x_mean = np.mean(x_fit)
                p0 = [y_range, x_mean, 1.0, y_mean - y_range / 2]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=2000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c, d = popt
                    if abs(c) < 1e-10:
                        return None
                    def predict(X_in):
                        xi = X_in[:, 0]
                        return a * np.exp(-((xi - b) / c) ** 2) + d
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    # ===== Try all models =====

    # Polynomials
    for deg in range(1, 13):
        if remaining() < 5:
            break
        try_model(build_poly(deg))

    # Fourier
    for max_k in range(1, 10):
        if remaining() < 5:
            break
        try_model(build_fourier(max_k))

    # Transformed features
    transforms = [
        ("identity", lambda t: t),
        ("reciprocal", lambda t: 1.0 / np.where(np.abs(t) < 1e-10, 1e-10, t)),
        ("inv_quad", lambda t: 1.0 / (1.0 + t ** 2)),
        ("exp", lambda t: np.exp(np.clip(t, -10, 10))),
        ("tanh", lambda t: np.tanh(t)),
        ("log_abs", lambda t: np.log(np.abs(t) + 1e-10)),
        ("sqrt_abs", lambda t: np.sqrt(np.abs(t))),
        ("x_cubed", lambda t: t ** 3),
    ]
    for name, tf in transforms:
        if remaining() < 5:
            break
        try_model(build_transformed(tf, name))

    # RBF
    for nc in [5, 10, 20, 30, 50]:
        for bw in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
            if remaining() < 5:
                break
            try_model(build_rbf(nc, bw))

    # Splines
    for nk in [5, 10, 20, 30, 50]:
        if remaining() < 5:
            break
        try_model(build_spline(nk))

    # Parametric forms
    if remaining() > 10:
        try_model(build_exp_form(None, None))
    if remaining() > 10:
        try_model(build_log_form())
    if remaining() > 10:
        try_model(build_reciprocal_form())
    if remaining() > 10:
        try_model(build_cubic_form())
    if remaining() > 10:
        try_model(build_sin_cos_single())
    if remaining() > 10:
        try_model(build_tanh_form())
    if remaining() > 10:
        try_model(build_gaussian_bump())

    # Additional: try combinations of basis functions
    if remaining() > 10:
        # x, x^2, sin(x), cos(x), exp(x), 1/x
        def build_combo():
            def build(X_fit, y_fit):
                try:
                    x_fit = X_fit[:, 0]
                    safe_x = np.where(np.abs(x_fit) < 1e-10, 1e-10, x_fit)
                    cols = [
                        np.ones(len(x_fit)),
                        x_fit,
                        x_fit ** 2,
                        np.sin(x_fit),
                        np.cos(x_fit),
                        np.exp(np.clip(x_fit, -10, 10)),
                        1.0 / safe_x,
                        np.tanh(x_fit),
                    ]
                    A = np.column_stack(cols)
                    w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                    if not np.all(np.isfinite(w)):
                        return None
                    def predict(X_in):
                        xi = X_in[:, 0]
                        safe_xi = np.where(np.abs(xi) < 1e-10, 1e-10, xi)
                        cols_i = [
                            np.ones(len(xi)),
                            xi,
                            xi ** 2,
                            np.sin(xi),
                            np.cos(xi),
                            np.exp(np.clip(xi, -10, 10)),
                            1.0 / safe_xi,
                            np.tanh(xi),
                        ]
                        Ai = np.column_stack(cols_i)
                        return Ai @ w
                    return predict
                except Exception:
                    return None
            return build
        try_model(build_combo())

    # Fallback: constant
    if best_predictor is None:
        mean_y = np.mean(y)
        best_predictor = lambda X_in: np.full(len(X_in), mean_y)

    # Final safety wrapper
    def final_predict(X_in):
        X_in = np.asarray(X_in, dtype=np.float64)
        try:
            res = best_predictor(X_in)
            res = np.asarray(res, dtype=np.float64)
            if res.shape != (len(X_in),):
                res = np.atleast_1d(res).flatten()
                if len(res) != len(X_in):
                    res = np.resize(res, len(X_in))
            if not np.all(np.isfinite(res)):
                res = np.where(np.isfinite(res), res, np.mean(y))
            return res
        except Exception:
            return np.full(len(X_in), np.mean(y))

    return final_predict

# EVOLVE-BLOCK-END
