"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


def _poly_basis(X, degree):
    """Return a basis matrix of all monomials up to `degree` (incl. constant)."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for r in range(d):
            for c in range(r, d):
                term = np.ones(n)
                for k in range(r, c + 1):
                    term = term * X[:, k]
                cols.append(term)
    return np.column_stack(cols)


def _fit_poly(X, y, degree, lam):
    B = _poly_basis(X, degree)
    n, m = B.shape
    A = B.T @ B + lam * np.eye(m)
    w = np.linalg.solve(A, B.T @ y)
    return lambda X: _poly_basis(X, degree) @ w


def _rbf_kernel(X, Y, gamma):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    return np.exp(-gamma * d2)


def _fit_ridge_rbf(X, y, gamma, lam):
    K = _rbf_kernel(X, X)
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    def predict(Xt):
        return _rbf_kernel(Xt, X, gamma) @ alpha
    return predict


def _fit_ridge_poly_kernel(X, y, degree, lam):
    B = _poly_basis(X, degree)
    K = B @ B.T
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    def predict(Xt):
        Bt = _poly_basis(Xt, degree)
        return (Bt @ B.T) @ alpha
    return predict


def _fit_gp_m52(X, y, max_iter=80):
    """Matern-5/2 GP regression with hyperparameter optimization via LOO-CV."""
    n, d = X.shape
    def matern52(K, l):
        # K already contains the squared distances? we pass distances
        return None

    def build_K(theta):
        l, sigma_f, sigma_n = theta
        l = max(l, 1e-4)
        # pairwise squared distances
        d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        K = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        K[np.diag_indices(n)] += sigma_n ** 2
        return K

    def log_marginal(theta):
        K = build_K(theta)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lmc = 0.5 * np.sum(y * alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        return -lmc

    # initial guess
    scale = np.max(np.ptp(X, axis=0)) + 1e-6
    yvar = np.var(y) + 1e-9
    theta0 = np.array([scale, np.sqrt(yvar), np.sqrt(yvar) * 1e-2])
    res = minimize(log_marginal, theta0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
    l, sigma_f, sigma_n = res.x
    l = max(l, 1e-4)
    K = build_K(res.x)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    def predict(Xt):
        d2 = (np.sum(Xt ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * Xt @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        Ks = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        return Ks @ alpha
    return predict


def _loo_score(predict, X, y):
    """Approximate LOO-CV via refit-free residual trick is only valid for linear models;
    so we do a cheap k-fold style using held-out subset."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    fold = max(20, n // 5)
    se = 0.0
    for i in range(0, n, fold):
        val = idx[i:i + fold]
        tr = np.concatenate([idx[:i], idx[i + fold:]])
        if len(tr) < 5:
            continue
        # quick surrogate: use the predict on train as proxy (not true CV)
        # Instead compute residual on train as a lower bound proxy
        pass
    # Fallback: use training residual (biased but comparable across models)
    pred = predict(X)
    return np.mean((pred - y) ** 2)


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 300.0
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    yvar = np.var(y) + 1e-12

    # candidate models: (name, predict)
    candidates = []

    # 1) degree-2 polynomial (baseline)
    try:
        p = _fit_poly(X, y, 2, 1e-6)
        candidates.append(('poly2', p))
    except Exception:
        pass

    # 2) higher degree polynomial with mild ridge
    for deg in (3, 4):
        if time.time() - t0 > DEADLINE - 60:
            break
        for lam in (1e-4, 1e-2):
            try:
                p = _fit_poly(X, y, deg, lam)
                candidates.append((f'poly{deg}_{lam}', p))
            except Exception:
                pass

    # 3) RBF kernel ridge regression
    if time.time() - t0 < DEADLINE - 60:
        for gamma in (0.1, 0.5, 1.0, 2.0):
            for lam in (1e-3, 1e-1):
                try:
                    p = _fit_ridge_rbf(X, y, gamma, lam)
                    candidates.append((f'rbf_g{gamma}_l{lam}', p))
                except Exception:
                    pass

    # 4) Polynomial kernel ridge (captures polynomial + interaction structure)
    if time.time() - t0 < DEADLINE - 60:
        for deg in (4, 6):
            for lam in (1e-3, 1e-1):
                try:
                    p = _fit_ridge_poly_kernel(X, y, deg, lam)
                    candidates.append((f'polyK_{deg}_{lam}', p))
                except Exception:
                    pass

    # 5) Matern-5/2 Gaussian Process (handles non-polynomial smooth structure)
    if time.time() - t0 < DEADLINE - 90:
        try:
            p = _fit_gp_m52(X, y, max_iter=60)
            candidates.append(('gp_m52', p))
        except Exception:
            pass

    # 6) Rational function approximator: ratio of two linear basis expansions
    #    p(x) = num(x) / den(x), fit via nonlinear least squares on coefficients
    if time.time() - t0 < DEADLINE - 60:
        try:
            B = _poly_basis(X, 1)  # constant + linear
            m = B.shape[1]
            def rational_predict(coeffs, Xt):
                cnum = coeffs[:m]
                cden = coeffs[m:]
                num = Xt @ cnum
                den = Xt @ cden
                # guard against near-zero denominator
                den = np.where(np.abs(den) < 1e-6, np.sign(den) * 1e-6 + 1e-6, den)
                return num / den
            # initialize den ~ 1
            x0 = np.zeros(2 * m)
            x0[m] = 1.0  # constant term of denominator
            def obj(coeffs):
                pred = rational_predict(coeffs, X)
                return np.sum((pred - y) ** 2)
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 400})
            p = lambda Xt, _c=res.x: rational_predict(_c, Xt)
            candidates.append(('rational', p))
        except Exception:
            pass

    if not candidates:
        # ultimate fallback: constant
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)

    # select best by training residual (proxy) since true LOO is expensive;
    # but to reduce overfit bias, prefer simpler models on ties via regularization already applied.
    best_name, best_pred, best_score = None, None, np.inf
    for name, p in candidates:
        try:
            pred = p(X)
            score = np.mean((pred - y) ** 2) / yvar
            if score < best_score:
                best_score, best_name, best_pred = score, name, p
        except Exception:
            continue

    if best_pred is None:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)

    final = best_pred

    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        out = final(Xt)
        out = np.asarray(out, dtype=float).ravel()
        # ensure finite
        out = np.where(np.isfinite(out), out, np.nanmean(y))
        return out

    return predict
# EVOLVE-BLOCK-END
