"""Symbolic regression via Gaussian Process with optimized RBF kernel."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cho_inv


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize inputs
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    # Standardize y for GP (GP is scale-invariant in y if we include sigma)
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    ys = (y - y_mean) / y_std

    def rbf_kernel(X1, X2, lengthscales, variance):
        """Compute RBF kernel matrix between X1 and X2."""
        # Scale inputs by inverse lengthscales
        X1s = X1 / lengthscales
        X2s = X2 / lengthscales
        # Compute squared distances
        sq1 = np.sum(X1s ** 2, axis=1)[:, None]
        sq2 = np.sum(X2s ** 2, axis=1)[None, :]
        sq_d = sq1 + sq2 - 2.0 * X1s @ X2s.T
        np.maximum(sq_d, 0.0, out=sq_d)  # numerical stability
        return variance * np.exp(-0.5 * sq_d)

    def log_marginal_likelihood(theta):
        """Negative log marginal likelihood for optimization."""
        # theta = [log(lengthscale_0), ..., log(lengthscale_{d-1}), log(variance), log(noise)]
        logs = theta
        lengthscales = np.exp(logs[:d])
        variance = np.exp(logs[d])
        noise = np.exp(logs[d + 1])

        K = rbf_kernel(Xs, Xs, lengthscales, variance)
        K[np.diag_indices(n)] += noise

        try:
            L, info = cho_factor(K, lower=True, check_finite=False)
            if info > 0:
                return 1e10  # Cholesky failed
            # Log determinant
            log_det = 2.0 * np.sum(np.log(np.diag(cho_factor(K, lower=True, check_finite=False)[0])))
            # Actually, cho_factor returns L, so log_det = 2 * sum(log(diag(L)))
            L_mat = cho_factor(K, lower=True, check_finite=False)[0]
            log_det = 2.0 * np.sum(np.log(np.diag(L_mat)))
            # Solve K^{-1} y
            alpha = cho_solve((L_mat, True), ys)
            # Negative log marginal likelihood
            nll = 0.5 * ys @ alpha + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
            return nll
        except Exception:
            return 1e10

    # Initialize hyperparameters
    # Start with moderate lengthscales
    init_lengthscales = np.ones(d)
    init_variance = 1.0
    init_noise = 1e-2

    theta0 = np.concatenate([
        np.log(init_lengthscales),
        [np.log(init_variance), np.log(init_noise)]
    ])

    best_theta = theta0.copy()
    best_nll = 1e10

    # Optimize hyperparameters
    try:
        result = minimize(
            log_marginal_likelihood,
            theta0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
        )
        if result.fun < best_nll:
            best_nll = result.fun
            best_theta = result.x
    except Exception:
        pass

    # Check time
    if time.time() - t0 > DEADLINE - 5.0:
        # Fallback to polynomial ridge
        return _poly_fallback(X, y, mu, sd)

    # Extract best hyperparameters
    best_lengthscales = np.exp(best_theta[:d])
    best_variance = np.exp(best_theta[d])
    best_noise = np.exp(best_theta[d + 1])

    # Precompute Cholesky for fast prediction
    K = rbf_kernel(Xs, Xs, best_lengthscales, best_variance)
    K[np.diag_indices(n)] += best_noise

    try:
        L, info = cho_factor(K, lower=True, check_finite=False)
        if info > 0:
            # Fallback if Cholesky fails
            return _poly_fallback(X, y, mu, sd)
        alpha = cho_solve((L, True), ys)
    except Exception:
        return _poly_fallback(X, y, mu, sd)

    def predict(X_new):
        Xn = np.asarray(X_new, dtype=np.float64)
        Xn_std = (Xn - mu) / sd
        K_star = rbf_kernel(Xn_std, Xs, best_lengthscales, best_variance)
        pred_std = K_star @ alpha
        pred = y_mean + y_std * pred_std
        return pred

    return predict


def _poly_fallback(X, y, mu, sd):
    """Degree-3 polynomial ridge regression as fallback."""
    Xs = (X - mu) / sd
    cols = [np.ones(len(Xs))]
    for j in range(Xs.shape[1]):
        cols.append(Xs[:, j])
    for i in range(Xs.shape[1]):
        for j in range(i, Xs.shape[1]):
            cols.append(Xs[:, i] * Xs[:, j])
    for i in range(Xs.shape[1]):
        for j in range(Xs.shape[1]):
            for k in range(j, Xs.shape[1]):
                cols.append(Xs[:, i] * Xs[:, j] * Xs[:, k])
    F = np.column_stack(cols)
    # Ridge regression
    lam = 1e-3
    w, *_ = np.linalg.lstsq(F + lam * np.eye(F.shape[1]), y, rcond=None)

    def predict(X_new):
        Xn = np.asarray(X_new, dtype=np.float64)
        Xn_std = (Xn - mu) / sd
        cols = [np.ones(len(Xn_std))]
        for j in range(Xn_std.shape[1]):
            cols.append(Xn_std[:, j])
        for i in range(Xn_std.shape[1]):
            for j in range(i, Xn_std.shape[1]):
                cols.append(Xn_std[:, i] * Xn_std[:, j])
        for i in range(Xn_std.shape[1]):
            for j in range(Xn_std.shape[1]):
                for k in range(j, Xn_std.shape[1]):
                    cols.append(Xn_std[:, i] * Xn_std[:, j] * Xn_std[:, k])
        Fn = np.column_stack(cols)
        return Fn @ w

    return predict
# EVOLVE-BLOCK-END
