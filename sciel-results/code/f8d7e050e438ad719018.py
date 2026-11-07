# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_basis(X, d):
    """Build a rich set of candidate basis functions evaluated on X. Returns (N, p) array."""
    N = X.shape[0]
    cols = []

    # Constant
    cols.append(np.ones(N))

    # Linear
    for j in range(d):
        cols.append(X[:, j])

    # Squares through 6th power
    for j in range(d):
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
        cols.append(X[:, j] ** 4)
        cols.append(X[:, j] ** 5)
        cols.append(X[:, j] ** 6)

    # Pairwise products
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])

    # Triple products
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])

    # Quartic cross terms (distinct indices)
    if d >= 4:
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for l in range(k + 1, d):
                        cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])

    # Trig: sin/cos of x, 2x, 3x
    for j in range(d):
        xj = X[:, j]
        cols.append(np.sin(xj))
        cols.append(np.cos(xj))
        cols.append(np.sin(2.0 * xj))
        cols.append(np.cos(2.0 * xj))
        cols.append(np.sin(3.0 * xj))
        cols.append(np.cos(3.0 * xj))

    # Exp, log, sqrt, sigmoid, tanh
    for j in range(d):
        xj = X[:, j]
        cols.append(np.exp(xj))
        cols.append(np.exp(-xj))
        cols.append(np.exp(2.0 * xj))
        cols.append(np.exp(-2.0 * xj))
        cols.append(np.log1p(xj))
        cols.append(np.sqrt(xj))
        cols.append(np.tanh(xj))
        cols.append(np.tanh(2.0 * xj))
        cols.append(1.0 / (1.0 + np.exp(-xj)))

    # Cross terms with trig and exp
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                xi, xj = X[:, i], X[:, j]
                cols.append(np.sin(xi) * xj)
                cols.append(np.cos(xi) * xj)
                cols.append(np.sin(xi) * np.sin(xj))
                cols.append(np.cos(xi) * np.cos(xj))
                cols.append(xi * np.exp(xj))
                cols.append(np.exp(xi) * xj)
                cols.append(xi * np.sqrt(xj))
                cols.append(np.sqrt(xi) * xj)
                cols.append(np.sin(xi) * np.exp(xj))
                cols.append(np.cos(xi) * np.exp(xj))

    # Global features
    cols.append(np.sum(X, axis=1))
    cols.append(np.prod(X, axis=1))
    cols.append(np.sum(X ** 2, axis=1))
    cols.append(np.sum(X ** 3, axis=1))
    cols.append(np.linalg.norm(X, axis=1))
    cols.append(np.linalg.norm(X, axis=1) ** 2)
    cols.append(np.min(X, axis=1))
    cols.append(np.max(X, axis=1))

    # Differences
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] - X[:, j])
                cols.append(X[:, i] * (X[:, i] - X[:, j]))

    return np.column_stack(cols)


def _lasso_coord_descent(F, y, n_iter=200, lam=1e-4):
    """
    Lasso via coordinate descent.
    Returns (w, active_indices).
    """
    n, p = F.shape
    w = np.zeros(p)
    if p == 0:
        return w, []

    # Precompute column norms squared
    col_norm_sq = np.sum(F ** 2, axis=0)
    col_norm_sq = np.maximum(col_norm_sq, 1e-15)

    # Precompute F^T y
    FTy = F.T @ y

    for _ in range(n_iter):
        if _time_left() < 3.0:
            break
        for j in range(p):
            # rho_j = F_j^T * y - (F^T F)_j * w + F_j^T F_j * w_j
            # = FTy[j] - sum_k F_j^T F_k w_k + col_norm_sq[j] * w[j]
            # = FTy[j] - (F^T F w)[j] + col_norm_sq[j] * w[j]
            # But (F^T F w)[j] = sum_k F_j^T F_k w_k
            # We can compute F^T F w incrementally, but for simplicity:
            # resid_no_j = y - F w + F_j w_j
            # rho_j = F_j^T resid_no_j = F_j^T y - F_j^T F w + col_norm_sq[j] w_j
            # = FTy[j] - (F^T F w)[j] + col_norm_sq[j] w_j
            # Let's compute F^T F w each time? That's O(p^2) per iter. Too slow.
            # Instead, maintain a running "residual" vector.
            pass

    # Let's do it with residual tracking
    w = np.zeros(p)
    residual = y.copy()  # y - F @ w

    for _ in range(n_iter):
        if _time_left() < 3.0:
            break
        for j in range(p):
            # Update w_j
            # rho_j = F_j^T residual + F_j^T F_j w_j = F_j^T residual + col_norm_sq[j] w_j
            rho_j = F[:, j] @ residual + col_norm_sq[j] * w[j]
            if rho_j > lam:
                w_j_new = (rho_j - lam) / col_norm_sq[j]
            elif rho_j < -lam:
                w_j_new = (rho_j + lam) / col_norm_sq[j]
            else:
                w_j_new = 0.0
            if w_j_new != w[j]:
                residual -= F[:, j] * (w_j_new - w[j])
                w[j] = w_j_new

    active = list(np.where(np.abs(w) > 1e-12)[0])
    return w, active


def _fit_ridge(F, y, alpha=1e-8):
    n, p = F.shape
    A = F.T @ F + alpha * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return w


def _rbf_predict(X_train, y_train, X, gamma, alpha=1e-6):
    Xa = X_train
    Xb = X
    sq_a = np.sum(Xa ** 2, axis=1)[:, None]
    sq_b = np.sum(Xb ** 2, axis=1)[None, :]
    K_train = np.exp(-gamma * (sq_a + sq_a.T - 2.0 * Xa @ Xa.T))
    K_test = np.exp(-gamma * (sq_a + sq_b - 2.0 * Xa @ X.T))

    n = X_train.shape[0]
    A = K_train + alpha * np.eye(n)
    try:
        theta = np.linalg.solve(A, y_train)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, y_train, rcond=None)[0]
    return K_test @ theta


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape

    if N == 0:
        return lambda X: np.zeros(np.asarray(X).shape[0])

    # Build basis
    F = _build_basis(X_train, d)
    p = F.shape[1]

    # Ensure finite
    finite_cols = np.all(np.isfinite(F), axis=0)
    F = F[:, finite_cols]
    p = F.shape[1]
    finite_mask = finite_cols

    if p == 0:
        return lambda X: np.zeros(np.asarray(X).shape[0])

    # Standardize features (keep constant)
    F_mean = np.mean(F, axis=0)
    F_std = np.std(F, axis=0)
    F_std[F_std < 1e-12] = 1.0
    F_std[0] = 1.0
    F_mean[0] = 0.0
    F_norm = (F - F_mean) / F_std

    # y stats
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-12:
        return lambda X: np.full(np.asarray(X).shape[0], y_mean)
    y_norm = (y_train - y_mean) / y_std

    # Lasso
    w_norm, active = _lasso_coord_descent(F_norm, y_norm, n_iter=200, lam=1e-4)

    if len(active) == 0:
        active = [0]

    F_active = F_norm[:, active]
    w_active = _fit_ridge(F_active, y_norm, alpha=1e-8)

    pred_train_norm = F_active @ w_active
    train_mse_norm = np.mean((pred_train_norm - y_norm) ** 2)
    train_mse = train_mse_norm * y_std ** 2

    best_mse = train_mse
    best_type = "basis"
    best_gamma = None

    # RBF fallback
    if train_mse_norm > 0.05 and _time_left() > 15.0:
        X_scale = np.std(X_train, axis=0)
        X_scale[X_scale < 1e-12] = 1.0
        d_eff = np.sqrt(np.sum(X_scale ** 2))
        gamma_candidates = [1.0 / (2.0 * d * d_eff ** 2),
                            1.0 / (d_eff ** 2),
                            1.0 / (d_eff ** 2) * 5.0,
                            1.0 / (d_eff ** 2) * 0.2]
        for gamma in gamma_candidates:
            if _time_left() < 5.0:
                break
            try:
                pred_rbf = _rbf_predict(X_train, y_train, X_train, gamma, alpha=1e-6)
                mse_rbf = np.mean((pred_rbf - y_train) ** 2)
                if mse_rbf < best_mse:
                    best_mse = mse_rbf
                    best_type = "rbf"
                    best_gamma = gamma
            except Exception:
                continue

    # Capture
    sel_indices = active
    F_mean_c = F_mean.copy()
    F_std_c = F_std.copy()
    w_c = w_active.copy()
    y_mean_c = y_mean
    y_std_c = y_std
    best_type_c = best_type
    best_gamma_c = best_gamma
    X_train_c = X_train.copy()
    y_train_c = y_train.copy()
    finite_mask_c = finite_mask.copy()

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if best_type_c == "rbf" and best_gamma_c is not None:
            return _rbf_predict(X_train_c, y_train_c, X, best_gamma_c, alpha=1e-6)
        else:
            F_pred = _build_basis(X, d)
            F_pred = F_pred[:, finite_mask_c]
            F_pred_norm = (F_pred - F_mean_c) / F_std_c
            F_pred_sel = F_pred_norm[:, sel_indices]
            return F_pred_sel @ w_c * y_std_c + y_mean_c

    return predict
# EVOLVE-BLOCK-END
