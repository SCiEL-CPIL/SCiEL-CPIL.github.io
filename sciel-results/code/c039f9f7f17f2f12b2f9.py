"""Symbolic regression over 10 hidden equations: exact sparse stepwise selection on noiseless data."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 295.0  # safety margin vs 330s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _safe_div(a, b, eps=1e-6):
    denom = np.where(np.abs(b) < eps, np.sign(b) * eps + (b == 0) * eps, b)
    return a / denom

def _build_basis(X):
    """Curated, numerically stable feature basis."""
    n, d = X.shape
    cols = [np.ones(n)]

    # Linear
    for j in range(d):
        cols.append(X[:, j])

    # Squares and cubes
    for j in range(d):
        cols.append(X[:, j] ** 2)
    for j in range(d):
        cols.append(X[:, j] ** 3)

    # Pairwise products, sums, differences
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] + X[:, j])
            cols.append(X[:, i] - X[:, j])

    # Pairwise squares of products (x_i^2 * x_j)
    for i in range(d):
        for j in range(d):
            if i != j:
                cols.append(X[:, i] ** 2 * X[:, j])

    # Trigonometric of individual inputs
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))

    # Trigonometric of pairwise sums/differences
    for i in range(d):
        for j in range(i + 1, d):
            s = X[:, i] + X[:, j]
            df = X[:, i] - X[:, j]
            cols.append(np.sin(s))
            cols.append(np.cos(s))
            cols.append(np.sin(df))
            cols.append(np.cos(df))

    # Square roots (inputs are positive in [1, ...])
    for j in range(d):
        cols.append(np.sqrt(np.abs(X[:, j])))

    # Exponential (inputs bounded, safe)
    for j in range(d):
        cols.append(np.exp(np.clip(X[:, j], -20, 20)))

    # Log (inputs positive)
    for j in range(d):
        cols.append(np.log(np.abs(X[:, j]) + 1e-8))

    # Rational (safe)
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(_safe_div(X[:, i], X[:, j]))

    # Product of all inputs
    if d >= 1:
        p = np.ones(n)
        for j in range(d):
            p = p * X[:, j]
        cols.append(p)

    # Sum of all inputs and its transforms
    s = np.sum(X, axis=1)
    cols.append(s)
    cols.append(s ** 2)
    cols.append(np.sin(s))
    cols.append(np.cos(s))

    # Norm
    norm = np.sqrt(np.sum(X ** 2, axis=1))
    cols.append(norm)
    cols.append(np.sin(norm))

    A = np.column_stack(cols)
    # Remove non-finite columns
    mask = np.isfinite(A)
    A = np.where(mask, A, 0.0)
    col_valid = mask.all(axis=0)
    A = A[:, col_valid]

    # Standardize columns for numerical stability (keep intercept unstandardized)
    # Actually keep raw to preserve exact representability; but center non-intercept for conditioning
    # We'll standardize non-intercept columns; intercept stays 1.
    return A

def _standardize(A):
    """Standardize columns except the first (intercept)."""
    A = A.copy()
    if A.shape[1] > 1:
        mu = A[:, 1:].mean(axis=0)
        sd = A[:, 1:].std(axis=0)
        sd[sd < 1e-12] = 1.0
        A[:, 1:] = (A[:, 1:] - mu) / sd
    return A

def _lstsq_solve(A, y):
    """Least-squares solve with fallback."""
    try:
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None

def _stepwise_select(A, y, max_terms=12, tol=1e-9):
    """Forward stepwise selection. Returns (selected_indices, w_on_selected, residual)."""
    n, p = A.shape
    residual = y.copy()
    ss_res = float(np.sum(y ** 2))
    ss_tot = float(np.sum(y ** 2))
    selected = []
    used = np.zeros(p, dtype=bool)

    # Precompute A^T A and A^T y for fast greedy
    try:
        AtA = A.T @ A
        Aty = A.T @ y
    except Exception:
        return [], np.zeros(0), ss_res

    for step in range(max_terms):
        if _remaining() < 5.0:
            break
        # For each unused column, compute the reduction in residual if added
        # Use the fact that adding column j to current set:
        # new residual = residual - proj of residual on new column orthogonal to current
        # Simpler: just refit on selected+[j] using lstsq on the small subset
        best_j = -1
        best_r = ss_res
        # candidate columns: all unused
        candidates = np.where(~used)[0]
        for j in candidates:
            idx = selected + [j]
            sub = A[:, idx]
            w = _lstsq_solve(sub, y)
            if w is None:
                continue
            pred = sub @ w
            if not np.all(np.isfinite(pred)):
                continue
            r = float(np.sum((y - pred) ** 2))
            if r < best_r:
                best_r = r
                best_j = j
        if best_j < 0:
            break
        selected.append(best_j)
        used[best_j] = True
        ss_res = best_r
        # Noiseless: stop when residual essentially zero
        if ss_res < tol * max(ss_tot, 1.0):
            break
        # Stop if improvement is negligible
        if step > 0 and (ss_res_prev - ss_res) < 1e-12 * max(ss_tot, 1.0):
            break
        ss_res_prev = ss_res

    # Final fit on selected
    if len(selected) == 0:
        return [], np.zeros(0), ss_res
    sub = A[:, selected]
    w = _lstsq_solve(sub, y)
    if w is None:
        # fallback to all
        w = _lstsq_solve(A, y)
        selected = list(range(p))
    pred = sub @ w
    r = float(np.sum((y - pred) ** 2))
    return selected, w, r

def _make_predict(basis_fn, A_train, w, X_train, selected):
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        A = basis_fn(X)
        # Match column count: if basis size differs (shouldn't), fallback
        if A.shape[1] != A_train.shape[1]:
            # Rebuild using same logic; sizes should match for same d
            # Fallback: use all available columns that exist
            p = min(A.shape[1], A_train.shape[1])
            A = A[:, :p]
            w_use = w[:p]
        else:
            A = A
            w_use = w
        if len(selected) > 0 and len(selected) <= A.shape[1]:
            A = A[:, selected]
        return A @ w_use
    return predict

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)

    if _remaining() < 10.0:
        # Emergency fallback: simple linear
        A = np.column_stack([np.ones(len(X)), X])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.column_stack([np.ones(len(X)), X]) @ w
        return predict

    A = _build_basis(X)
    if A.shape[1] == 0:
        w = np.zeros(1)
        def predict(X):
            return np.ones(len(np.asarray(X))) * w[0]
        return predict

    # Standardize for conditioning (intercept preserved)
    A_std = _standardize(A)

    # Try stepwise selection
    selected, w, r = _stepwise_select(A_std, y, max_terms=15)

    # Compute training MSE to sanity check
    if len(selected) > 0:
        A_sel = A_std[:, selected]
        pred = A_sel @ w
        if not np.all(np.isfinite(pred)):
            selected = list(range(A_std.shape[1]))
            w = _lstsq_solve(A_std, y)
            pred = A_std @ w
    else:
        # fallback full
        selected = list(range(A_std.shape[1]))
        w = _lstsq_solve(A_std, y)
        pred = A_std @ w

    # If stepwise gave a bad fit, try a few ridge alphas as backup and pick best
    best_mse = float(np.mean((pred - y) ** 2)) if np.all(np.isfinite(pred)) else float('inf')
    best_sel = selected
    best_w = w

    # Quick ridge backup on standardized full basis
    for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
        if _remaining() < 5.0:
            break
        try:
            p = A_std.shape[1]
            AtA = A_std.T @ A_std
            Aty = A_std.T @ y
            reg = alpha * np.eye(p)
            w_r = np.linalg.solve(AtA + reg, Aty)
            pred_r = A_std @ w_r
            if np.all(np.isfinite(pred_r)):
                mse_r = float(np.mean((pred_r - y) ** 2))
                if mse_r < best_mse:
                    best_mse = mse_r
                    best_sel = list(range(p))
                    best_w = w_r
        except Exception:
            pass

    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        A = _build_basis(X)
        if A.shape[1] != A_std.shape[1]:
            p = min(A.shape[1], A_std.shape[1])
            A = A[:, :p]
            A = _standardize(A)
            w_use = best_w[:p]
        else:
            A = _standardize(A)
            w_use = best_w
        if len(best_sel) > 0 and len(best_sel) <= A.shape[1]:
            A = A[:, best_sel]
        return A @ w_use

    return predict
# EVOLVE-BLOCK-END
