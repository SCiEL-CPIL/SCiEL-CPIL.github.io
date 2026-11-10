import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 135.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.array([np.mean(y) if n > 0 else 0.0, 0, 0, 0, 0, 0], dtype=np.float64)

    # Standardize x for better conditioning
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    xc = (x - mu) / sigma

    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15

    best_score = np.inf
    best_coeffs_orig = np.zeros(6, dtype=np.float64)
    best_deg = 1
    best_alpha = 0.0

    # Build Vandermonde matrices for each degree once
    def _build_vandermonde(xc, degree):
        return np.column_stack([xc**k for k in range(degree + 1)])

    # Grid search over degree and regularization
    degrees = [1, 2, 3, 4, 5]
    alphas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # Precompute Vandermonde matrices
    vandermondes = {}
    for deg in degrees:
        vandermondes[deg] = _build_vandermonde(xc, deg)

    # For each (deg, alpha), do a quick 5-fold CV to estimate generalization error
    def _cv_mse_fast(xc, y, V, degree, alpha):
        n_samples = len(xc)
        if n_samples < (degree + 2) * 5:
            # Too few samples for 5-fold; use a single split
            n_val = min(500, n_samples // 5)
            n_val = max(n_val, degree + 2)
            if n_val >= n_samples:
                return np.var(y)
            idx = np.random.RandomState(42).permutation(n_samples)
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
            x_train = xc[train_idx]
            y_train = y[train_idx]
            x_val = xc[val_idx]
            y_val = y[val_idx]
            V_train = np.column_stack([x_train**k for k in range(degree + 1)])
            V_val = np.column_stack([x_val**k for k in range(degree + 1)])
            if alpha > 0:
                A = V_train.T @ V_train
                b = V_train.T @ y_train
                reg = alpha * np.eye(degree + 1)
                try:
                    c = np.linalg.solve(A + reg, b)
                except np.linalg.LinAlgError:
                    c = np.zeros(degree + 1)
            else:
                try:
                    c = np.linalg.lstsq(V_train, y_train, rcond=None)[0]
                except Exception:
                    c = np.zeros(degree + 1)
            pred = V_val @ c
            return np.mean((pred - y_val)**2)
        else:
            K = 5
            idx = np.random.RandomState(42).permutation(n_samples)
            fold_size = n_samples // K
            mse_sum = 0.0
            for k in range(K):
                lo = k * fold_size
                hi = lo + fold_size
                if k == K - 1:
                    hi = n_samples
                val_idx = idx[lo:hi]
                train_idx = np.concatenate([idx[:lo], idx[hi:]])
                x_train = xc[train_idx]
                y_train = y[train_idx]
                x_val = xc[val_idx]
                y_val = y[val_idx]
                V_train = np.column_stack([x_train**k2 for k2 in range(degree + 1)])
                V_val = np.column_stack([x_val**k2 for k2 in range(degree + 1)])
                if alpha > 0:
                    A = V_train.T @ V_train
                    b = V_train.T @ y_train
                    reg = alpha * np.eye(degree + 1)
                    try:
                        c = np.linalg.solve(A + reg, b)
                    except np.linalg.LinAlgError:
                        c = np.zeros(degree + 1)
                else:
                    try:
                        c = np.linalg.lstsq(V_train, y_train, rcond=None)[0]
                    except Exception:
                        c = np.zeros(degree + 1)
                pred = V_val @ c
                mse_sum += np.mean((pred - y_val)**2)
            return mse_sum / K

    # First pass: coarse grid
    for deg in degrees:
        if time.time() > deadline:
            break
        V = vandermondes[deg]
        for alpha in alphas:
            if time.time() > deadline:
                break
            try:
                cv_mse = _cv_mse_fast(xc, y, V, deg, alpha)
                score = cv_mse / y_var
                if score < best_score:
                    best_score = score
                    best_deg = deg
                    best_alpha = alpha
            except Exception:
                continue

    # Second pass: refine around the best degree with finer alphas
    if time.time() < deadline - 10.0:
        refined_alphas = np.logspace(-6, 3, 20)
        # Also try neighboring degrees
        for deg in [max(1, best_deg - 1), best_deg, min(5, best_deg + 1)]:
            if time.time() > deadline:
                break
            V = vandermondes[deg]
            for alpha in refined_alphas:
                if time.time() > deadline:
                    break
                try:
                    cv_mse = _cv_mse_fast(xc, y, V, deg, float(alpha))
                    score = cv_mse / y_var
                    if score < best_score:
                        best_score = score
                        best_deg = deg
                        best_alpha = float(alpha)
                except Exception:
                    continue

    # Final fit on all data with best settings
    V = vandermondes[best_deg]
    if best_alpha > 0:
        A = V.T @ V
        b = V.T @ y
        reg = best_alpha * np.eye(best_deg + 1)
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            c_centered = np.linalg.lstsq(V, y, rcond=None)[0]
    else:
        try:
            c_centered = np.linalg.lstsq(V, y, rcond=None)[0]
        except Exception:
            c_centered = np.zeros(best_deg + 1)

    # Convert from centered-standardized basis to original x basis
    # p(x) = sum_i c_i * ((x - mu)/sigma)^i
    # Coefficient of x^k: sum_{i=k}^{deg} c_i * C(i,k) * (-mu)^(i-k) / sigma^i
    coeffs_orig = np.zeros(6, dtype=np.float64)
    for i in range(best_deg + 1):
        ci = c_centered[i]
        inv_si = (1.0 / sigma) ** i
        for k in range(i + 1):
            # Compute binomial coefficient C(i, k)
            if k == 0 or k == i:
                Ck = 1.0
            else:
                mm = min(k, i - k)
                num = 1
                den = 1
                for t in range(1, mm + 1):
                    num *= (i - mm + t)
                    den *= t
                Ck = num / den
            coeffs_orig[k] += ci * Ck * ((-mu) ** (i - k)) * inv_si

    # Ensure finite
    if not np.all(np.isfinite(coeffs_orig)):
        coeffs_orig = np.zeros(6, dtype=np.float64)
        coeffs_orig[0] = np.mean(y)

    return coeffs_orig
