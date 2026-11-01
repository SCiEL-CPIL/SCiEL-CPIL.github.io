"""Circle packing in perimeter-4 rectangle: L-BFGS-B smooth penalty + radial expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _smooth_objective(z, w, h, beta=50.0):
    """
    Smooth objective: -sum(r) + penalty for violations.
    z = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Uses smooth max(0, v) approx via softplus to keep gradient well-behaved.
    """
    n = N
    x = z[0:n]
    y = z[n:2*n]
    r = z[2*n:3*n]

    # Base objective: maximize sum of radii
    obj = -np.sum(r)

    # Softplus: sp(v) = log(1 + exp(beta*v)) / beta  ~ max(0, v)
    # But for large beta this overflows; use stable form
    def softplus(v):
        # stable softplus
        return np.where(v > 20/beta, v, np.log1p(np.exp(beta * v)) / beta)

    # Wall constraints: x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
    wall_viol = (
        softplus(r - x) + softplus(x + r - w) +
        softplus(r - y) + softplus(y + r - h)
    )
    obj += 100.0 * np.sum(wall_viol)

    # Pairwise overlap: d_ij >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    d = np.sqrt(np.maximum(d_sq, 1e-12))
    r_sum = r[:, None] + r[None, :]
    overlap = softplus(r_sum - d)
    # Only count upper triangle
    iu = np.triu_indices(n, k=1)
    obj += 100.0 * np.sum(overlap[iu])

    return obj


def _grid_init(w, h, n_circles):
    """Generate grid-based initial positions and radii."""
    cols = int(np.ceil(np.sqrt(n_circles * w / h)))
    rows = int(np.ceil(n_circles / cols))
    cw = w / cols
    ch = h / rows
    r_cell = min(cw, ch) / 2.0 - 1e-6
    x = np.zeros(n_circles)
    y = np.zeros(n_circles)
    r = np.zeros(n_circles)
    for i in range(n_circles):
        row, col = divmod(i, cols)
        x[i] = (col + 0.5) * cw
        y[i] = (row + 0.5) * ch
        r[i] = r_cell
    return x, y, r


def _lbfgs_optimize(w, h, x0, y0, r0, max_time=8.0, start_time=None, beta=50.0):
    """
    Run L-BFGS-B on the smooth objective.
    Returns (x, y, r, value) or None if failed.
    """
    if start_time is None:
        start_time = time.time()

    n = N
    z0 = np.concatenate([x0, y0, r0])

    r_max = min(w, h) / 2.0 - 1e-6
    bounds = [(0, w)] * n + [(0, h)] * n + [(0, r_max)] * n

    def obj(z):
        return _smooth_objective(z, w, h, beta=beta)

    try:
        res = minimize(
            obj, z0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        z_opt = res.x
        x_opt = z_opt[0:n]
        y_opt = z_opt[n:2*n]
        r_opt = z_opt[2*n:3*n]

        # Clip to valid ranges
        x_opt = np.clip(x_opt, 0, w)
        y_opt = np.clip(y_opt, 0, h)
        r_opt = np.clip(r_opt, 0, r_max)

        # Check validity; if invalid, shrink radii slightly
        if not _is_valid(x_opt, y_opt, r_opt, w, h):
            # Try shrinking
            for scale in [0.99, 0.98, 0.95, 0.9]:
                r_try = r_opt * scale
                if _is_valid(x_opt, y_opt, r_try, w, h):
                    r_opt = r_try
                    break
            else:
                return None

        val = float(np.sum(r_opt))
        return x_opt, y_opt, r_opt, val

    except Exception:
        return None


def _radial_expansion(x, y, r, w, h, max_time=5.0, start_time=None):
    """
    Try to uniformly expand all radii by a small factor, then do
    a final L-BFGS polish. This exploits the fact that after optimization,
    there may be slack in the packing.
    """
    if start_time is None:
        start_time = time.time()

    n = N
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = float(np.sum(r))

    # Try uniform expansion
    for scale in np.linspace(1.001, 1.05, 10):
        if time.time() - start_time > max_time:
            break
        r_try = r * scale
        # Check if valid
        if _is_valid(x, y, r_try, w, h):
            best_r = r_try
            best_val = float(np.sum(r_try))
        else:
            # Try L-BFGS to find a valid expanded configuration
            z0 = np.concatenate([x, y, r_try])
            result = _lbfgs_optimize(w, h, x, y, r_try, max_time=3.0,
                                       start_time=start_time, beta=80.0)
            if result is not None:
                x_r, y_r, r_r, val_r = result
                if val_r > best_val:
                    best_x, best_y, best_r, best_val = x_r, y_r, r_r, val_r

    return best_x, best_y, best_r, best_val


def _random_init(w, h, rng, n_circles):
    """Random initialization with non-overlapping circles."""
    x = np.zeros(n_circles)
    y = np.zeros(n_circles)
    r = np.zeros(n_circles)

    # Start with small circles and place them
    r_base = min(w, h) / (2 * n_circles) * 0.5
    for i in range(n_circles):
        # Random position
        x[i] = rng.uniform(r_base, w - r_base)
        y[i] = rng.uniform(r_base, h - r_base)
        r[i] = r_base

    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    best_value = 0.0
    best_result = None

    def check_and_update(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)

    rng = np.random.default_rng(42)

    # Scan aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h = 2
    aspect_ratios = np.linspace(0.5, 1.5, 11)  # w from 0.5 to 1.5

    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break

        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Multiple restarts per aspect ratio
        n_restarts = 4
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break

            if restart == 0:
                # Grid init
                x, y, r = _grid_init(w0, h0, N)
            else:
                # Random init
                x, y, r = _random_init(w0, h0, rng, N)

            # L-BFGS optimization
            result = _lbfgs_optimize(w0, h0, x, y, r, max_time=6.0,
                                      start_time=start_time, beta=50.0)
            if result is not None:
                x_opt, y_opt, r_opt, val_opt = result
                check_and_update(x_opt, y_opt, r_opt, w0, h0)

                # Radial expansion polish
                if time.time() > deadline - 10:
                    break
                x_exp, y_exp, r_exp, val_exp = _radial_expansion(
                    x_opt, y_opt, r_opt, w0, h0, max_time=4.0,
                    start_time=start_time)
                check_and_update(x_exp, y_exp, r_exp, w0, h0)

    # Focused search on best aspect ratio with more restarts
    if best_result is not None and time.time() < deadline - 25:
        x_best, y_best, r_best, w_best, h_best = best_result

        for restart in range(6):
            if time.time() > deadline - 12:
                break

            # Perturb the best solution
            x_p = x_best + rng.normal(0, 0.02, N)
            y_p = y_best + rng.normal(0, 0.02, N)
            r_p = r_best * (1 + rng.normal(0, 0.03, N))
            r_p = np.clip(r_p, 0.01, min(w_best, h_best)/2 - 1e-6)
            x_p = np.clip(x_p, 0, w_best)
            y_p = np.clip(y_p, 0, h_best)

            result = _lbfgs_optimize(w_best, h_best, x_p, y_p, r_p,
                                      max_time=5.0, start_time=start_time, beta=80.0)
            if result is not None:
                x_opt, y_opt, r_opt, val_opt = result
                check_and_update(x_opt, y_opt, r_opt, w_best, h_best)

                # Expansion polish
                if time.time() > deadline - 8:
                    break
                x_exp, y_exp, r_exp, val_exp = _radial_expansion(
                    x_opt, y_opt, r_opt, w_best, h_best, max_time=3.0,
                    start_time=start_time)
                check_and_update(x_exp, y_exp, r_exp, w_best, h_best)

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
