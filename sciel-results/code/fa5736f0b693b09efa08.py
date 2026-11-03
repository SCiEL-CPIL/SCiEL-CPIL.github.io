"""Circle packing in perimeter-4 rectangle: maximize sum of radii of 21 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _check_validity(arr):
    """Return (is_valid, perimeter)."""
    if arr is None or len(arr) != N:
        return False, np.inf
    cx, cy, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False, np.inf
    xs = np.minimum(cx - r, 0.0)
    xs = np.maximum(xs, cx + r)
    ys = np.minimum(cy - r, 0.0)
    ys = np.maximum(ys, cy + r)
    W = float(np.max(xs) - np.min(xs))
    H = float(np.max(ys) - np.min(ys))
    if W <= 0 or H <= 0:
        return False, np.inf
    perim = 2.0 * (W + H)
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False, perim
    return perim <= 4.0 + 1e-6, perim


def _make_initial(rows, cols):
    """Create an initial grid of N circles in a rows x cols layout."""
    # Place on a grid, then let optimizer refine
    xs = np.arange(cols)
    ys = np.arange(rows)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()
    # Take first N positions
    idx = np.arange(N)
    gx = gx[idx]
    gy = gy[idx]
    # Normalize into unit-ish space
    cx = gx / max(cols - 1, 1)
    cy = gy / max(rows - 1, 1)
    r = np.full(N, 0.05)
    return np.column_stack([cx, cy, r])


def _objective(x, rows, cols):
    """x = [W, H, cx0, cy0, r0, cx1, cy1, r1, ...]"""
    W = x[0]
    H = x[1]
    cx = x[2:2 + 3 * N:3]
    cy = x[3:2 + 3 * N:3]
    r = x[4:2 + 3 * N:3]

    # Penalties
    pen = 0.0

    # Perimeter constraint: 2(W+H) <= 4  => W+H <= 2
    if W + H > 2.0:
        pen += 1e4 * (W + H - 2.0) ** 2

    # Circles must be inside [0,W]x[0,H]
    for i in range(N):
        if cx[i] < r[i]:
            pen += 1e4 * (r[i] - cx[i]) ** 2
        if cx[i] > W - r[i]:
            pen += 1e4 * (cx[i] - (W - r[i])) ** 2
        if cy[i] < r[i]:
            pen += 1e4 * (r[i] - cy[i]) ** 2
        if cy[i] > H - r[i]:
            pen += 1e4 * (cy[i] - (H - r[i])) ** 2

    # Non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            s = r[i] + r[j]
            if d2 < s * s:
                pen += 1e4 * (s * s - d2)

    # Maximize sum of radii => minimize negative
    return pen - np.sum(r)


def _bounds(rows, cols):
    lo = [0.1, 0.1]
    hi = [2.0, 2.0]
    for i in range(N):
        lo.extend([0.0, 0.0, 1e-6])
        hi.extend([2.0, 2.0, 0.5])
    return list(zip(lo, hi))


def _local_optimize(x0, rows, cols, max_time=10.0):
    """Run L-BFGS-B with a time limit."""
    t0 = time.time()
    best_x = x0.copy()
    best_f = _objective(x0, rows, cols)

    def cb(xk):
        nonlocal best_x, best_f
        f = _objective(xk, rows, cols)
        if f < best_f:
            best_f = f
            best_x = xk.copy()
        if time.time() - t0 > max_time:
            raise RuntimeError("timeout")

    try:
        res = minimize(
            _objective, x0, args=(rows, cols),
            method='L-BFGS-B',
            bounds=_bounds(rows, cols),
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000},
            callback=cb,
        )
        if res.x is not None:
            f = _objective(res.x, rows, cols)
            if f < best_f:
                best_f = f
                best_x = res.x.copy()
    except RuntimeError:
        pass
    return best_x


def _extract(x):
    W = x[0]
    H = x[1]
    cx = x[2:2 + 3 * N:3]
    cy = x[3:2 + 3 * N:3]
    r = x[4:2 + 3 * N:3]
    return W, H, cx, cy, r


def _refine(arr, max_time=5.0):
    """Given a valid arr, try to increase radii slightly via local search on radii only."""
    # Simple: nudge radii up a tiny bit and re-check
    t0 = time.time()
    best = arr.copy()
    for trial in range(200):
        if time.time() - t0 > max_time:
            break
        cand = best.copy()
        cand[:, 2] *= (1.0 + 1e-4 * np.random.rand(N))
        ok, perim = _check_validity(cand)
        if ok and np.sum(cand[:, 2]) > np.sum(best[:, 2]) - 1e-12:
            best = cand
    return best


def construct():
    """Return array of shape (21, 3): 21 disjoint circles, bounding rect perimeter <= 4."""
    t_start = time.time()

    # Try several grid layouts
    layouts = [(2, 11), (3, 7), (4, 6), (5, 5), (1, 21), (2, 10)]

    best_arr = None
    best_sum = -np.inf

    for rows, cols in layouts:
        if time.time() - t_start > _DEADLINE - 15:
            break

        # Build initial x
        x0 = _make_initial(rows, cols)
        # Convert to full x vector
        W0 = 2.0 * cols * 0.05
        H0 = 2.0 * rows * 0.05
        if W0 > 2.0:
            W0 = 2.0
        if H0 > 2.0:
            H0 = 2.0
        x_full = np.array([W0, H0])
        for i in range(N):
            x_full = np.append(x_full, [x0[i, 0], x0[i, 1], x0[i, 2]])

        # Local optimize
        x_opt = _local_optimize(x_full, rows, cols, max_time=8.0)

        # Extract
        W, H, cx, cy, r = _extract(x_opt)

        # Build arr
        arr = np.column_stack([cx, cy, r])

        # Check validity
        ok, perim = _check_validity(arr)
        if ok:
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_arr = arr.copy()

    if best_arr is None:
        # Fallback: single row
        r = 1.0 / (N + 1)
        cx = np.arange(N) * 2 * r + r
        cy = np.full(N, r)
        best_arr = np.column_stack([cx, cy, np.full(N, r)])
        ok, perim = _check_validity(best_arr)
        if not ok:
            # Shrink to fit
            scale = 1.0 / (1.0 + perim / 4.0)
            best_arr[:, 0] *= scale
            best_arr[:, 1] *= scale
            best_arr[:, 2] *= scale

    # Final refinement pass with remaining time
    if time.time() - t_start < _DEADLINE - 5:
        best_arr = _refine(best_arr, max_time=max(1.0, _DEADLINE - 5 - (time.time() - t_start)))

    # Final safety: ensure validity
    ok, perim = _check_validity(best_arr)
    if not ok:
        # Shrink to fit
        if perim > 4.0 and perim > 0:
            scale = 4.0 / perim * 0.999
            best_arr[:, 0] *= scale
            best_arr[:, 1] *= scale
            best_arr[:, 2] *= scale

    return best_arr
