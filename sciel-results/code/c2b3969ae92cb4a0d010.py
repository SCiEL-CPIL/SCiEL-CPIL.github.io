"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_T0 = time.time()
_DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150s)


def _softmax_penalty(v, beta):
    """Smooth max of v via log-sum-exp (penalty for constraint violations)."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return 0.0
    m = v.max()
    if not np.isfinite(m):
        return 1e12
    s = np.exp((v - m) * beta)
    return m + np.log(s.sum()) / beta


def _make_objective(beta, gamma):
    def obj(x):
        x = np.asarray(x, dtype=float)
        W = x[0]; H = x[1]
        cx = x[2:2 + N]; cy = x[2 + N:2 + 2 * N]; r = x[2 + 2 * N:2 + 3 * N]
        # perimeter constraint: 2*(W+H) <= 4
        pen = []
        pen.append(2.0 * (W + H) - 4.0)
        # radii positive
        pen.append(-r)
        # circles inside rectangle
        pen.append(cx - r)
        pen.append(cy - r)
        pen.append(cx - W + r)
        pen.append(cy - H + r)
        # pair separation: dist >= r_i + r_j
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]; dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                pen.append(r[i] + r[j] - d)
        pen = np.concatenate(pen)
        penalty = gamma * _softmax_penalty(pen, beta)
        return -np.sum(r) + penalty
    return obj


def _feasible(x):
    x = np.asarray(x, dtype=float)
    W = x[0]; H = x[1]
    cx = x[2:2 + N]; cy = x[2 + N:2 + 2 * N]; r = x[2 + 2 * N:2 + 3 * N]
    if W <= 0 or H <= 0:
        return False
    if 2.0 * (W + H) > 4.0 + 1e-7:
        return False
    if np.any(r <= 0):
        return False
    if np.any(cx - r < -1e-7) or np.any(cy - r < -1e-7):
        return False
    if np.any(cx - W + r > 1e-7) or np.any(cy - H + r > 1e-7):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d < r[i] + r[j] - 1e-7:
                return False
    return True


def _radius_sum(x):
    return float(np.sum(x[2 + 2 * N:2 + 3 * N]))


def _grid_seed(side):
    """Grid of up to `side*side` cells in a 2x2 rectangle (perimeter 4)."""
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * 2.0 / side
            cy = (j + 0.5) * 2.0 / side
            r = 1.0 / side - 1e-4
            cells.append((cx, cy, r))
    return cells


def _random_seed():
    W = 2.0; H = 2.0
    cx = np.random.uniform(0.1, W - 0.1, N)
    cy = np.random.uniform(0.1, H - 0.1, N)
    r = np.random.uniform(0.02, 0.15, N)
    return np.concatenate([[W, H], cx, cy, r])


def _perturb(x, scale):
    x = np.array(x, dtype=float)
    # perturb only centers and radii
    x[2:2 + N] += np.random.uniform(-scale, scale, N)
    x[2 + N:2 + 2 * N] += np.random.uniform(-scale, scale, N)
    x[2 + 2 * N:2 + 3 * N] *= (1.0 + np.random.uniform(-scale, scale, N))
    return x


def construct():
    t_start = time.time()
    best_x = None
    best_val = -np.inf

    def elapsed():
        return time.time() - t_start

    def try_optimize(x0, beta, gamma, maxiter, tol):
        nonlocal best_x, best_val
        if elapsed() > _DEADLINE - 10:
            return
        obj = _make_objective(beta, gamma)
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol * 0.1})
        if elapsed() > _DEADLINE - 10:
            return
        x = res.x
        if _feasible(x):
            v = _radius_sum(x)
            if v > best_val:
                best_val = v
                best_x = x.copy()

    # Initial grid-based feasible start (square 2x2)
    side = 5
    cells = _grid_seed(side)[:N]
    x_grid = np.concatenate([[2.0, 2.0],
                             [c[0] for c in cells],
                             [c[1] for c in cells],
                             [c[2] for c in cells]])
    try_optimize(x_grid, beta=20.0, gamma=50.0, maxiter=300, tol=1e-6)

    # Random restarts
    n_restarts = 40
    for k in range(n_restarts):
        if elapsed() > _DEADLINE - 30:
            break
        # mix: random and perturbed-best
        if k % 3 == 0 and best_x is not None:
            x0 = _perturb(best_x, 0.05)
        else:
            x0 = _random_seed()
        beta = 20.0 + 5.0 * (k % 4)
        gamma = 50.0 + 20.0 * (k % 3)
        try_optimize(x0, beta, gamma, maxiter=200, tol=1e-5)

    # Polish passes on best with tighter tolerance
    if best_x is not None and elapsed() < _DEADLINE - 15:
        for beta, gamma, it, tol in [(40.0, 100.0, 400, 1e-8),
                                     (80.0, 200.0, 400, 1e-9)]:
            if elapsed() > _DEADLINE - 10:
                break
            try_optimize(best_x, beta, gamma, it, tol)

    if best_x is None:
        # fallback: return grid (should be feasible)
        best_x = x_grid

    # Build output: (cx, cy, r)
    W = best_x[0]; H = best_x[1]
    cx = best_x[2:2 + N]; cy = best_x[2 + N:2 + 2 * N]; r = best_x[2 + 2 * N:2 + 3 * N]
    out = np.column_stack([cx, cy, r])
    return out
