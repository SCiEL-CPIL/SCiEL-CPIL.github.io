"""Circle packing in unit square - smooth surrogate + polish."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0  # safety margin before 150s kill

def _objective(params):
    """Negative sum of radii (minimize). params: (n,3) -> (x,y,r)."""
    x, y, r = params[:, 0], params[:, 1], params[:, 2]
    n = len(r)
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    # Required separation: r_i + r_j
    req = r[:, None] + r[None, :]
    # Soft violation of circle-circle: soft-min of (dist - req)
    # We want dist >= req. Penalty for dist < req.
    # Use smooth max(0, req - dist) via log-sum-exp-ish
    diff = req - dist  # positive means overlap
    # soft penalty
    m = np.max(diff, axis=1)
    # sum over j of softmax(0, diff)
    # simpler: sum of softplus-like: log(1+exp(k*diff))/k
    k = 50.0
    # avoid overflow
    safe = np.clip(diff, -50, 50)
    pen_cc = np.sum(np.logaddexp(0, k * safe), axis=1) / k

    # Wall constraints: x in [r, 1-r], y in [r, 1-r]
    # violation: max(0, r - x), max(0, x - (1-r)), same for y
    viol = np.maximum(0, r - x) + np.maximum(0, x - (1 - r)) + \
           np.maximum(0, r - y) + np.maximum(0, y - (1 - r))
    pen_wall = np.sum(np.logaddexp(0, k * np.clip(viol, -50, 50))) / k

    # radius positivity
    pen_r = np.sum(np.logaddexp(0, -k * np.clip(r, -50, 50))) / k

    total_pen = np.sum(pen_cc) + np.sum(pen_wall) + np.sum(pen_r)
    return -np.sum(r) + 100.0 * total_pen


def _objective_grad(params):
    """Gradient of objective via finite differences on small delta (slow)."""
    # Use analytic where possible; for speed use central diff on r only is hard.
    # We'll rely on a cheap gradient: treat penalties as approximated.
    # For robustness, use numerical gradient with modest step.
    eps = 1e-5
    n = params.shape[0]
    g = np.zeros_like(params)
    base = _objective(params)
    for i in range(n):
        for d in range(3):
            p = params.copy()
            p[i, d] += eps
            g[i, d] = (_objective(p) - base) / eps
    return g


def _is_valid(params, tol=1e-7):
    x, y, r = params[:, 0], params[:, 1], params[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    req = r[:, None] + r[None, :]
    if np.any(dist < req - tol):
        return False
    return True


def _shrink_to_valid(params, tol=1e-9):
    """Gently shrink radii (proportional) until valid."""
    params = params.copy()
    r0 = params[:, 2].copy()
    for it in range(200):
        if _is_valid(params, tol=1e-7):
            return params
        # scale radii down a bit
        params[:, 2] *= 0.98
        # also pull centers inward to satisfy walls
        r = params[:, 2]
        x = np.clip(params[:, 0], r, 1 - r)
        y = np.clip(params[:, 1], r, 1 - r)
        params[:, 0] = x
        params[:, 1] = y
    # last resort: very small
    params[:, 2] *= 0.5
    r = params[:, 2]
    params[:, 0] = np.clip(params[:, 0], r, 1 - r)
    params[:, 1] = np.clip(params[:, 1], r, 1 - r)
    return params


def _grid_init(n):
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:n], dtype=float)


def _random_init(n, seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = np.full(n, 0.05)
    return np.column_stack([x, y, r])


def _l_bfgs(params, max_iter=2000, tol=1e-10):
    """Simple L-BFGS using scipy if available, else hand-rolled BFGS."""
    try:
        from scipy.optimize import minimize
        res = minimize(_objective, params, jac=_objective_grad,
                       method='L-BFGS-B',
                       bounds=[(0, 1)] * (3 * params.shape[0]),
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        return res.x
    except Exception:
        # fallback: gradient descent with momentum
        p = params.copy()
        lr = 0.01
        mom = 0.9
        v = np.zeros_like(p)
        for it in range(max_iter):
            g = _objective_grad(p)
            v = mom * v - lr * g
            p = p + v
            # clip
            p[:, 0] = np.clip(p[:, 0], 0, 1)
            p[:, 1] = np.clip(p[:, 1], 0, 1)
            p[:, 2] = np.clip(p[:, 2], 0, 0.5)
        return p


def _polish(params, max_iter=500, tol=1e-12):
    """High-precision polish: alternate projection + gradient step."""
    p = params.copy()
    for it in range(max_iter):
        # gradient step
        g = _objective_grad(p)
        # adaptive step
        step = 1e-3 / (1 + it * 0.01)
        p = p - step * g
        # project walls
        r = np.clip(p[:, 2], 0, 0.5)
        p[:, 2] = r
        p[:, 0] = np.clip(p[:, 0], r, 1 - r)
        p[:, 1] = np.clip(p[:, 1], r, 1 - r)
        # if valid and small gradient, stop
        if _is_valid(p, tol=1e-9) and np.linalg.norm(g) < tol * 1e3:
            break
    return p


def construct():
    t0 = time.time()
    best = _grid_init(N)
    best_val = -np.sum(best[:, 2])

    def elapsed():
        return time.time() - t0

    def time_left():
        return _DEADLINE - elapsed()

    def consider(p):
        nonlocal best, best_val
        if _is_valid(p, tol=1e-8):
            val = -np.sum(p[:, 2])
            if val < best_val:
                best_val = val
                best = p.copy()

    # Phase 1: L-BFGS from grid
    if time_left() > 5:
        p = _l_bfgs(_grid_init(N), max_iter=3000)
        consider(p)

    # Phase 2: multiple random restarts
    for seed in range(1, 40):
        if time_left() < 10:
            break
        p0 = _random_init(N, seed)
        p = _l_bfgs(p0, max_iter=1500)
        consider(p)
        if time_left() < 10:
            break

    # Phase 3: polish best
    if time_left() > 5:
        p = _polish(best, max_iter=800)
        consider(p)
        # second polish
        if time_left() > 5:
            p = _polish(best, max_iter=800)
            consider(p)

    # Final: ensure validity with gentle shrink
    out = _shrink_to_valid(best, tol=1e-9)
    # verify and if invalid, fall back to grid
    if not _is_valid(out, tol=1e-8):
        out = _grid_init(N)
    return out
# EVOLVE-BLOCK-END
