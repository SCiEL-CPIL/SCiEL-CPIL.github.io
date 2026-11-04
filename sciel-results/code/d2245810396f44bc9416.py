"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # safety margin under 150s wall-clock


def _validate(x, w, h):
    """Check that all circles are valid: inside rectangle, non-overlapping."""
    n = len(x)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = x[i]
        if r <= 1e-12:
            return False
        if cx - r < -1e-9 or cx + r > w + 1e-9:
            return False
        if cy - r < -1e-9 or cy + r > h + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2
            if d2 < (x[i, 2] + x[j, 2]) ** 2 - 1e-9:
                return False
    return True


def _total_radius(x):
    return float(np.sum(x[:, 2]))


def _make_grid(n, w, h):
    """Create a grid of initial circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.85
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _relax_circles(x, w, h, max_iter=300, dt=0.01):
    """Iterative relaxation: push circles apart and grow radii."""
    n = len(x)
    x = x.copy()
    for it in range(max_iter):
        # Compute overlap forces
        overlap = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                need = x[i, 2] + x[j, 2]
                if d < need and d > 1e-12:
                    ov = (need - d) * 0.5
                    overlap[i] += ov
                    overlap[j] += ov
        # Update positions: push apart
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                need = x[i, 2] + x[j, 2]
                if d < need and d > 1e-12:
                    push = (need - d) * 0.3
                    x[i, 0] += push * dx / d
                    x[i, 1] += push * dy / d
                    x[j, 0] -= push * dx / d
                    x[j, 1] -= push * dy / d
        # Clamp to walls
        for i in range(n):
            x[i, 0] = max(x[i, 2], min(x[i, 0], w - x[i, 2]))
            x[i, 1] = max(x[i, 2], min(x[i, 1], h - x[i, 2]))
        # Grow radii toward wall limit
        for i in range(n):
            wall_r = min(x[i, 0], w - x[i, 0], x[i, 1], h - x[i, 1])
            x[i, 2] = min(x[i, 2] + dt * (wall_r - x[i, 2]) * 0.5, wall_r)
    return x


def _slsqp_optimize(x0, w, h, t0, maxiter=500):
    """SLSQP optimization with explicit constraints."""
    n = len(x0)
    x0 = x0.copy().flatten()

    def objective(v):
        x = v.reshape(n, 3)
        return -np.sum(x[:, 2])

    def objective_grad(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0  # d(-sum r)/dr = -1
        return g

    # Inequality constraints: g(v) >= 0
    # Walls: r - (cx) <= 0 => cx - r >= 0
    #        r - (w - cx) <= 0 => w - cx - r >= 0
    #        r - (cy) <= 0 => cy - r >= 0
    #        r - (h - cy) <= 0 => h - cy - r >= 0
    # Pairs: d_ij - (r_i + r_j) >= 0

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
                            'jac': lambda v, i=i: _wall_jac(v, i, 0, w, h)})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: w - v[3*i] - v[3*i+2],
                            'jac': lambda v, i=i: _wall_jac(v, i, 1, w, h)})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
                            'jac': lambda v, i=i: _wall_jac(v, i, 2, w, h)})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: h - v[3*i+1] - v[3*i+2],
                            'jac': lambda v, i=i: _wall_jac(v, i, 3, w, h)})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq',
                                'fun': lambda v, i=i, j=j: _pair_dist(v, i, j) - v[3*i+2] - v[3*j+2],
                                'jac': lambda v, i=i, j=j: _pair_jac(v, i, j)})

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w / 2))   # cx
        bounds.append((1e-6, h / 2))   # cy
        bounds.append((1e-6, min(w, h) / 2))  # r

    try:
        res = minimize(objective, x0, jac=objective_grad, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-8})
        if res.success or (res.x is not None and np.isfinite(res.fun)):
            x = res.x.reshape(n, 3)
            if _validate(x, w, h):
                return x
    except Exception:
        pass
    return x0.reshape(n, 3)


def _wall_jac(v, i, wall_idx, w, h):
    g = np.zeros(len(v))
    if wall_idx == 0:  # cx - r >= 0
        g[3*i] = 1.0
        g[3*i+2] = -1.0
    elif wall_idx == 1:  # w - cx - r >= 0
        g[3*i] = -1.0
        g[3*i+2] = -1.0
    elif wall_idx == 2:  # cy - r >= 0
        g[3*i+1] = 1.0
        g[3*i+2] = -1.0
    elif wall_idx == 3:  # h - cy - r >= 0
        g[3*i+1] = -1.0
        g[3*i+2] = -1.0
    return g


def _pair_dist(v, i, j):
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    return np.sqrt(dx*dx + dy*dy + 1e-24)


def _pair_jac(v, i, j):
    g = np.zeros(len(v))
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    d = np.sqrt(dx*dx + dy*dy + 1e-24)
    g[3*i] = dx / d
    g[3*i+1] = dy / d
    g[3*j] = -dx / d
    g[3*j+1] = -dy / d
    g[3*i+2] = -1.0
    g[3*j+2] = -1.0
    return g


def _polish_slsqp(x, w, h, t0, maxiter=200):
    """Final polish with tighter tolerance."""
    n = len(x)
    x0 = x.copy().flatten()

    def objective(v):
        return -np.sum(v[2::3])

    def objective_grad(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
                            'jac': lambda v, i=i: _wall_jac(v, i, 0, w, h)})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: w - v[3*i] - v[3*i+2],
                            'jac': lambda v, i=i: _wall_jac(v, i, 1, w, h)})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
                            'jac': lambda v, i=i: _wall_jac(v, i, 2, w, h)})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: h - v[3*i+1] - v[3*i+2],
                            'jac': lambda v, i=i: _wall_jac(v, i, 3, w, h)})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq',
                                'fun': lambda v, i=i, j=j: _pair_dist(v, i, j) - v[3*i+2] - v[3*j+2],
                                'jac': lambda v, i=i, j=j: _pair_jac(v, i, j)})

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w / 2))
        bounds.append((1e-6, h / 2))
        bounds.append((1e-6, min(w, h) / 2))

    try:
        res = minimize(objective, x0, jac=objective_grad, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-10})
        if res.x is not None:
            x_new = res.x.reshape(n, 3)
            if _validate(x_new, w, h) and _total_radius(x_new) >= _total_radius(x) - 1e-9:
                return x_new
    except Exception:
        pass
    return x


def _solve_aspect(w, h, t0):
    """Solve packing for a given aspect ratio."""
    x = _make_grid(N, w, h)
    x = _relax_circles(x, w, h, max_iter=200, dt=0.02)

    # First SLSQP pass
    if time.time() - t0 < DEADLINE:
        x = _slsqp_optimize(x, w, h, t0, maxiter=400)

    # Polish
    if time.time() - t0 < DEADLINE:
        x = _polish_slsqp(x, w, h, t0, maxiter=200)

    return x


def construct():
    t0 = time.time()
    best = None
    best_score = -np.inf

    # Try a range of aspect ratios
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 0.75, 1.33, 0.65, 1.55]
    for ar in ratios:
        if time.time() - t0 > DEADLINE:
            break
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        # Also try swapped orientation
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() - t0 > DEADLINE:
                break
            try:
                x = _solve_aspect(ww, hh, t0)
            except Exception:
                continue
            s = _total_radius(x)
            if _validate(x, ww, hh) and s > best_score:
                best_score = s
                best = (x.copy(), ww, hh)

    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, w, h = best
    # Final safety check
    if not _validate(x, w, h):
        for i in range(N):
            x[i, 2] *= 0.999
        if not _validate(x, w, h):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    return x
