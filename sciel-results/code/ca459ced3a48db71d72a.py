"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _clamp_circles(c):
    """Ensure circles are inside [0,1]^2 by shrinking radii and clamping centers."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    r = np.maximum(r, 1e-6)
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    return np.column_stack([x, y, r])

def _penalty(c, w_overlap=1000.0, w_boundary=1000.0, w_small=1.0):
    """Penalty for invalid packing. Lower is better (0 = valid)."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    n = len(c)
    pen = 0.0
    # Boundary violations
    for i in range(n):
        if r[i] <= 0:
            pen += w_small * (1e-6 - r[i]) ** 2
        for bound in (x[i], y[i], 1 - x[i], 1 - y[i]):
            if bound < r[i]:
                pen += w_boundary * (r[i] - bound) ** 2
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            need = r[i] + r[j]
            if d < need:
                pen += w_overlap * (need - d) ** 2
    return pen

def _objective(c):
    """Maximize sum of radii => minimize -(sum r) + penalty."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    r = c[:, 2]
    return -np.sum(r) + _penalty(c)

def _is_valid(c, tol=1e-6):
    c = np.array(c, dtype=float).reshape(-1, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    n = len(c)
    if np.any(r <= 0):
        return False
    for i in range(n):
        if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol or y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if np.sqrt(dx * dx + dy * dy) < r[i] + r[j] - tol:
                return False
    return True

def _grid_init():
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
    return np.array(cells[:N])

def _hex_init():
    """Hexagonal-ish lattice init."""
    cells = []
    rows = int(np.ceil(np.sqrt(N)))
    for j in range(rows):
        for i in range(rows):
            x = (i + 0.5 + (0.25 if j % 2 else 0)) / (rows + 0.25)
            y = (j + 0.5) / (rows + 0.25)
            cells.append([x, y, 0.5 / (rows + 0.25) - 1e-4])
    return np.array(cells[:N])

def _random_init(rng):
    cells = []
    for _ in range(N):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = rng.uniform(0.02, 0.12)
        cells.append([x, y, r])
    return np.array(cells)

def _local_refine(c, max_iter=200, tol=1e-8):
    """Local L-BFGS-B refinement with penalty."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    bounds = []
    for i in range(len(c)):
        bounds.append((1e-6, 1 - 1e-6))  # x
        bounds.append((1e-6, 1 - 1e-6))  # y
        bounds.append((1e-6, 0.5))       # r
    try:
        res = minimize(_objective, c.ravel(), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        if res.success or res.fun < _objective(c.ravel()):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return c

def _polish_de(c, rng, time_budget=15.0):
    """Local differential evolution polishing around current best."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    n = len(c)
    bounds = []
    for i in range(n):
        cx, cy, cr = c[i]
        bounds.append((max(1e-6, cx - 0.15), min(1 - 1e-6, cx + 0.15)))
        bounds.append((max(1e-6, cy - 0.15), min(1 - 1e-6, cy + 0.15)))
        bounds.append((max(1e-6, cr * 0.5), min(0.5, cr * 1.8)))
    start = time.time()
    def wrapper(x):
        if time.time() - start > time_budget:
            return 1e6
        return _objective(x)
    try:
        res = differential_evolution(wrapper, bounds, seed=int(rng.integers(0, 1e6)),
                                     maxiter=60, popsize=10, tol=1e-5,
                                     mutation=(0.3, 1.0), recombination=0.8,
                                     init='sohal')
        if res.fun < _objective(c.ravel()):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return c

def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(20240517)

    # Candidate initializations
    inits = [
        _grid_init(),
        _hex_init(),
    ]
    for _ in range(6):
        inits.append(_random_init(rng))

    # Stage 1: local refine each init
    for c0 in inits:
        if _remaining() < 10:
            break
        c = _clamp_circles(c0)
        c = _local_refine(c, max_iter=300)
        c = _clamp_circles(c)
        val = -_objective(c)  # approx sum r minus penalty
        if _is_valid(c) and np.sum(c[:, 2]) > best_val:
            best_val = np.sum(c[:, 2])
            best = c.copy()

    # Stage 2: DE polishing on best candidates
    if best is not None and _remaining() > 25:
        c = _polish_de(best, rng, time_budget=min(20.0, max(10.0, _remaining() - 15)))
        c = _clamp_circles(c)
        c = _local_refine(c, max_iter=150)
        c = _clamp_circles(c)
        if _is_valid(c) and np.sum(c[:, 2]) > best_val:
            best_val = np.sum(c[:, 2])
            best = c.copy()

    # Stage 3: a few more random restarts if time permits
    if best is not None and _remaining() > 40:
        for _ in range(8):
            if _remaining() < 15:
                break
            c0 = _random_init(rng)
            c = _clamp_circles(c0)
            c = _local_refine(c, max_iter=200)
            c = _clamp_circles(c)
            if _is_valid(c) and np.sum(c[:, 2]) > best_val:
                best_val = np.sum(c[:, 2])
                best = c.copy()
            if _remaining() > 30:
                c = _polish_de(best, rng, time_budget=min(12.0, max(6.0, _remaining() - 15)))
                c = _clamp_circles(c)
                c = _local_refine(c, max_iter=100)
                c = _clamp_circles(c)
                if _is_valid(c) and np.sum(c[:, 2]) > best_val:
                    best_val = np.sum(c[:, 2])
                    best = c.copy()

    if best is None:
        # Fallback: always-valid grid
        best = _grid_init()

    return _clamp_circles(best)
