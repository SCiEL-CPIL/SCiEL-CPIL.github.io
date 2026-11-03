"""Circle packing in unit square: joint (x,y,r) optimization with smooth penalties.

Structural change: optimize all 78 variables (x, y, r per circle) jointly using
a smooth penalty formulation. The objective is:
    maximize sum(r_i) - lambda * (sum of constraint violations)^2

where violations include:
  - r_i + dist_to_boundary < 0  (circle outside square)
  - r_i + r_j - dist(i,j) < 0   (circles overlap)

This avoids the non-differentiability of min() in radius derivation and allows
gradient-based methods to smoothly push circles into tight packing configurations.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 140.0  # Leave 10s safety margin
START = time.time()


def _elapsed():
    return time.time() - START


def _penalty_objective(params, lam=10.0):
    """
    Objective to minimize: -sum(r) + lam * (sum of squared violations)
    params: 78-dim vector [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Clip to valid ranges
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    r = np.clip(r, 0.0, 0.5)
    
    # Boundary violations: r_i - dist_to_boundary should be <= 0
    d_left = x
    d_right = 1.0 - x
    d_bottom = y
    d_top = 1.0 - y
    d_bound = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    boundary_viol = np.maximum(0.0, r - d_bound)
    
    # Pairwise overlap violations: r_i + r_j - dist(i,j) should be <= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    overlap_viol = np.maximum(0.0, r[:, None] + r[None, :] - dist)
    
    total_viol = np.sum(boundary_viol ** 2) + np.sum(overlap_viol ** 2)
    
    return -np.sum(r) + lam * total_viol


def _valid(x, y, r, eps=1e-6):
    """Check if configuration is valid."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    if np.any(need + eps > dist):
        return False
    return True


def _fix_to_valid(x, y, r, max_iter=50):
    """
    Iteratively shrink radii until configuration is valid.
    Returns (x, y, r) with reduced radii if needed.
    """
    for _ in range(max_iter):
        if _valid(x, y, r):
            return x, y, r
        
        # Find the worst violation and shrink that radius
        d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        boundary_viol = r - d
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        overlap_viol = r[:, None] + r[None, :] - dist
        
        # For each circle, find max violation
        circle_viol = np.zeros(N)
        for i in range(N):
            bv = max(0.0, boundary_viol[i])
            ov = np.max(np.maximum(0.0, overlap_viol[i, :]))
            circle_viol[i] = max(bv, ov)
        
        if np.max(circle_viol) <= 1e-10:
            break
        
        # Shrink radii proportionally to violations
        shrink = np.maximum(0.0, circle_viol) * 0.5
        r = r - shrink
        r = np.maximum(r, 1e-8)
    
    return x, y, r


def _grid_init_5x5_plus1(rng):
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    # Initialize radii as half the minimum cell size
    cell_size = min(1.0 / cols, 1.0 / rows)
    r = np.full(N, 0.4 * cell_size)
    return x, y, r


def _grid_init_4x7(rng):
    cols, rows = 4, 7
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    cell_size = min(1.0 / cols, 1.0 / rows)
    r = np.full(N, 0.4 * cell_size)
    return x, y, r


def _grid_init_6x5(rng):
    cols, rows = 6, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    cell_size = min(1.0 / cols, 1.0 / rows)
    r = np.full(N, 0.4 * cell_size)
    return x, y, r


def _random_init(rng):
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = rng.uniform(0.05, 0.15, size=N)
    return x, y, r


def _hex_init(rng):
    """Hexagonal lattice initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    spacing = 0.19
    for i in range(N):
        row = i // 6
        col = i % 6
        x[i] = 0.15 + col * spacing + (0.5 * spacing if row % 2 else 0)
        y[i] = 0.15 + row * spacing * 0.866
        if x[i] > 0.9:
            x[i] = 0.9
        if y[i] > 0.9:
            y[i] = 0.9
    r = np.full(N, 0.4 * spacing)
    return x, y, r


def _optimize_all(x, y, r, method='L-BFGS-B', lam=10.0):
    """Optimize all (x, y, r) variables to maximize sum(r) with penalty for violations."""
    params = np.zeros(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-4, 0.5))  # r
    
    if method == 'L-BFGS-B':
        try:
            res = minimize(lambda p: _penalty_objective(p, lam), params, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 3000})
        except Exception:
            return x, y, r
    elif method == 'SLSQP':
        try:
            res = minimize(lambda p: _penalty_objective(p, lam), params, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10})
        except Exception:
            return x, y, r
    else:
        return x, y, r
    
    v = res.x
    xx = np.clip(v[0::3], 0.0, 1.0)
    yy = np.clip(v[1::3], 0.0, 1.0)
    rr = np.clip(v[2::3], 1e-4, 0.5)
    
    return xx, yy, rr


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(42)

    inits = [
        _grid_init_5x5_plus1,
        _grid_init_4x7,
        _grid_init_6x5,
        _random_init,
        _hex_init,
    ]

    for init_func in inits:
        if _elapsed() > DEADLINE - 10.0:
            break
        
        x, y, r = init_func(rng)
        
        for restart in range(4):
            if _elapsed() > DEADLINE - 10.0:
                break
            
            # Add small jitter
            x0 = np.clip(x + rng.normal(0, 0.01, N), 0.01, 0.99)
            y0 = np.clip(y + rng.normal(0, 0.01, N), 0.01, 0.99)
            r0 = np.clip(r + rng.normal(0, 0.005, N), 0.02, 0.3)
            
            # First pass: L-BFGS-B with moderate penalty
            xx, yy, rr = _optimize_all(x0, y0, r0, method='L-BFGS-B', lam=5.0)
            
            # Fix to valid if needed
            xx, yy, rr = _fix_to_valid(xx, yy, rr)
            
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])
            
            # Second pass: SLSQP refinement with higher penalty
            if _elapsed() < DEADLINE - 15.0:
                xx2, yy2, rr2 = _optimize_all(xx, yy, rr, method='SLSQP', lam=50.0)
                xx2, yy2, rr2 = _fix_to_valid(xx2, yy2, rr2)
                
                if _valid(xx2, yy2, rr2):
                    val2 = float(np.sum(rr2))
                    if val2 > best_val:
                        best_val = val2
                        best = np.column_stack([xx2, yy2, rr2])
            
            # Third pass: L-BFGS-B with even higher penalty for final polish
            if _elapsed() < DEADLINE - 20.0:
                xx3, yy3, rr3 = _optimize_all(xx2, yy2, rr2, method='L-BFGS-B', lam=100.0)
                xx3, yy3, rr3 = _fix_to_valid(xx3, yy3, rr3)
                
                if _valid(xx3, yy3, rr3):
                    val3 = float(np.sum(rr3))
                    if val3 > best_val:
                        best_val = val3
                        best = np.column_stack([xx3, yy3, rr3])

    if best is not None:
        return best

    # Guaranteed fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
