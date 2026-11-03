"""Circle packing in unit square using Delaunay-based radius derivation.

Key structural change: instead of optimizing all 78 variables (x, y, r) jointly,
we optimize only the 52 center positions (x, y). Radii are derived geometrically
from the Delaunay triangulation: each circle's radius is half the minimum of
its incident edge lengths and its distance to the boundary. This enforces a
"tight packing" structure where every circle touches at least one neighbor or
the boundary, reducing the search space and making the objective smoother.

Uses:
- Delaunay triangulation for neighbor discovery and radius computation
- L-BFGS-B on 52-dim position space
- Multiple diverse initializations
- SLSQP refinement on positions
- Deadline guard for safety
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, distance as cdist_mod

N = 26
DEADLINE = 135.0  # Leave 15s safety margin under 150s
START = time.time()


def _elapsed():
    return time.time() - START


def _compute_radii(x, y):
    """
    Given center positions, compute radii based on Delaunay triangulation.
    Each circle's radius = 0.5 * min(shortest incident edge, distance to boundary).
    """
    pts = np.column_stack([x, y])
    
    # Distance to boundary
    d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    try:
        tri = Delaunay(pts)
    except Exception:
        # Fallback: use nearest neighbor distance
        from scipy.spatial.distance import cdist
        dist_mat = cdist(pts, pts)
        np.fill_diagonal(dist_mat, np.inf)
        nn_dist = np.min(dist_mat, axis=1)
        return np.minimum(0.5 * nn_dist, d_bound)
    
    # Build adjacency from Delaunay
    n = len(pts)
    min_edge = np.full(n, np.inf)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                d = np.sqrt((x[a] - x[b])**2 + (y[a] - y[b])**2)
                if d < min_edge[a]:
                    min_edge[a] = d
                if d < min_edge[b]:
                    min_edge[b] = d
    
    # For any point with no Delaunay edge (shouldn't happen for n>=3), use nearest neighbor
    mask = np.isinf(min_edge)
    if np.any(mask):
        from scipy.spatial.distance import cdist
        dist_mat = cdist(pts, pts)
        np.fill_diagonal(dist_mat, np.inf)
        nn_dist = np.min(dist_mat, axis=1)
        min_edge[mask] = nn_dist[mask]
    
    # Radius is half the minimum of shortest edge and boundary distance
    r = 0.5 * np.minimum(min_edge, d_bound)
    r = np.maximum(r, 1e-8)
    return r


def _objective_positions(params):
    """
    Objective to minimize: -sum(r) where r is derived from positions.
    params: 52-dim vector [x0, y0, x1, y1, ..., x25, y25]
    """
    x = params[0::2]
    y = params[1::2]
    
    # Clip to valid ranges
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    
    r = _compute_radii(x, y)
    
    return -np.sum(r)


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


def _fix_to_valid(x, y, r, max_iters=50):
    """
    Iteratively shrink radii and adjust positions to eliminate overlaps.
    """
    for _ in range(max_iters):
        # Check boundary
        d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        r = np.minimum(r, d_bound)
        
        # Check pairwise overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < r[i] + r[j]:
                    total_r = r[i] + r[j]
                    if total_r > 0:
                        scale = dist[i, j] / total_r * 0.999
                        r[i] *= scale
                        r[j] *= scale
                    else:
                        r[i] = r[j] = 1e-6
                    changed = True
        
        if not changed:
            break
    
    return x, y, np.maximum(r, 1e-6)


def _grid_init_5x5_plus1(rng):
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _grid_init_4x7(rng):
    cols, rows = 4, 7
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _grid_init_6x5(rng):
    cols, rows = 6, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _random_init(rng):
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    return x, y


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
    return x, y


def _optimize_positions(x, y, method='L-BFGS-B'):
    """Optimize center positions only."""
    params = np.zeros(2 * N)
    params[0::2] = x
    params[1::2] = y
    
    if method == 'L-BFGS-B':
        try:
            res = minimize(_objective_positions, params, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 5000})
        except Exception:
            return x, y
    elif method == 'SLSQP':
        try:
            res = minimize(_objective_positions, params, method='SLSQP',
                           options={'maxiter': 300, 'ftol': 1e-10})
        except Exception:
            return x, y
    else:
        return x, y
    
    v = res.x
    xx = np.clip(v[0::2], 0.0, 1.0)
    yy = np.clip(v[1::2], 0.0, 1.0)
    
    return xx, yy


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
        if _elapsed() > DEADLINE - 15.0:
            break
        
        x, y = init_func(rng)
        
        for restart in range(5):
            if _elapsed() > DEADLINE - 15.0:
                break
            
            # Add small jitter
            x0 = np.clip(x + rng.normal(0, 0.02, N), 0.01, 0.99)
            y0 = np.clip(y + rng.normal(0, 0.02, N), 0.01, 0.99)
            
            # First pass: L-BFGS-B on positions
            xx, yy = _optimize_positions(x0, y0, method='L-BFGS-B')
            
            # Compute radii from optimized positions
            rr = _compute_radii(xx, yy)
            
            # Fix to valid
            xx, yy, rr = _fix_to_valid(xx, yy, rr)
            
            # Second pass: SLSQP refinement on positions
            if _elapsed() < DEADLINE - 20.0:
                xx2, yy2 = _optimize_positions(xx, yy, method='SLSQP')
                rr2 = _compute_radii(xx2, yy2)
                xx2, yy2, rr2 = _fix_to_valid(xx2, yy2, rr2)
                
                if _valid(xx2, yy2, rr2):
                    val2 = float(np.sum(rr2))
                    if val2 > best_val:
                        best_val = val2
                        best = np.column_stack([xx2, yy2, rr2])
            
            # Check first pass result
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])

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
