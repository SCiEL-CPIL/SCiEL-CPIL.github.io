"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

_DEADLINE = 120.0
_MARGIN = 5.0


def _rect_dims(w):
    """Given width w, height h = (4 - 2w)/2 = 2 - w."""
    h = 2.0 - w
    return w, h


def _evaluate(circles):
    """
    circles: (N,3) array of (x, y, r).
    Returns (sum_radii, valid) where valid checks:
      - all r > 0
      - all circles inside rectangle [0,w]x[0,h]
      - no overlapping interiors
    """
    n = len(circles)
    if n != N:
        return 0.0, False
    if np.any(circles[:, 2] <= 0):
        return 0.0, False
    # Determine bounding rectangle: use the rectangle that best fits
    # We'll optimize w as part of the problem, but for evaluation
    # we check against the tightest enclosing rectangle with perimeter 4.
    # Actually, the problem says "bounding rectangle has perimeter at most 4".
    # So we need: find min over all rectangles of perimeter that contain all circles.
    # For simplicity, we'll assume the rectangle is axis-aligned and we choose w, h
    # such that 2(w+h) <= 4, i.e., w+h <= 2.
    # The circles must be inside [0,w]x[0,h] (after translation).
    # We'll translate so min x = 0, min y = 0.
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    x_min = xs - rs
    x_max = xs + rs
    y_min = ys - rs
    y_max = ys + rs
    width_needed = x_max.max() - x_min.min()
    height_needed = y_max.max() - y_min.min()
    # Perimeter constraint: 2*(width + height) <= 4 => width + height <= 2
    if width_needed + height_needed > 2.0 + 1e-9:
        return 0.0, False
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return 0.0, False
    return rs.sum(), True


def _objective_penalty(x, w):
    """
    x: flattened array of (x_i, y_i, r_i) for i in 0..N-1.
    w: rectangle width (height = 2 - w).
    Returns negative sum of radii + penalties for violations.
    """
    n = N
    h = 2.0 - w
    if h <= 0 or w <= 0:
        return 1e6
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Objective: maximize sum of radii
    obj = -rs.sum()
    
    penalty = 0.0
    
    # Boundary penalties: circles must be inside [0,w]x[0,h]
    # x - r >= 0, x + r <= w, y - r >= 0, y + r <= h
    pen_xmin = np.maximum(0, -(xs - rs))
    pen_xmax = np.maximum(0, (xs + rs) - w)
    pen_ymin = np.maximum(0, -(ys - rs))
    pen_ymax = np.maximum(0, (ys + rs) - h)
    boundary_pen = np.sum(pen_xmin) + np.sum(pen_xmax) + np.sum(pen_ymin) + np.sum(pen_ymax)
    penalty += 100.0 * boundary_pen ** 2
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                penalty += 100.0 * overlap ** 2
    
    # Radius positivity
    neg_r = np.maximum(0, -rs)
    penalty += 10.0 * np.sum(neg_r ** 2)
    
    return obj + penalty


def _grid_init(w):
    """Grid initialization: place circles in a grid pattern."""
    side = int(np.ceil(np.sqrt(N)))
    h = 2.0 - w
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = min(w, h) / (2 * side) * 0.95
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _random_init(w):
    """Random initialization."""
    h = 2.0 - w
    xs = np.random.uniform(0.05 * w, 0.95 * w, N)
    ys = np.random.uniform(0.05 * h, 0.95 * h, N)
    rs = np.random.uniform(0.02, 0.08, N)
    return np.column_stack([xs, ys, rs])


def _hilbert_init(w):
    """Space-filling curve (Z-order) initialization for structured spread."""
    h = 2.0 - w
    # Use a simple Z-order / Morton code approach
    side = int(np.ceil(np.sqrt(N)))
    # Generate Z-order points
    points = []
    for i in range(side * side):
        if len(points) >= N:
            break
        # Convert linear index to 2D using bit interleaving (simplified)
        # Just use a snake pattern for simplicity
        row = i // side
        col = i % side
        if row % 2 == 1:
            col = side - 1 - col
        cx = (col + 0.5) * w / side
        cy = (row + 0.5) * h / side
        r = min(w, h) / (2 * side) * 0.9
        points.append([cx, cy, r])
    return np.array(points[:N])


def _optimize_from(init_pts, w, time_budget):
    """Optimize from an initial configuration."""
    x0 = init_pts.flatten()
    
    def fun(x):
        return _objective_penalty(x, w)
    
    best_x = x0.copy()
    best_val = fun(x0)
    
    # Try L-BFGS-B
    try:
        res = minimize(fun, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Also try Nelder-Mead for local refinement
    try:
        res2 = minimize(fun, best_x, method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
        if res2.fun < best_val:
            best_val = res2.fun
            best_x = res2.x.copy()
    except Exception:
        pass
    
    return best_x, best_val


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    best_circles = None
    best_score = -1e9
    
    # Try multiple widths and initializations
    widths = [0.8, 0.9, 1.0, 1.1, 1.2]
    inits = [_grid_init, _random_init, _hilbert_init]
    
    random.seed(42)
    np.random.seed(42)
    
    for w in widths:
        if time.time() > deadline:
            break
        for init_fn in inits:
            if time.time() > deadline:
                break
            init_pts = init_fn(w)
            opt_x, opt_val = _optimize_from(init_pts, w, 10.0)
            
            # Evaluate
            circles = opt_x.reshape(N, 3)
            # Clamp radii to positive
            circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
            
            score, valid = _evaluate(circles)
            if valid and score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    # If nothing valid found, fall back to grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
