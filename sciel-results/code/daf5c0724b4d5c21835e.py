"""Circle packing: Voronoi inradius optimization with multi-start and local polish."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay, ConvexHull

N = 26
_START = time.time()
_DEADLINE = 145.0
_EPS = 1e-9


def _compute_inradii(points):
    """Compute the inradius for each point (max radius of circle centered at point[i]
    that fits within the Voronoi cell and the unit square)."""
    n = len(points)
    inradii = np.zeros(n)
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: use distance to nearest other point / 2
        for i in range(n):
            dmin = min(np.sqrt((points[i] - points[j])**2).sum() for j in range(n) if j != i)
            inradii[i] = max(dmin * 0.5, 1e-4)
        return inradii
    
    # For each point, find its Voronoi cell and compute inradius
    for i in range(n):
        region_idx = vor.regions[vor.point_region[i]]
        if len(region_idx) == 0:
            inradii[i] = 1e-4
            continue
        
        vertices = vor.vertices[region_idx]
        if len(vertices) == 0:
            inradii[i] = 1e-4
            continue
        
        min_dist = 0.5  # max possible radius in unit square
        
        # Distance to unit square boundaries
        px, py = points[i]
        min_dist = min(min_dist, px, py, 1.0 - px, 1.0 - py)
        
        # Distance to each edge of the Voronoi cell
        nv = len(vertices)
        for k in range(nv):
            v1 = vor.vertices[vertices[k]]
            v2 = vor.vertices[vertices[(k + 1) % nv]]
            if v1 is None or v2 is None:
                continue
            x1, y1 = v1
            x2, y2 = v2
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1e-12:
                continue
            dist = abs(dy*(px - x1) - dx*(py - y1)) / length
            min_dist = min(min_dist, dist)
        
        inradii[i] = max(min_dist, 1e-4)
    
    return inradii


def _voronoi_objective(p):
    """Negative sum of Voronoi inradii (to be minimized)."""
    points = p.reshape(N, 2)
    inradii = _compute_inradii(points)
    return -np.sum(inradii)


def _valid_and_score(arr):
    """Return sum of radii if valid (disjoint, in square), else -inf."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    if np.any(r <= 0) or np.any(np.isnan(arr)):
        return -np.inf
    if np.any(x < -_EPS) or np.any(x > 1 + _EPS) or np.any(y < -_EPS) or np.any(y > 1 + _EPS):
        return -np.inf
    if np.any(r < -_EPS) or np.any(x + r > 1 + _EPS) or np.any(x - r < -_EPS) \
            or np.any(y + r > 1 + _EPS) or np.any(y - r < -_EPS):
        return -np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if d < r[i] + r[j] - 1e-7:
                return -np.inf
    return float(np.sum(r))


def _polish_positions(p):
    """Local optimization of positions only, maximizing sum of Voronoi inradii."""
    res = minimize(_voronoi_objective, p, method='L-BFGS-B',
                   bounds=[(0.01, 0.99)] * (2 * N),
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    return res.x


def _greedy_init(rng):
    """Greedy largest-empty-circle initialization using Voronoi vertices."""
    # Start with a few random points
    points = rng.uniform(0.1, 0.9, size=(5, 2))
    
    for _ in range(N - 5):
        # Compute Voronoi diagram
        try:
            vor = Voronoi(points)
        except Exception:
            # Fallback: add random point
            points = np.vstack([points, rng.uniform(0.05, 0.95, size=2)])
            continue
        
        # Find the vertex with maximum inradius (largest empty circle)
        best_r = 0
        best_pos = None
        
        # Check all Voronoi vertices
        for v in vor.vertices:
            if v is None:
                continue
            x, y = v
            if x < 0.01 or x > 0.99 or y < 0.01 or y > 0.99:
                continue
            
            # Check if this vertex is inside the unit square
            # Compute distance to all existing points
            dmin = min(np.sqrt((x - px)**2 + (y - py)**2) for px, py in points)
            if dmin > best_r:
                best_r = dmin
                best_pos = np.array([x, y])
        
        # If no good vertex found, add a random point
        if best_pos is None:
            best_pos = rng.uniform(0.05, 0.95, size=2)
        
        points = np.vstack([points, best_pos])
    
    return points


def construct():
    n = N
    best_arr = None
    best_val = -np.inf
    rng = np.random.default_rng(42)
    
    def _update_best(arr):
        nonlocal best_arr, best_val
        val = _valid_and_score(arr)
        if val > best_val:
            best_val = val
            best_arr = arr.copy()
    
    # ---- Phase 1: Multi-start with greedy initialization ----
    n_starts = 0
    max_starts = 15
    while time.time() < _DEADLINE - 30.0 and n_starts < max_starts:
        # Greedy initialization
        points = _greedy_init(rng)
        p0 = points.flatten()
        
        # Optimize positions
        p_opt = _polish_positions(p0)
        x_opt = p_opt[:n]
        y_opt = p_opt[n:2*n]
        points_opt = np.column_stack([x_opt, y_opt])
        
        # Compute inradii
        inradii = _compute_inradii(points_opt)
        arr = np.column_stack([x_opt, y_opt, inradii])
        arr[:, 0] = np.clip(arr[:, 0], 0, 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, 1)
        arr[:, 2] = np.clip(arr[:, 2], 1e-4, 0.5)
        
        _update_best(arr)
        n_starts += 1
    
    # ---- Phase 2: Random restarts with local polish ----
    n_random = 0
    max_random = 10
    while time.time() < _DEADLINE - 15.0 and n_random < max_random:
        x = rng.uniform(0.05, 0.95, size=n)
        y = rng.uniform(0.05, 0.95, size=n)
        p0 = np.concatenate([x, y])
        
        p_opt = _polish_positions(p0)
        x_opt = p_opt[:n]
        y_opt = p_opt[n:2*n]
        points_opt = np.column_stack([x_opt, y_opt])
        
        inradii = _compute_inradii(points_opt)
        arr = np.column_stack([x_opt, y_opt, inradii])
        arr[:, 0] = np.clip(arr[:, 0], 0, 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, 1)
        arr[:, 2] = np.clip(arr[:, 2], 1e-4, 0.5)
        
        _update_best(arr)
        n_random += 1
    
    # ---- Phase 3: Polish the best candidate ----
    if best_arr is not None and time.time() < _DEADLINE - 5.0:
        p = np.concatenate([best_arr[:, 0], best_arr[:, 1]])
        p_opt = _polish_positions(p)
        x_opt = p_opt[:n]
        y_opt = p_opt[n:2*n]
        points_opt = np.column_stack([x_opt, y_opt])
        
        inradii = _compute_inradii(points_opt)
        arr = np.column_stack([x_opt, y_opt, inradii])
        arr[:, 0] = np.clip(arr[:, 0], 0, 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, 1)
        arr[:, 2] = np.clip(arr[:, 2], 1e-4, 0.5)
        
        _update_best(arr)
    
    # ---- Fallback: grid-based initialization ----
    if best_arr is None or not np.isfinite(best_val):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    # Final safety clip
    best_arr[:, 0] = np.clip(best_arr[:, 0], 0, 1)
    best_arr[:, 1] = np.clip(best_arr[:, 1], 0, 1)
    best_arr[:, 2] = np.clip(best_arr[:, 2], 1e-4, 0.5)
    return best_arr
