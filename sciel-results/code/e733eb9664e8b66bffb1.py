"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200

def _score(points):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    # Min distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    # Min pairwise distance
    d = pdist(points)
    min_pair = np.min(d)
    return min(min_pair / 2.0, min_boundary)

def _neg_radius(points):
    """Negative of the radius for minimization."""
    if not np.all(np.isfinite(points)):
        return 1.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 1.0
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    d = pdist(points)
    min_pair = np.min(d)
    return -min(min_pair / 2.0, min_boundary)

def _square_grid(n):
    """Generate a square grid packing."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:n]
    return points

def _hex_grid(n):
    """Generate a hexagonal grid packing."""
    best_points = None
    best_score = 0
    
    for rows in range(10, 30):
        cols = n // rows
        if cols < 1:
            continue
        
        s_x = 1.0 / cols
        s_y = 1.0 / ((rows - 1) * np.sqrt(3) / 2)
        s = min(s_x, s_y)
        
        points = []
        for j in range(rows):
            offset = (j % 2) * s / 2
            for i in range(cols):
                x = -0.5 + s/2 + i * s + offset
                y = -0.5 + s * np.sqrt(3) / 4 + j * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points[:n])
        if len(points) == n:
            sc = _score(points)
            if sc > best_score:
                best_score = sc
                best_points = points.copy()
    
    if best_points is None:
        best_points = _square_grid(n)
    return best_points

def _optimize(points, deadline, max_iter=500):
    """Use L-BFGS-B to optimize the packing."""
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return _neg_radius(pts)
    
    def grad(x):
        pts = x.reshape(-1, 2)
        # Numerical gradient
        g = np.zeros_like(x)
        eps = 1e-7
        f0 = _neg_radius(pts)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            g[i] = (_neg_radius(xp.reshape(-1, 2)) - f0) / eps
        return g
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
                callback=None
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            break
        if time.time() > deadline - 10:
            break
    
    return best_x.reshape(-1, 2), -best_f

def construct():
    deadline = time.time() + 140.0
    
    # Generate initial points from different grids
    sq_pts = _square_grid(N)
    hex_pts = _hex_grid(N)
    
    # Also try a slightly perturbed hex grid
    np.random.seed(42)
    hex_pts2 = hex_pts + np.random.randn(N, 2) * 0.001
    
    candidates = [sq_pts, hex_pts, hex_pts2]
    best_points = None
    best_score = 0
    
    for pts in candidates:
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # Optimize the best candidate
    if time.time() < deadline - 20:
        opt_pts, opt_score = _optimize(best_points, deadline, max_iter=300)
        if opt_score > best_score:
            best_points = opt_pts
            best_score = opt_score
    
    # Try optimizing the second-best candidate if time permits
    if time.time() < deadline - 20:
        for pts in candidates:
            if np.array_equal(pts, best_points):
                continue
            sc = _score(pts)
            if sc < best_score - 0.001:
                continue
            opt_pts, opt_score = _optimize(pts, deadline, max_iter=200)
            if opt_score > best_score:
                best_points = opt_pts
                best_score = opt_score
            if time.time() > deadline - 15:
                break
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        opt_pts, opt_score = _optimize(best_points, deadline, max_iter=100)
        if opt_score > best_score:
            best_points = opt_pts
            best_score = opt_score
    
    return best_points
