"""Circle packing in unit square n=32."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 32
DEADLINE = 140.0  # safety margin before 150s kill


def _voronoi_inradii(points, bounds=(0, 1, 0, 1)):
    """
    Compute the inradius of each Voronoi cell for the given points.
    For each point i, r_i = min(distance to square boundary, 0.5 * min distance to other points)
    """
    x, y = points[:, 0], points[:, 1]
    n = len(points)
    radii = np.zeros(n)
    
    # Distance to square boundaries (vectorized)
    dist_boundary = np.minimum(
        np.minimum(x, 1.0 - x),
        np.minimum(y, 1.0 - y)
    )
    
    # For each point, find minimum distance to any other point (vectorized)
    # Compute pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Minimum distance to any other point for each i
    min_dists = np.min(dist_matrix, axis=1)
    r_from_neighbors = 0.5 * min_dists
    
    radii = np.minimum(dist_boundary, r_from_neighbors)
    return radii


def _voronoi_objective(params):
    """
    Objective for Voronoi-based packing.
    params: shape (2N,) -> [x_0, y_0, x_1, y_1, ...]
    We maximize sum of inradii (minimize negative sum).
    The radii are implicitly determined by the Voronoi cells.
    """
    n = len(params) // 2
    points = params.reshape(n, 2)
    # Clamp to valid range to avoid issues
    points = np.clip(points, 0.01, 0.99)
    radii = _voronoi_inradii(points)
    return -np.sum(radii)


def _voronoi_to_circles(params):
    """Convert Voronoi parameters to circle representation (x, y, r)."""
    n = len(params) // 2
    points = params.reshape(n, 2)
    radii = _voronoi_inradii(points)
    circles = np.zeros((n, 3))
    circles[:, 0] = points[:, 0]
    circles[:, 1] = points[:, 1]
    circles[:, 2] = radii
    return circles


def _is_valid(params, tol=1e-6):
    """Check if the configuration is valid (all circles inside square, no overlaps)."""
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Check bounds (vectorized)
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1.0 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1.0 + tol):
        return False
    
    # Check overlaps (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    min_dist_matrix = r[:, None] + r[None, :]
    np.fill_diagonal(dist_matrix, np.inf)
    np.fill_diagonal(min_dist_matrix, 0)
    
    if np.any(dist_matrix < min_dist_matrix - tol):
        return False
    return True


def _grid_init(side_x, side_y, n):
    """Create a grid-based initialization."""
    cells = []
    for i in range(side_x):
        for j in range(side_y):
            x = (i + 0.5) / side_x
            y = (j + 0.5) / side_y
            r = 0.5 / max(side_x, side_y) * 0.95
            cells.append([x, y, r])
    
    # Shuffle and take first n
    idx = np.random.permutation(len(cells))[:n]
    cells = [cells[i] for i in idx]
    
    params = np.array([v for c in cells for v in c])
    return params


def _random_init(n):
    """Create a random initialization."""
    x = np.random.uniform(0.1, 0.9, size=n)
    y = np.random.uniform(0.1, 0.9, size=n)
    r = np.random.uniform(0.02, 0.15, size=n)
    params = np.empty(3 * n)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def _perturb(params, scale=0.05):
    """Add random perturbation to parameters."""
    p = params.copy()
    p[0::3] += np.random.normal(0, scale, size=N)
    p[1::3] += np.random.normal(0, scale, size=N)
    p[2::3] += np.random.normal(0, scale * 0.5, size=N)
    # Clip to valid ranges
    p[0::3] = np.clip(p[0::3], 0.01, 0.99)
    p[1::3] = np.clip(p[1::3], 0.01, 0.99)
    p[2::3] = np.clip(p[2::3], 0.005, 0.4)
    return p


def _voronoi_grid_init(side_x, side_y, n):
    """Create grid-based initialization for Voronoi (positions only)."""
    points = np.zeros((n, 2))
    count = 0
    for i in range(side_x):
        for j in range(side_y):
            if count >= n:
                break
            points[count, 0] = (i + 0.5) / side_x
            points[count, 1] = (j + 0.5) / side_y
            count += 1
        if count >= n:
            break
    return points.flatten()


def _physics_objective(params, epsilon=0.01):
    """
    Physics-based repulsive potential objective.
    params: shape (3N,) -> [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    
    Uses a softened repulsive potential to avoid singularities:
    - Repulsion between circles: sum over pairs of 1/(d_ij - r_i - r_j + eps)^2 when overlapping
    - Attraction to boundaries: encourages circles to be near walls (maximizes space)
    - Penalty for being outside the square
    
    We minimize: -sum(r_i) + repulsion_penalty + boundary_penalty
    """
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Clamp radii to positive
    r_clipped = np.maximum(r, 0.001)
    
    # Boundary penalty: circles should be inside [0,1]^2
    # Soft penalty: (max(0, r - dist_to_wall))^2
    dist_left = x
    dist_right = 1.0 - x
    dist_bottom = y
    dist_top = 1.0 - y
    
    boundary_violation = np.maximum(0, r_clipped - dist_left) + \
                          np.maximum(0, r_clipped - dist_right) + \
                          np.maximum(0, r_clipped - dist_bottom) + \
                          np.maximum(0, r_clipped - dist_top)
    boundary_penalty = np.sum(boundary_violation**2)
    
    # Repulsion between circles: softened 1/d^2 potential
    # For pairs where d < r_i + r_j, add penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    min_d = r_clipped[:, None] + r_clipped[None, :]
    
    # Avoid division by zero
    d_safe = np.maximum(d, epsilon)
    
    # Overlap amount: positive when circles overlap
    overlap = np.maximum(0, min_d - d_safe)
    
    # Softened repulsion: sum of overlap^2 (smooth)
    # Use upper triangle only to avoid double counting
    iu = np.triu_indices(n, k=1)
    repulsion = np.sum(overlap[iu]**2)
    
    # Objective: maximize sum of radii minus penalties
    return -np.sum(r_clipped) + 100.0 * repulsion + 50.0 * boundary_penalty


def construct():
    start_time = time.time()
    best_circles = None
    best_score = -np.inf
    
    def _update_best(circles):
        nonlocal best_circles, best_score
        if circles is None:
            return
        if not _is_valid(circles.flatten()):
            return
        score = np.sum(circles[:, 2])
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    def _elapsed():
        return time.time() - start_time
    
    # Strategy 1: Voronoi-based optimization (positions only, radii from Voronoi)
    if _elapsed() < DEADLINE * 0.35:
        grid_configs = [
            (4, 8), (8, 4), (5, 7), (7, 5), (6, 6), (4, 6), (6, 4)
        ]
        
        for side_x, side_y in grid_configs:
            if _elapsed() > DEADLINE * 0.33:
                break
            
            init_points = _voronoi_grid_init(side_x, side_y, N)
            
            # Optimize positions to maximize sum of Voronoi inradii
            try:
                result = minimize(
                    _voronoi_objective, init_points,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                circles = _voronoi_to_circles(result.x)
                _update_best(circles)
            except Exception:
                pass
            
            # Also try Powell
            if _elapsed() > DEADLINE * 0.33:
                break
            try:
                result = minimize(
                    _voronoi_objective, init_points,
                    method='Powell',
                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                circles = _voronoi_to_circles(result.x)
                _update_best(circles)
            except Exception:
                pass
    
    # Strategy 2: Physics-based optimization with repulsive potential
    if _elapsed() < DEADLINE * 0.65:
        # Grid-based inits
        grid_configs = [
            (4, 8), (8, 4), (5, 7), (7, 5), (6, 6), (4, 6), (6, 4)
        ]
        
        for side_x, side_y in grid_configs:
            if _elapsed() > DEADLINE * 0.63:
                break
            
            init = _grid_init(side_x, side_y, N)
            
            # Try physics-based objective with Nelder-Mead
            try:
                result = minimize(
                    _physics_objective, init,
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                if _is_valid(result.x):
                    circles = np.zeros((N, 3))
                    circles[:, 0] = result.x[0::3]
                    circles[:, 1] = result.x[1::3]
                    circles[:, 2] = np.maximum(result.x[2::3], 0)
                    _update_best(circles)
            except Exception:
                pass
            
            # Also try with Powell
            if _elapsed() > DEADLINE * 0.63:
                break
            try:
                result = minimize(
                    _physics_objective, init,
                    method='Powell',
                    options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                if _is_valid(result.x):
                    circles = np.zeros((N, 3))
                    circles[:, 0] = result.x[0::3]
                    circles[:, 1] = result.x[1::3]
                    circles[:, 2] = np.maximum(result.x[2::3], 0)
                    _update_best(circles)
            except Exception:
                pass
        
        # Random inits with physics objective
        for trial in range(10):
            if _elapsed() > DEADLINE * 0.63:
                break
            
            init = _random_init(N)
            for scale in [0.05, 0.02]:
                if _elapsed() > DEADLINE * 0.63:
                    break
                perturbed = _perturb(init, scale)
                try:
                    result = minimize(
                        _physics_objective, perturbed,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
                    )
                    if _is_valid(result.x):
                        circles = np.zeros((N, 3))
                        circles[:, 0] = result.x[0::3]
                        circles[:, 1] = result.x[1::3]
                        circles[:, 2] = np.maximum(result.x[2::3], 0)
                        _update_best(circles)
                except Exception:
                    pass
    
    # Strategy 3: Final high-precision polish pass
    if best_circles is not None and _elapsed() < DEADLINE * 0.92:
        flat_params = best_circles.flatten()
        
        # Use the physics objective for polishing (smoother landscape)
        try:
            final_result = minimize(
                _physics_objective, flat_params,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if _is_valid(final_result.x):
                circles = np.zeros((N, 3))
                circles[:, 0] = final_result.x[0::3]
                circles[:, 1] = final_result.x[1::3]
                circles[:, 2] = np.maximum(final_result.x[2::3], 0)
                _update_best(circles)
        except Exception:
            pass
        
        # Second polish with even tighter tolerance
        if _elapsed() < DEADLINE * 0.95:
            try:
                final_result2 = minimize(
                    _physics_objective, best_circles.flatten(),
                    method='Powell',
                    options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14}
                )
                if _is_valid(final_result2.x):
                    circles = np.zeros((N, 3))
                    circles[:, 0] = final_result2.x[0::3]
                    circles[:, 1] = final_result2.x[1::3]
                    circles[:, 2] = np.maximum(final_result2.x[2::3], 0)
                    _update_best(circles)
            except Exception:
                pass
    
    if best_circles is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Ensure validity one more time
    if not _is_valid(best_circles.flatten()):
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles


def _objective(params):
    """
    params: shape (3N,) -> [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Objective: maximize sum of radii subject to:
      1. Circles inside [0,1]^2
      2. Disjoint interiors
    Vectorized version for speed.
    """
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    r_clipped = np.maximum(r, 0.001)
    
    # Boundary penalty (vectorized)
    boundary_violation = np.maximum(0, r_clipped - x) + \
                          np.maximum(0, r_clipped - (1.0 - x)) + \
                          np.maximum(0, r_clipped - y) + \
                          np.maximum(0, r_clipped - (1.0 - y))
    boundary_penalty = np.sum(boundary_violation**2)
    
    # Overlap penalty (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    min_d = r_clipped[:, None] + r_clipped[None, :]
    d_safe = np.maximum(d, 0.01)
    overlap = np.maximum(0, min_d - d_safe)
    iu = np.triu_indices(n, k=1)
    overlap_penalty = np.sum(overlap[iu]**2)
    
    return -np.sum(r_clipped) + 100.0 * overlap_penalty + 50.0 * boundary_penalty
