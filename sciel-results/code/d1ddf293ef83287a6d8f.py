"""Circle packing in unit square n=32."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 32
DEADLINE = 110.0  # safety margin before 120s limit


def _voronoi_inradii(points, bounds=(0, 1, 0, 1)):
    """
    Compute the inradius of each Voronoi cell for the given points.
    For each point, the inradius is the minimum distance to:
      - The boundary of the unit square
      - The perpendicular bisectors with other points (approximated by half-distance to neighbors)
    
    This is a simplified Voronoi inradius computation: for each point i,
    r_i = min(distance to square boundary, 0.5 * min distance to other points)
    
    This ensures circles are non-overlapping and inside the square.
    """
    x, y = points[:, 0], points[:, 1]
    n = len(points)
    radii = np.zeros(n)
    
    # Distance to square boundaries
    dist_boundary = np.minimum(
        np.minimum(x, 1.0 - x),
        np.minimum(y, 1.0 - y)
    )
    
    # For each point, find the nearest other point
    for i in range(n):
        min_dist = np.inf
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < min_dist:
                min_dist = d
        # The Voronoi cell edge to neighbor j is at half the distance
        # But we need to consider all neighbors, not just nearest
        # Use a more accurate estimate: min over all j of distance/2
        # Actually, for a proper Voronoi inradius, we'd need the actual cell vertices
        # Simplified: use half the minimum distance to any other point
        r_from_neighbors = 0.5 * min_dist
        radii[i] = min(dist_boundary[i], r_from_neighbors)
    
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
    
    for i in range(n):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1.0 + tol:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
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
    np.random.shuffle(cells)
    cells = cells[:n]
    
    params = np.array([v for c in cells for v in c])
    return params


def _random_init(n):
    """Create a random initialization."""
    params = np.zeros(3 * n)
    for i in range(n):
        params[3 * i] = np.random.uniform(0.1, 0.9)      # x
        params[3 * i + 1] = np.random.uniform(0.1, 0.9)  # y
        params[3 * i + 2] = np.random.uniform(0.02, 0.15) # r
    return params


def _perturb(params, scale=0.05):
    """Add random perturbation to parameters."""
    p = params.copy()
    p[0::3] += np.random.normal(0, scale, size=len(p[0::3]))
    p[1::3] += np.random.normal(0, scale, size=len(p[1::3]))
    p[2::3] += np.random.normal(0, scale * 0.5, size=len(p[2::3]))
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
    
    # Strategy 1: Voronoi-based optimization
    if time.time() - start_time < DEADLINE * 0.5:
        # Try different grid configurations for initial positions
        grid_configs = [
            (4, 8), (8, 4), (5, 7), (7, 5), (6, 6), (4, 6), (6, 4)
        ]
        
        for side_x, side_y in grid_configs:
            if time.time() - start_time > DEADLINE * 0.45:
                break
            
            init_points = _voronoi_grid_init(side_x, side_y, N)
            
            # Optimize positions to maximize sum of Voronoi inradii
            try:
                result = minimize(
                    _voronoi_objective, init_points,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                circles = _voronoi_to_circles(result.x)
                _update_best(circles)
            except Exception:
                pass
            
            # Also try Powell
            if time.time() - start_time > DEADLINE * 0.45:
                break
            try:
                result = minimize(
                    _voronoi_objective, init_points,
                    method='Powell',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                circles = _voronoi_to_circles(result.x)
                _update_best(circles)
            except Exception:
                pass
    
    # Strategy 2: Direct optimization with multiple restarts (fallback/enhancement)
    if time.time() - start_time < DEADLINE * 0.8:
        # Grid-based inits with different aspect ratios
        grid_configs = [
            (4, 8), (8, 4), (5, 7), (7, 5), (6, 6), (4, 6), (6, 4)
        ]
        
        for side_x, side_y in grid_configs:
            if time.time() - start_time > DEADLINE * 0.75:
                break
            
            init = _grid_init(side_x, side_y, N)
            
            # Try multiple optimizers
            for method in ['Nelder-Mead', 'Powell']:
                if time.time() - start_time > DEADLINE * 0.75:
                    break
                
                try:
                    result = minimize(
                        _objective, init,
                        method=method,
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if _is_valid(result.x):
                        score = np.sum(result.x[2::3])
                        if score > best_score:
                            best_score = score
                            best_params = result.x.copy()
                            # Convert to circles format
                            circles = np.zeros((N, 3))
                            circles[:, 0] = result.x[0::3]
                            circles[:, 1] = result.x[1::3]
                            circles[:, 2] = result.x[2::3]
                            _update_best(circles)
                except Exception:
                    pass
        
        # Random inits
        for trial in range(15):
            if time.time() - start_time > DEADLINE * 0.75:
                break
            
            init = _random_init(N)
            # Add perturbation at multiple scales
            for scale in [0.1, 0.05, 0.02]:
                if time.time() - start_time > DEADLINE * 0.75:
                    break
                perturbed = _perturb(init, scale)
                try:
                    result = minimize(
                        _objective, perturbed,
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if _is_valid(result.x):
                        score = np.sum(result.x[2::3])
                        if score > best_score:
                            best_score = score
                            circles = np.zeros((N, 3))
                            circles[:, 0] = result.x[0::3]
                            circles[:, 1] = result.x[1::3]
                            circles[:, 2] = result.x[2::3]
                            _update_best(circles)
                except Exception:
                    pass
    
    # Final refinement if we have a good solution
    if best_circles is not None and time.time() - start_time < DEADLINE * 0.9:
        # Convert to flat params for refinement
        flat_params = best_circles.flatten()
        try:
            final_result = minimize(
                _objective, flat_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if _is_valid(final_result.x):
                score = np.sum(final_result.x[2::3])
                if score > best_score:
                    best_score = score
                    circles = np.zeros((N, 3))
                    circles[:, 0] = final_result.x[0::3]
                    circles[:, 1] = final_result.x[1::3]
                    circles[:, 2] = final_result.x[2::3]
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
      1. Circles inside [0,1]^2: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      2. Disjoint interiors: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    We minimize the negative sum of radii plus penalty for violations.
    """
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for being outside the square
    penalty = 0.0
    for i in range(n):
        # Must be inside [0,1]^2
        if r[i] > x[i]:
            penalty += (r[i] - x[i]) ** 2
        if r[i] > 1.0 - x[i]:
            penalty += (r[i] - (1.0 - x[i])) ** 2
        if r[i] > y[i]:
            penalty += (r[i] - y[i]) ** 2
        if r[i] > 1.0 - y[i]:
            penalty += (r[i] - (1.0 - y[i])) ** 2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return -np.sum(r) + 1000.0 * penalty
