import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points (assumes valid shape)."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calc_ratio(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        if len(points) < 3:
            return 0.0
            
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
            
        min_area = _min_triangle_area(points)
        return min_area / hull_area
    except Exception:
        return 0.0

def objective(points_flat):
    """
    Objective function for minimization.
    Minimizes negative score.
    """
    points = points_flat.reshape(-1, 2)
    score = calc_ratio(points)
    return -score

def normalize_points(points):
    """
    Scale points so that the convex hull area is 1.0.
    """
    try:
        area = _hull_area(points)
        if area < 1e-10:
            return points
        scale_factor = 1.0 / np.sqrt(area)
        return points * scale_factor
    except Exception:
        return points

def _regular_polygon(n, radius=1.0):
    """Generate regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_circles(n, n_outer=7, n_inner=7, r_inner=0.5):
    """Generate points on two concentric circles."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _perturbed_polygon(n, radius=1.0, noise_scale=0.05, seed=42):
    """Generate perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    base = _regular_polygon(n, radius)
    noise = rng.uniform(-noise_scale, noise_scale, (n, 2))
    return base + noise

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    # List of initial configurations to try
    inits = []
    
    # 1. Regular polygon
    inits.append(_regular_polygon(N, 1.0))
    
    # 2. Perturbed regular polygons with different noise scales
    for noise_scale in [0.02, 0.05, 0.1, 0.15]:
        inits.append(_perturbed_polygon(N, 1.0, noise_scale, seed=42+int(noise_scale*100)))
    
    # 3. Concentric circles
    inits.append(_concentric_circles(N, 7, 7, 0.5))
    inits.append(_concentric_circles(N, 7, 7, 0.4))
    inits.append(_concentric_circles(N, 8, 6, 0.5))
    inits.append(_concentric_circles(N, 6, 8, 0.5))
    
    # 4. Random points in unit disk
    rng = np.random.RandomState(123)
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        random_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(random_pts)
    
    # 5. Grid-like arrangement
    # 4x4 grid slightly perturbed
    x = np.linspace(-0.5, 0.5, 4)
    y = np.linspace(-0.5, 0.5, 4)
    gx, gy = np.meshgrid(x, y)
    grid_pts = np.column_stack([gx.flatten(), gy.flatten()])
    grid_pts = grid_pts[:N]
    inits.append(grid_pts)
    
    # Optimization Loop
    for idx_init, current_init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Normalize initial points to unit area to keep scale consistent
        current_init = normalize_points(current_init)
        
        # Try multiple optimization methods
        methods = ['L-BFGS-B', 'Powell', 'Nelder-Mead']
        
        for method in methods:
            if time.time() > deadline:
                break
                
            try:
                time_left = deadline - time.time()
                
                # Determine maxiter based on time remaining
                if method == 'L-BFGS-B':
                    maxiter = int(max(50, time_left * 10))
                    options = {'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
                elif method == 'Powell':
                    maxiter = int(max(50, time_left * 5))
                    options = {'maxiter': maxiter, 'ftol': 1e-10, 'xtol': 1e-10}
                else:
                    maxiter = int(max(100, time_left * 20))
                    options = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                
                res = scipy.optimize.minimize(
                    objective, 
                    current_init.flatten(), 
                    method=method, 
                    options=options
                )
                
                optimized_points = res.x.reshape(-1, 2)
                
                # Normalize to unit area
                optimized_points = normalize_points(optimized_points)
                
                score = calc_ratio(optimized_points)
                
                if score > best_score:
                    best_score = score
                    best_points = optimized_points.copy()
                    
            except Exception:
                pass

    # If no points found or score is 0, return a regular polygon as fallback
    if best_points is None:
        best_points = normalize_points(_regular_polygon(N, 1.0))
    else:
        # Final normalization to ensure strict unit area hull
        best_points = normalize_points(best_points)
        
    return best_points
