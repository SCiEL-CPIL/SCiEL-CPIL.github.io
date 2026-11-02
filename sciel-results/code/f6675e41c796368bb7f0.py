"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 135  # seconds, safety margin from 150s limit


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area.
    Uses efficient pairwise cross-product computation.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Precompute pairwise differences
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # For each pair (i,j), compute cross product with all k
    # area(i,j,k) = 0.5 * |dx(i,j)*dy(i,k) - dy(i,j)*dx(i,k)|
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dxi_j = dx[i, j]
            dyi_j = dy[i, j]
            # Cross products for all k > j
            cross = dxi_j * dy[i, j+1:] - dyi_j * dx[i, j+1:]
            areas = 0.5 * np.abs(cross)
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    
    if min_area == float('inf'):
        return 0.0
    
    return min_area / TRI_AREA


def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < 0 or y > S3H:
        return False
    if x < 0 or x > 1:
        return False
    if y > S3H * (1 - x) + 1e-9:
        return False
    if y > S3H * x + 1e-9:
        return False
    return True


def _penalized_min_area(x):
    """Objective for optimization: minimize negative min area plus penalty for out-of-bounds."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        # Distances to boundaries (positive if outside)
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dist = max(0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        penalty += dist ** 2
    
    min_a = _min_area_normalized(points)
    return -min_a + 10.0 * penalty


def _local_polish(points, maxiter=2000):
    """Nelder-Mead polish on the point configuration."""
    x0 = np.array(points, dtype=float).flatten()
    try:
        res = minimize(
            _penalized_min_area,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        polished = res.x.reshape(-1, 2)
        valid = all(_is_inside_triangle(p) for p in polished)
        if valid:
            return polished
    except Exception:
        pass
    return None


def _initial_heilbronn_config():
    """Provide a reasonable initial configuration based on known patterns."""
    # Start with vertices, midpoints, and centroid-like points
    # This gives a structured starting point for optimization
    h = S3H
    config = [
        [0.05, 0.05],           # near bottom-left
        [0.95, 0.05],           # near bottom-right
        [0.5, h - 0.05],        # near top
        [0.25, h * 0.25],       # left middle area
        [0.75, h * 0.25],       # right middle area
        [0.5, h * 0.5],         # center
        [0.15, h * 0.55],       # upper left
        [0.85, h * 0.55],       # upper right
        [0.35, h * 0.75],       # upper center left
        [0.65, h * 0.75],       # upper center right
        [0.5, h * 0.35],        # lower center
    ]
    return np.array(config, dtype=float)


def construct():
    start_time = time.time()
    
    # Initial random configuration
    rng = np.random.RandomState(42)
    
    def random_point_in_triangle():
        # Rejection sampling
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = S3H * v
                # Ensure strictly inside to avoid boundary issues in initial guess
                if 0.01 < x < 0.99 and 0.01 < y < S3H * 0.99:
                    if y < S3H * (1 - x) * 0.99 and y < S3H * x * 0.99:
                        return [x, y]

    best_points = None
    best_score = -float('inf')
    
    # Start with the structured initial configuration
    init_pts = _initial_heilbronn_config()
    init_score = _min_area_normalized(init_pts)
    if init_score > best_score:
        best_score = init_score
        best_points = init_pts.copy()
    
    # Phase 1: Multiple restarts with Nelder-Mead
    n_restarts = 40
    
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        # Alternate between random and perturbed structured starts
        if r % 3 == 0:
            # Perturb the best configuration found so far
            if best_points is not None:
                x0 = (best_points + rng.randn(N, 2) * 0.05).flatten()
            else:
                x0 = np.array([random_point_in_triangle() for _ in range(N)]).flatten()
        elif r % 3 == 1:
            # Random start
            x0 = np.array([random_point_in_triangle() for _ in range(N)]).flatten()
        else:
            # Perturb the structured initial configuration
            x0 = (init_pts + rng.randn(N, 2) * 0.1).flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(
                _penalized_min_area, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
        except Exception:
            continue
        
        if time.time() - start_time > DEADLINE - 10:
            break
            
        pts = res.x.reshape(-1, 2)
        
        # Check validity and score
        valid = True
        for p in pts:
            if not _is_inside_triangle(p):
                valid = False
                break
        
        if valid:
            score = _min_area_normalized(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: return a valid random configuration
        best_points = np.array([random_point_in_triangle() for _ in range(N)])
        best_score = _min_area_normalized(best_points)
    
    # Phase 2: Final polish on the best candidate if time permits
    if time.time() - start_time < DEADLINE - 5:
        polished = _local_polish(best_points, maxiter=5000)
        if polished is not None:
            score = _min_area_normalized(polished)
            if score > best_score:
                best_points = polished
                best_score = score
    
    return best_points
# EVOLVE-BLOCK-END
