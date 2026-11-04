import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14
TIME_LIMIT = 115.0  # Safety margin for 120s limit

# Precompute triple indices for N=14
# There are C(14, 3) = 364 triples
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation using precomputed indices
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _get_ring_points(r_in, r_out, theta_offset, n_inner=7, n_outer=7):
    """
    Generate points for two concentric regular polygons.
    """
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
    angles_outer = np.linspace(theta_offset, 2*np.pi + theta_offset, n_outer, endpoint=False)
    
    inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
    outer_pts = np.column_stack([r_out * np.cos(angles_outer), r_out * np.sin(angles_outer)])
    
    return np.vstack([inner_pts, outer_pts])

def _score_ring_params(params):
    """
    Score function for ring parameters: [r_in, r_out, theta_offset]
    """
    r_in, r_out, theta_offset = params
    if r_in < 0 or r_out <= 0 or r_in >= r_out:
        return 0.0
        
    points = _get_ring_points(r_in, r_out, theta_offset)
    return _score(points)

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate: -soft_min of triangle areas / hull area.
    Returns a smooth value that approximates -min_area/hull_area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    # This is smooth and approximates min(a) as beta -> inf
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

def _laplacian_relax(points, iterations=50, step_size=0.05):
    """
    Iteratively move points to maximize minimum circumradius (approximated by min triangle area).
    Uses a Delaunay triangulation to identify local neighborhoods.
    """
    points = points.copy()
    
    for _ in range(iterations):
        try:
            # Compute Delaunay triangulation
            tri = Delaunay(points)
            
            # For each point, find its neighbors in the Delaunay mesh
            # and compute the gradient of the minimum area triangle involving this point.
            # Simplified heuristic: Move each point slightly away from the centroid
            # of the smallest triangles it participates in.
            
            # Calculate areas of all triangles in the Delaunay mesh
            tri_points = points[tri.simplices]
            v1 = tri_points[:, 1] - tri_points[:, 0]
            v2 = tri_points[:, 2] - tri_points[:, 0]
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            # Find the minimum area triangle
            min_idx = np.argmin(areas)
            min_tri = tri.simplices[min_idx]
            
            # Move points in the minimum triangle slightly outward
            # to increase the area
            centroid = np.mean(points[min_tri], axis=0)
            
            for idx in min_tri:
                direction = points[idx] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                    points[idx] += step_size * direction * 0.1 # Small step
            
        except Exception:
            break
            
    return points

def _optimize_free_points(points_init, time_left):
    """
    Optimize the 28 coordinates of the points directly using L-BFGS-B with soft-min.
    """
    def neg_soft_min(coords):
        pts = coords.reshape(-1, 2)
        return -_soft_min_score(pts, beta=50.0)
    
    x0 = points_init.flatten()
    
    # Define bounds to keep points within a reasonable range (e.g., unit circle)
    # This is a heuristic to prevent points from flying off to infinity
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    start_time = time.time()
    
    try:
        res = minimize(neg_soft_min, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-10})
        
        # Check time
        if time.time() - start_time > time_left:
            return points_init
            
        best_pts = res.x.reshape(-1, 2)
        return best_pts
        
    except Exception:
        return points_init

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_score = 0
    best_points = None
    
    # 1. Initialize with a good ring configuration
    # Grid search for coarse initialization of ring parameters
    r_in_vals = np.linspace(0.2, 0.9, 8)
    theta_vals = np.linspace(0, np.pi/7, 8)
    
    for r_in in r_in_vals:
        if time.time() > deadline - 5:
            break
        for theta in theta_vals:
            s = _score_ring_params([r_in, 1.0, theta])
            if s > best_score:
                best_score = s
                best_points = _get_ring_points(r_in, 1.0, theta)
                
    if best_points is None:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    # 2. Refine the ring parameters locally
    if time.time() < deadline - 20:
        def _neg_score_params(params):
            return -_score_ring_params(params)
            
        # Use Nelder-Mead to polish the ring solution
        try:
            res = minimize(_neg_score_params, [best_points[0, 0], 1.0, 0], method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            # Note: This is a bit rough. Let's extract the actual best params from the grid search instead.
            # Actually, let's just use the best_points found in the grid and try to optimize it directly.
        except Exception:
            pass
            
    # 3. Structural Change: Free the points from the ring constraint.
    # Use the ring solution as an initial guess for full 28D optimization.
    # Use L-BFGS-B with a soft-min surrogate to handle the non-smoothness of the min operator.
    
    if time.time() < deadline - 10:
        current_points = best_points.copy()
        
        # Apply a few steps of Laplacian-like relaxation to break symmetry slightly?
        # Or just go straight to L-BFGS-B.
        
        time_left = deadline - time.time() - 5
        optimized_points = _optimize_free_points(current_points, time_left)
        
        # Evaluate the optimized points
        opt_score = _score(optimized_points)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized_points
            
    # 4. Final Polish with Nelder-Mead on the 28D space (if time permits)
    # Nelder-Mead is robust for low-dimensional but 28D is high. 
    # However, if L-BFGS-B got us close, NM might polish the corners.
    # Given 28D, NM might be slow. Let's try a few iterations.
    
    if time.time() < deadline - 5:
        def _neg_score_full(coords):
            pts = coords.reshape(-1, 2)
            return -_score(pts)
            
        x0 = best_points.flatten()
        
        try:
            # Use a small maxiter to avoid timeout
            res = minimize(_neg_score_full, x0, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
            
            final_points = res.x.reshape(-1, 2)
            final_score = _score(final_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
                
        except Exception:
            pass
            
    # 5. Compare with regular 14-gon just in case
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg14 = np.column_stack([np.cos(angles), np.sin(angles)])
    reg14_score = _score(reg14)
    
    if reg14_score > best_score:
        return reg14
        
    return best_points

# Test/Run
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
