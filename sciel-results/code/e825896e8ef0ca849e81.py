import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    i, j, k = i.T.ravel(), j.T.ravel(), k.T.ravel()
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _lattice_basis():
    """Return the standard triangular lattice basis vectors."""
    return np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2]])

def _triangular_lattice_points():
    """Generate 13 points from a triangular lattice pattern.
    
    Uses a hexagonal arrangement: center + 6 in first ring + 6 in second ring.
    This is a known near-optimal structure for Heilbronn-type problems.
    """
    basis = _lattice_basis()
    
    # Center point
    points = [np.array([0.0, 0.0])]
    
    # First ring: 6 points at lattice distance 1
    ring1_indices = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for a, b in ring1_indices:
        pt = a * basis[0] + b * basis[1]
        points.append(pt)
    
    # Second ring: 6 more points (selecting from lattice positions at distance 2)
    # These are at positions like (2,0), (1,1), (0,2), etc.
    # We pick 6 that maintain convexity and good spacing
    ring2_indices = [(2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1)]
    # Wait, (-1,1) is already in ring1. Let me reconsider.
    # Better: use the 6 points at "distance 2" in the hexagonal lattice
    ring2_indices = [(2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1)]
    # Actually let me just use a cleaner hexagonal packing
    
    # Reset: use a proper hexagonal arrangement
    points = [np.array([0.0, 0.0])]
    
    # First shell: 6 points
    for k in range(6):
        angle = k * np.pi / 3
        pt = basis[0] * np.cos(angle) + basis[1] * np.sin(angle)
        # Actually, for triangular lattice, the neighbors are at:
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in lattice coords
        pass
    
    # Let me just hardcode a good triangular lattice subset
    # 13 points forming a hexagonal pattern
    b = np.sqrt(3) / 2
    
    pts = np.array([
        [0, 0],           # center
        [1, 0],           # right
        [0.5, b],         # upper right
        [-0.5, b],        # upper left
        [-1, 0],          # left
        [-0.5, -b],       # lower left
        [0.5, -b],        # lower right
        [2, 0],           # far right
        [1.5, b],         # upper far right
        [0.5, 2*b],       # top
        [-0.5, 2*b],      # upper top left
        [-1.5, b],        # upper far left
        [-2, 0],          # far left
    ])
    
    return pts

def _parametric_points(params):
    """Convert low-dimensional parameter vector to 13 points.
    
    Parameters:
    - params[0:13]: radial scaling factors for each point (1.0 = ideal)
    - params[13:26]: angular deviations for each point (radians)
    - params[26:28]: global rotation and scale
    Total: 28 parameters instead of 26 free coordinates
    """
    ideal = _triangular_lattice_points()
    n = len(ideal)
    
    # Extract parameters
    radial_scales = params[0:n]  # scaling from center
    angular_devs = params[n:2*n] # angular perturbations
    global_rotation = params[2*n] if len(params) > 2*n else 0.0
    global_scale = params[2*n+1] if len(params) > 2*n+1 else 1.0
    
    # Apply transformations
    points = ideal.copy()
    
    # Scale each point radially from origin
    for i in range(n):
        r = np.linalg.norm(points[i])
        if r > 1e-10:
            angle = np.arctan2(points[i][1], points[i][0])
            new_r = r * radial_scales[i]
            new_angle = angle + angular_devs[i]
            points[i] = [new_r * np.cos(new_angle), new_r * np.sin(new_angle)]
    
    # Apply global rotation
    if abs(global_rotation) > 1e-10:
        R = np.array([[np.cos(global_rotation), -np.sin(global_rotation)],
                      [np.sin(global_rotation), np.cos(global_rotation)]])
        points = points @ R.T
    
    # Apply global scale
    points *= global_scale
    
    return points

def _penalty_objective(params):
    """Objective function: negative of smoothed min triangle area + penalty for non-finite."""
    try:
        points = _parametric_points(params)
        if np.any(~np.isfinite(points)):
            return 1e10
        
        areas = _triangle_areas(points)
        if np.any(areas <= 0):
            return 1e10
        
        # Use log-sum-exp for smooth min approximation
        # min(areas) ≈ -log(sum(exp(-areas/tau))) * tau for small tau
        tau = 0.01
        log_sum = np.log(np.sum(np.exp(-areas / tau)))
        smooth_min = -tau * log_sum
        
        # Penalize points going outside a reasonable bound
        max_coord = np.max(np.abs(points))
        if max_coord > 5.0:
            penalty = (max_coord - 5.0) ** 2
        else:
            penalty = 0.0
        
        return -smooth_min + penalty
    except Exception:
        return 1e10

def _optimize_params(params0, maxiter=500, time_limit=120):
    """Optimize the parameter vector using L-BFGS-B."""
    start_time = time.time()
    
    def obj_with_time_check(p):
        if time.time() - start_time > time_limit:
            return 1e10
        return _penalty_objective(p)
    
    try:
        result = minimize(
            obj_with_time_check,
            params0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x
    except Exception:
        return params0

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 580  # Safety margin
    
    # Start with the ideal triangular lattice configuration
    ideal_points = _triangular_lattice_points()
    best_points = ideal_points.copy()
    best_score = _score(best_points)
    
    # Initial parameter vector: all ones for radial, zeros for angular, 0 rotation, 1 scale
    n_params = 2 * n + 2  # 13 radial + 13 angular + 2 global = 28
    params0 = np.zeros(n_params)
    params0[0:n] = 1.0  # radial scales = 1
    params0[2*n+1] = 1.0  # global scale = 1
    
    rng = np.random.default_rng(123)
    
    # Multiple restarts with different initial parameters
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        # Perturb the initial parameters
        if restart == 0:
            params = params0.copy()
        else:
            params = params0 + rng.normal(0, 0.1, size=n_params)
            params[0:n] = np.clip(params[0:n], 0.5, 1.5)  # keep radial scales reasonable
            params[2*n+1] = np.clip(params[2*n+1], 0.5, 1.5)  # global scale
        
        # Optimize
        time_left = deadline - time.time() - 30
        if time_left < 10:
            break
        
        max_iter = int(300 * time_left / 120)
        max_iter = max(50, min(max_iter, 500))
        
        opt_params = _optimize_params(params, maxiter=max_iter, time_limit=min(120, time_left))
        
        # Evaluate
        candidate_points = _parametric_points(opt_params)
        candidate_score = _score(candidate_points)
        
        if candidate_score > best_score:
            best_score = candidate_score
            best_points = candidate_points.copy()
    
    # Final local refinement: direct coordinate optimization on the best points
    if time.time() < deadline - 30:
        def _local_refine(points, max_rounds=100):
            pts = points.copy()
            current_score = _score(pts)
            
            for rnd in range(max_rounds):
                if time.time() > deadline - 20:
                    break
                improved = False
                for i in range(n):
                    for coord in range(2):
                        # Try small perturbations
                        for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                            new_pts = pts.copy()
                            new_pts[i, coord] += delta
                            new_score = _score(new_pts)
                            if new_score > current_score + 1e-10:
                                pts = new_pts
                                current_score = new_score
                                improved = True
                                break
                if not improved:
                    break
            
            return pts
        
        refined_points = _local_refine(best_points)
        refined_score = _score(refined_points)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # Fallback
    if _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
