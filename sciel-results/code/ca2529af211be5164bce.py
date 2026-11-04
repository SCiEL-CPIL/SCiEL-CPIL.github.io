import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11
MIN_DIST = np.sqrt(3)  # Minimum distance between hexagon centers

def compute_min_enclosing_hexagon(points):
    """
    Compute the minimum side length of a regular hexagon that encloses all points.
    The hexagon can be rotated optimally.
    
    Parameters:
    points: (N, 2) array of point coordinates
    
    Returns:
    (side_length, rotation_angle) - optimal side length and rotation angle in radians
    """
    # Center points at origin
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    
    # For a regular hexagon with side length L and rotation angle theta,
    # the distance from center to each side is L * sqrt(3) / 2
    # The constraint is: max over all points and all 6 side normals of |p . n_k(theta)| <= L * sqrt(3) / 2
    
    def hexagon_side_length(theta):
        """Compute the minimum side length for a hexagon with rotation theta."""
        # Generate 6 normal vectors for the hexagon sides
        angles = theta + np.linspace(0, 2*np.pi, 7)[:-1]
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Project all points onto each normal
        projections = pts @ normals.T  # (N, 6)
        max_proj = np.max(np.abs(projections))
        
        # Side length L = max_proj / (sqrt(3)/2)
        return max_proj / (np.sqrt(3) / 2)
    
    # Optimize over rotation angle theta
    # The function is periodic with period pi/3 (60 degrees)
    # Try multiple starting points to find global minimum
    best_L = float('inf')
    best_theta = 0
    
    for theta_init in np.linspace(0, np.pi/3, 12, endpoint=False):
        result = minimize(hexagon_side_length, theta_init, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < best_L:
            best_L = result.fun
            best_theta = result.x[0]
    
    return best_L, best_theta

def construct():
    """
    Place 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Returns:
    (inner, center, side, angle) where:
    - inner: list of (x, y, angle_degrees) for each inner hexagon
    - center: (x, y) center of outer hexagon
    - side: side length of outer hexagon
    - angle: rotation angle of outer hexagon in degrees
    """
    def cost_func(params):
        """
        Cost function: minimize the area of the convex hull of the points,
        with a penalty for overlapping hexagons.
        """
        points = params.reshape(-1, 2)
        
        # Center at origin
        centroid = np.mean(points, axis=0)
        pts_centered = points - centroid
        
        # Compute convex hull area
        try:
            if len(pts_centered) < 3:
                hull_area = 0.0
            else:
                hull = ConvexHull(pts_centered)
                hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            hull_area = 100.0  # Large penalty if hull fails
        
        # Compute overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts_centered[i] - pts_centered[j])
                if d < MIN_DIST:
                    overlap_penalty += (MIN_DIST - d) ** 2
        
        # Combine: minimize hull area, penalize overlaps heavily
        return hull_area + 1000.0 * overlap_penalty
    
    # Optimization with multiple restarts
    best_cost = float('inf')
    best_params = None
    
    for restart in range(30):
        if restart == 0:
            # Hexagonal lattice initialization
            pts = []
            for q in range(-3, 4):
                for r in range(-3, 4):
                    x = 1.5 * q
                    y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                    pts.append([x, y])
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists)[:N]
            init_pts = pts[indices]
            init_pts += np.random.normal(0, 0.05, init_pts.shape)
        else:
            init_pts = np.random.uniform(-2.5, 2.5, (N, 2))
        
        initial_params = init_pts.flatten()
        
        # Use L-BFGS-B for better convergence
        res = minimize(cost_func, initial_params, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
        
        # Also try Nelder-Mead for local refinement
        res_nm = minimize(cost_func, initial_params, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res_nm.fun < best_cost:
            best_cost = res_nm.fun
            best_params = res_nm.x
    
    if best_params is None:
        # Fallback: return a valid but large solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract optimized points
    final_pts = best_params.reshape(-1, 2)
    centroid = np.mean(final_pts, axis=0)
    final_pts_centered = final_pts - centroid
    
    # Verify no overlaps
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(final_pts_centered[i] - final_pts_centered[j])
            if d < MIN_DIST - 1e-6:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback if overlaps detected
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Compute minimum enclosing hexagon
    side_length, rotation = compute_min_enclosing_hexagon(final_pts_centered)
    
    # Add small buffer to ensure all hexagons fit
    # Each inner hexagon has radius 1 (distance from center to vertex)
    # The enclosing hexagon needs to account for the extent of each inner hexagon
    # The maximum distance from an inner hexagon center to its vertex is 1
    # So we need to add 1 to the side length
    outer_side = side_length + 1.0
    
    # Construct output
    inner = [[float(x), float(y), 0.0] for x, y in final_pts_centered]
    outer_center = (0.0, 0.0)
    outer_angle = np.degrees(rotation)
    
    return inner, outer_center, outer_side, outer_angle
