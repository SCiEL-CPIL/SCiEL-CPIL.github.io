# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 50s margin from 630s to be safe

    # Pre-compute indices for all 286 triangles (i < j < k)
    # This is done once to save time in the inner loop
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]

    def calculate_score(points_flat):
        """
        Calculate A_min / A_hull. Returns negative for invalid or 0.
        """
        points = points_flat.reshape((n, 2))
        
        # Check for valid points (finite)
        if not np.all(np.isfinite(points)):
            return 0.0

        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return 0.0
        except Exception:
            return 0.0

        # Calculate min triangle area
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        # Cross product magnitude in 2D: |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)| / 2
        # Or 0.5 * | (p2-p1) x (p3-p1) |
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 0:
            return 0.0
            
        return A_min / A_hull

    def neg_objective(points_flat):
        """
        Negated objective for minimization.
        """
        score = calculate_score(points_flat)
        if score == 0.0:
            return 1e6
        return -score

    # Strategy: Multi-start local optimization
    # 1. Start with a regular 13-gon inscribed in a unit circle (centered at 0.5, 0.5)
    # 2. Run several local optimizations (L-BFGS-B) from slightly perturbed versions of this start
    # 3. Keep the best result

    best_points = None
    best_score = -np.inf
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 0.5
    cx, cy = 0.5, 0.5
    init_x = cx + radius * np.cos(angles)
    init_y = cy + radius * np.sin(angles)
    init_points = np.column_stack((init_x, init_y))
    
    # Scale to [0, 1] box roughly, though L-BFGS-B will handle bounds
    # Actually, let's just use the regular polygon and let the optimizer move it.
    # We need to ensure it stays within a valid convex region. 
    # The problem doesn't restrict points to [0,1], just that they form a convex hull.
    # However, scaling invariance means we can fix the hull size or just let it float.
    # A_min / A_hull is scale invariant. 
    # To help the optimizer, we can normalize the points in the objective? 
    # No, the objective is already scale invariant. 
    # But local optimizers struggle with scale invariance (flat directions).
    # Let's fix the centroid to (0,0) and normalize the max coordinate to 1? 
    # Or just optimize in raw coordinates and accept that the scale might drift.
    # Better: Constrain the points to lie within a unit square [0,1]x[0,1] to break scale invariance 
    # and provide a bounded domain for L-BFGS-B.
    
    bounds = [(0.0, 1.0)] * (n * 2)
    
    # Generate multiple starting points
    rng = np.random.default_rng(42)
    
    # Start 1: Regular polygon
    starts = [init_points.flatten()]
    
    # Start 2-10: Perturbed regular polygons
    for _ in range(9):
        perturbed = init_points + rng.normal(0, 0.1, (n, 2))
        # Clip to [0, 1] to keep them in bounds for the optimizer
        perturbed = np.clip(perturbed, 0.0, 1.0)
        starts.append(perturbed.flatten())
        
    # Start 11-20: Random points (to escape local minima of polygon structure)
    for _ in range(10):
        random_pts = rng.random((n, 2))
        starts.append(random_pts.flatten())

    for i, start in enumerate(starts):
        if time.time() - start_time > time_limit:
            break
            
        try:
            # Use L-BFGS-B for local optimization
            result = minimize(
                neg_objective, 
                start, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            current_points = result.x.reshape((n, 2))
            current_score = calculate_score(result.x)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points
                
        except Exception:
            continue

    # If best_points is still None or invalid, fallback to the initial regular polygon
    if best_points is None or not np.all(np.isfinite(best_points)):
         # Try to calculate score of init_points just in case
         init_score = calculate_score(init_points.flatten())
         if init_score > 0:
             best_points = init_points
         else:
             # Absolute fallback: random
             best_points = rng.random((n, 2))
             
    return best_points

# EVOLVE-BLOCK-END
