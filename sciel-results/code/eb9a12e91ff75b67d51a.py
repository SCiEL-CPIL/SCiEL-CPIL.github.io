# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Strategy:
    1. Generate diverse structural initializations (regular polygons, grids, random perturbations).
    2. Use a robust local optimizer (L-BFGS-B) with a smooth surrogate objective to refine each structure.
    3. Keep track of the best configuration found.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 50s margin from 630s to be safe
    
    # Pre-compute indices for all 286 triangles (i < j < k)
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]

    def calculate_metrics(points):
        """
        Calculate A_min / A_hull and validity.
        Returns (score, A_min, A_hull, is_valid)
        """
        if not np.all(np.isfinite(points)):
            return 0.0, 0.0, 0.0, False
        
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0, 0.0, A_hull, False
        except Exception:
            return 0.0, 0.0, 0.0, False

        # Calculate min triangle area
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        # Cross product magnitude in 2D
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0, A_min, A_hull, True
            
        score = A_min / A_hull
        return score, A_min, A_hull, True

    def objective(points_flat):
        """
        Objective function to minimize (negative score).
        Uses a smooth approximation to help the optimizer.
        """
        points = points_flat.reshape((n, 2))
        score, A_min, A_hull, is_valid = calculate_metrics(points)
        
        if not is_valid:
            return 1e6 # Large penalty for invalid points
            
        # We want to maximize A_min / A_hull.
        # To make it smoother for L-BFGS-B, we can minimize -log(A_min) + log(A_hull) 
        # but log(0) is problematic. 
        # A simpler robust approach: minimize -(A_min / A_hull) directly is non-smooth at min.
        # Instead, let's minimize a penalized version: 
        # Penalty for small areas + normalization.
        
        # Actually, L-BFGS-B handles the non-smoothness of 'min' reasonably well if the 
        # landscape isn't too jagged, but a better surrogate is often:
        # minimize: -log(sum(exp(-areas/T))) / log(sum(exp(-log(A_hull)))) ... complex.
        
        # Let's stick to the direct negative score but add a small epsilon to avoid 0 division
        # and rely on the multi-start nature to escape local minima.
        # However, 'min' is non-differentiable. 
        # Alternative: minimize the sum of squares of (areas - A_min) is not right.
        
        # Best practical approach for Heilbronn:
        # Minimize: A_hull / A_min  (which is 1/score)
        # This is also non-smooth.
        
        # Let's use a soft-min approximation for the objective to be minimized:
        # f = - (1/T) * log( sum( exp(-areas_i / T) ) ) / log(A_hull)
        # As T -> 0, this approaches -A_min / log(A_hull) ... wait, we want A_min/A_hull.
        
        # Let's just use the direct score but ensure the optimizer has a good start.
        # L-BFGS-B uses numerical gradients. The kinks in 'min' are the main issue.
        # Multi-start is the key.
        
        return -score

    def optimize_from_start(start_points, max_iter=200):
        """
        Run L-BFGS-B from a specific starting point.
        """
        if time.time() - start_time > time_limit:
            return None, 0.0
            
        start_flat = start_points.flatten()
        
        # Bounds: Keep points within a reasonable box to avoid drift
        # We can normalize after optimization, but bounds help stability.
        # Let's allow a wide range, say [-10, 10]
        bounds = [(-10.0, 10.0)] * (n * 2)
        
        try:
            res = minimize(
                objective, 
                start_flat, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < 0: # If it found a valid improvement
                best_points = res.x.reshape((n, 2))
                score, _, _, is_valid = calculate_metrics(best_points)
                if is_valid and score > 0:
                    return best_points, score
        except Exception:
            pass
            
        return None, 0.0

    # --- Generate Initial Structures ---
    
    structures = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    regular_13 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    structures.append(regular_13)
    
    # 2. Regular 7-gon + 6 points inside (e.g., another smaller 7-gon or grid)
    # Let's try a hexagon with a center point, plus 6 more?
    # Better: Two concentric polygons.
    # 7 outer, 6 inner?
    # Or 9 outer, 4 inner?
    
    # Try: Regular 9-gon, 4 points in a square inside
    angles9 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    outer9 = np.column_stack([1.0 * np.cos(angles9), 1.0 * np.sin(angles9)])
    # 4 points in a square
    sq_size = 0.5
    square_pts = np.array([
        [-sq_size, -sq_size],
        [sq_size, -sq_size],
        [sq_size, sq_size],
        [-sq_size, sq_size]
    ])
    structures.append(np.vstack([outer9, square_pts]))
    
    # Try: Regular 12-gon + 1 center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer12 = np.column_stack([1.0 * np.cos(angles12), 1.0 * np.sin(angles12)])
    center = np.array([[0.0, 0.0]])
    structures.append(np.vstack([outer12, center]))
    
    # Try: Grid-like structure
    # 4x4 grid minus 3 points?
    # Let's try a distorted grid
    grid_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [1, 3]
    ])
    # Center and scale
    grid_pts = (grid_pts - 1.5) * 1.0
    structures.append(grid_pts)
    
    # 3. Random perturbations of the regular 13-gon
    rng = np.random.default_rng(seed=42)
    for _ in range(10):
        perturbed = regular_13 + rng.normal(0, 0.1, (n, 2))
        structures.append(perturbed)
        
    # 4. Random points in a circle
    for _ in range(10):
        r = rng.uniform(0.5, 1.0, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        structures.append(pts)

    # --- Optimization Loop ---
    
    best_score = 0.0
    best_points = None
    
    # Sort structures by initial score to prioritize good starts?
    # Or just iterate. Let's iterate and refine.
    
    for i, start_pts in enumerate(structures):
        if time.time() - start_time > time_limit - 60: # Reserve 60s for final checks
            break
            
        # Refine this structure
        refined_pts, score = optimize_from_start(start_pts)
        
        if refined_pts is not None and score > best_score:
            best_score = score
            best_points = refined_pts
            
            # If we found a very good score, we might want to spend more time polishing it
            # But for now, we move to the next structure to explore the space.
            
    # --- Final Polish Phase ---
    # If we have time, refine the best point found further with a tighter tolerance
    if best_points is not None and time.time() - start_time < time_limit - 10:
        # Run another optimization round on the best point
        final_pts, final_score = optimize_from_start(best_points, max_iter=100)
        if final_pts is not None and final_score > best_score:
            best_points = final_pts
            best_score = final_score
            
    # --- Normalization ---
    # The score is invariant to scaling and translation, but we should ensure 
    # the points are well-scaled for the evaluator (though it shouldn't matter 
    # as long as they are finite and hull is valid).
    # Let's normalize to center at origin and max radius 1.
    if best_points is not None:
        centroid = np.mean(best_points, axis=0)
        best_points = best_points - centroid
        max_dist = np.max(np.linalg.norm(best_points, axis=1))
        if max_dist > 0:
            best_points = best_points / max_dist
            
        # Final verification
        final_score, _, _, is_valid = calculate_metrics(best_points)
        if is_valid and final_score > 0:
            return best_points
    
    # Fallback: Return a regular 13-gon if everything failed
    return regular_13

# EVOLVE-BLOCK-END
