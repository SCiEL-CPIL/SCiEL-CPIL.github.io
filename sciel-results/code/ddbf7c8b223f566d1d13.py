# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest 
    triangle area to the convex hull area.
    
    Strategy:
    1. Exploit scale and translation invariance: Fix point 0 at (0,0) and point 1 at (1,0).
       This reduces the search space from 26 to 24 dimensions and allows us to 
       normalize the convex hull area implicitly or calculate it once at the end.
       Actually, since the metric is A_min / A_hull, and both scale by L^2, the ratio is invariant.
       However, A_hull depends on the relative positions. 
       To avoid calling ConvexHull inside the expensive optimization loop, we can:
       - Use a fixed reference frame.
       - Wait, A_hull is not constant under translation/scale if we don't normalize?
       Yes, A_min scales with L^2, A_hull scales with L^2. Ratio is invariant.
       But we still need A_hull to compute the score.
       Calling ConvexHull is O(N log N) but N=13 is tiny. The bottleneck is usually the optimizer iterations.
       
       A better trick: Since the metric is invariant, we can fix the convex hull area to 1.0?
       No, the hull shape changes.
       
       Let's stick to fixing 2 points to remove translation/rotation/scale degrees of freedom?
       Translation: Fix centroid at 0? Or fix p0=(0,0).
       Scale: Fix distance p0-p1 = 1.
       Rotation: Fix p1=(1,0).
       
       This removes 4 degrees of freedom (2 trans, 1 rot, 1 scale).
       Remaining: 13*2 - 4 = 22 dimensions.
       
       Inside the objective function, we still need to compute A_hull.
       However, we can pre-compute the triangle indices.
       
       Key improvement: Use Differential Evolution (global) instead of L-BFGS-B (local).
       DE is robust to local minima and doesn't require gradients.
       Also, we can use a "smooth min" surrogate that is differentiable to help DE? 
       DE is derivative-free, so we can use the raw min or a very sharp soft-min.
       Raw min is non-differentiable but DE handles it fine.
       
       To speed up evaluation:
       - Vectorize triangle area calculation.
       - Avoid ConvexHull if possible? 
         Actually, for 13 points, ConvexHull is very fast (<1ms).
         The main cost is the number of function evaluations.
         DE with population 30, maxiter 200 = 6000 evals.
         Each eval: 286 triangle areas + 1 ConvexHull.
         This should be fast enough within 600s.
         
       Let's implement:
       1. Fixed frame: p0=(0,0), p1=(1,0).
       2. Variables: p2...p12 (22 variables).
       3. Objective: Minimize - (A_min / A_hull).
       4. Bounds: Keep points within a reasonable box, e.g., [-5, 5] to prevent drift, 
          but since scale is fixed by p0-p1, the hull area is constrained.
          If points fly away, A_hull grows, A_min might grow too, but ratio might drop.
          We should constrain the hull to be not too large?
          Actually, if we fix p0, p1, the scale is fixed. 
          If other points go to infinity, A_hull -> inf, A_min -> inf? 
          Ratio might approach 0 or a limit.
          We need to ensure points stay "compact".
          Let's bound coordinates to [-10, 10].
          
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Safety margin
    
    # Pre-compute triangle indices
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]

    def calculate_metrics_full(points):
        """Calculate A_min / A_hull for a full (13,2) array."""
        if not np.all(np.isfinite(points)):
            return 0.0, 0.0, 0.0, False
        
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0, 0.0, A_hull, False
        except Exception:
            return 0.0, 0.0, 0.0, False

        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0, A_min, A_hull, True
            
        score = A_min / A_hull
        return score, A_min, A_hull, True

    def objective_reduced(x):
        """
        Objective for Differential Evolution.
        x: array of shape (22,) representing p2..p12.
        Returns: - (A_min / A_hull)
        """
        # Reconstruct full points
        points = np.zeros((n, 2))
        points[0] = [0.0, 0.0]
        points[1] = [1.0, 0.0]
        points[2:] = x.reshape((-1, 2))
        
        # Quick validity check
        if not np.all(np.isfinite(points)):
            return 1e6
            
        # Calculate Hull Area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 1e6
        except Exception:
            return 1e6

        # Calculate Min Triangle Area
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 1e6
            
        score = A_min / A_hull
        return -score

    # Define bounds for the 22 variables (11 points * 2 coords)
    # Since p0, p1 are fixed, the "scale" is set. 
    # We want to allow points to be in a reasonable range.
    # If the optimal configuration is large, it will be scaled down by the invariance, 
    # but since we fixed p0, p1, we are effectively searching in a normalized space.
    # Bounds: [-5, 5] should be sufficient for a compact convex region.
    bounds = [(-5.0, 5.0)] * 22
    
    best_points_full = None
    best_score = 0.0

    # Strategy: Run Differential Evolution with multiple seeds/populations if time permits
    # DE is stochastic. We can run it for a fixed number of iterations or until time limit.
    
    # Initial run: Moderate population, moderate iterations
    # popsize=20 means 20*22=440 params? No, popsize is number of individuals.
    # Let's use popsize=30.
    
    # We need to check time inside the objective? No, DE doesn't support that easily.
    # We will run DE with a maxiter that fits within time.
    # Estimate time per eval: ~0.5ms (guess). 30 pop * 200 iter = 6000 evals * 0.5ms = 3s.
    # This is very fast. We can afford more iterations or multiple restarts.
    
    rng = np.random.default_rng(42)
    
    # Run 1: Standard DE
    try:
        res = differential_evolution(
            objective_reduced, 
            bounds, 
            maxiter=300, 
            popsize=30, 
            tol=1e-8, 
            seed=42,
            polish=False # We will polish manually
        )
        
        x_opt = res.x
        points_opt = np.zeros((n, 2))
        points_opt[0] = [0.0, 0.0]
        points_opt[1] = [1.0, 0.0]
        points_opt[2:] = x_opt.reshape((-1, 2))
        
        score, _, _, is_valid = calculate_metrics_full(points_opt)
        if is_valid and score > best_score:
            best_score = score
            best_points_full = points_opt
            
    except Exception as e:
        print(f"DE Run 1 failed: {e}")

    # Run 2: DE with different seed and larger popsize if time allows
    if time.time() - start_time < time_limit - 60:
        try:
            res2 = differential_evolution(
                objective_reduced, 
                bounds, 
                maxiter=200, 
                popsize=50, 
                tol=1e-8, 
                seed=123,
                polish=False
            )
            
            x_opt2 = res2.x
            points_opt2 = np.zeros((n, 2))
            points_opt2[0] = [0.0, 0.0]
            points_opt2[1] = [1.0, 0.0]
            points_opt2[2:] = x_opt2.reshape((-1, 2))
            
            score2, _, _, is_valid2 = calculate_metrics_full(points_opt2)
            if is_valid2 and score2 > best_score:
                best_score = score2
                best_points_full = points_opt2
                
        except Exception as e:
            print(f"DE Run 2 failed: {e}")

    # Run 3: If we have time, do a local polish on the best result found by DE
    # Using L-BFGS-B on the full 26-dim space (or reduced 22-dim) from the DE result
    if best_points_full is not None and time.time() - start_time < time_limit - 30:
        try:
            from scipy.optimize import minimize
            
            # Define objective for polish (reduced space)
            def obj_polish(x):
                points = np.zeros((n, 2))
                points[0] = [0.0, 0.0]
                points[1] = [1.0, 0.0]
                points[2:] = x.reshape((-1, 2))
                
                if not np.all(np.isfinite(points)):
                    return 1e6
                try:
                    hull = ConvexHull(points)
                    A_hull = hull.volume
                    if A_hull <= 1e-10:
                        return 1e6
                except Exception:
                    return 1e6

                p1 = points[i_idx]
                p2 = points[j_idx]
                p3 = points[k_idx]
                cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
                areas = 0.5 * np.abs(cross)
                A_min = np.min(areas)
                if A_min <= 1e-10:
                    return 1e6
                return -(A_min / A_hull)
            
            # Start from the DE result
            start_x = best_points_full[2:].flatten()
            
            res_polish = minimize(
                obj_polish, 
                start_x, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            x_polished = res_polish.x
            points_polished = np.zeros((n, 2))
            points_polished[0] = [0.0, 0.0]
            points_polished[1] = [1.0, 0.0]
            points_polished[2:] = x_polished.reshape((-1, 2))
            
            score_polish, _, _, is_valid_polish = calculate_metrics_full(points_polished)
            if is_valid_polish and score_polish > best_score:
                best_score = score_polish
                best_points_full = points_polished
                
        except Exception as e:
            print(f"Polish failed: {e}")

    # Final Normalization and Return
    # The metric is invariant, so we can return the points as is.
    # However, to be safe and clean, we can re-center and scale if needed, 
    # but since p0=0, p1=1, it's already in a canonical form.
    # Just ensure finite values.
    
    if best_points_full is not None and np.all(np.isfinite(best_points_full)):
        return best_points_full
    
    # Fallback: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    regular_13 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return regular_13

# EVOLVE-BLOCK-END
