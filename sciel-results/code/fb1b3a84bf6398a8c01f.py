"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import dual_annealing
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (task says 120s eval, but 150s kill switch noted in prompt, use 115s to be safe for 120s constraint)
    
    def get_exact_covering_radius(pts):
        """
        Computes the exact covering radius using the convex hull.
        Returns (radius_in_radians, worst_point_vector).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Fallback if hull fails (e.g., points are coplanar or coincident)
            # This shouldn't happen with random init and normalization, but just in case.
            # Return a large value to penalize invalid states.
            return np.pi, np.array([1.0, 0.0, 0.0])
        
        max_radius = 0.0
        worst_center = None
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            # Check if the triangle is degenerate
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            # Ensure c is on the correct side (pointing outwards)
            if np.dot(c, v1) < 0:
                c = -c
                
            # Calculate angular distance from c to v1
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            if radius > max_radius:
                max_radius = radius
                worst_center = c
                
        if worst_center is None:
            # Fallback: use a random point or grid estimate
            return np.pi, np.array([1.0, 0.0, 0.0])
            
        return max_radius, worst_center

    def soft_max_covering_radius(pts_flat, beta=50.0):
        """
        Objective function for dual_annealing.
        Takes a flat array of 3*N coordinates, reshapes to (N,3), normalizes,
        and computes a smooth approximation of the covering radius.
        
        We use a grid of test points on the sphere. The soft-max approximates
        the maximum distance to the nearest configuration point.
        """
        # Reshape and normalize
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # Generate a fixed grid of test points (created outside or cached? 
        # For simplicity in this closure, we create a moderate grid.
        # To speed up, we can use a smaller grid for the annealing phase.
        # 20000 points is a good balance.
        # Fibonacci spiral for test points
        M = 20000
        i = np.arange(M) + 0.5
        z_test = 1.0 - 2.0 * i / M
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i
        r_test = np.sqrt(np.clip(1.0 - z_test * z_test, 0, 1.0))
        test_pts = np.column_stack([
            r_test * np.cos(phi_test),
            r_test * np.sin(phi_test),
            z_test
        ])
        
        # Calculate distances from each test point to all N configuration points
        # dist_sq[i, j] = ||test_pts[i] - pts[j]||^2
        # We want the min over j for each i.
        # Using broadcasting: (M, 1, 3) - (1, N, 3) -> (M, N, 3)
        # This creates a large array (20000 * 50 * 3) ~ 3M floats ~ 24MB. Acceptable.
        diffs = test_pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs * diffs, axis=2)
        
        # Min distance squared for each test point
        min_dists_sq = np.min(dists_sq, axis=1)
        
        # Convert squared Euclidean distance to angular distance
        # d_eucl = sqrt(min_dists_sq)
        # angle = 2 * arcsin(d_eucl / 2)
        # To avoid sqrt and arcsin overhead in the inner loop, we can optimize the objective
        # or just compute it. Since M=20000 and N=50, this is fast enough.
        d_eucl = np.sqrt(min_dists_sq)
        angles = 2.0 * np.arcsin(np.clip(d_eucl / 2.0, 0.0, 1.0))
        
        # Soft max
        # max(angles) approximated by (1/beta) * log(sum(exp(beta * angles)))
        # To prevent overflow, shift by max
        max_angle = np.max(angles)
        if max_angle == 0:
            return 0.0
            
        # Stable soft max
        exp_terms = np.exp(beta * (angles - max_angle))
        soft_max_val = max_angle + np.log(np.sum(exp_terms)) / beta
        
        return soft_max_val

    # 1. Initial Guess: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimizer
    x0 = init_points.flatten()
    
    # 2. Global Optimization with Dual Annealing
    # Bounds: Each coordinate between -1 and 1. Normalization is handled in objective.
    bounds = [(-1.0, 1.0)] * (3 * N)
    
    best_result = None
    best_soft_score = np.inf
    best_points = init_points.copy()
    
    # Run dual_annealing with time limit
    # maxiter is not the primary constraint, time is.
    try:
        res = dual_annealing(
            soft_max_covering_radius,
            bounds=bounds,
            x0=x0,
            maxiter=100, # Limit iterations to avoid infinite loops
            seed=42,
            maxfun=5000, # Limit function evaluations
            # callback not supported in all versions or needed if we track internally
        )
        
        if res.fun < best_soft_score:
            best_soft_score = res.fun
            best_points = res.x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            best_points = best_points / norms
            
    except Exception as e:
        # If optimization fails, stick with initial points
        pass

    # 3. Local Refinement
    # The soft-max grid approximation might miss the exact worst-case facet.
    # We perform a local search on the exact covering radius.
    # Strategy: Identify the worst facet, move its vertices slightly towards the circumcenter.
    
    current_points = best_points.copy()
    
    # Local improvement loop
    step_size = 0.05
    for iter in range(50):
        if time.time() > deadline - 5:
            break
            
        radius, worst_center = get_exact_covering_radius(current_points)
        
        # Find the facet corresponding to worst_center
        try:
            hull = ConvexHull(current_points)
        except:
            break
            
        target_indices = None
        for simplex in hull.simplices:
            v1 = current_points[simplex[0]]
            v2 = current_points[simplex[1]]
            v3 = current_points[simplex[2]]
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c
            # Check if this is the worst center
            if np.dot(c, worst_center) > 0.9999:
                target_indices = simplex
                break
                
        if target_indices is None:
            break
            
        # Move vertices of the worst facet towards the circumcenter
        moved = False
        for idx in target_indices:
            v = current_points[idx]
            dir_vec = worst_center - v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
                new_v = v + step_size * dir_vec
                new_v /= np.linalg.norm(new_v)
                # Check if this move improves the exact radius
                old_pts = current_points.copy()
                current_points[idx] = new_v
                new_radius, _ = get_exact_covering_radius(current_points)
                
                if new_radius < radius:
                    moved = True
                else:
                    current_points[idx] = old_pts[idx]
        
        if not moved:
            # Try smaller step
            step_size *= 0.5
            if step_size < 0.001:
                break
        else:
            # Reset step size if we made progress, or keep it
            step_size = max(step_size * 0.9, 0.01)
            
        # Update best if improved
        if radius < best_soft_score: # Note: best_soft_score is in radians too (approx)
            # Actually, let's track exact best separately
            pass

    # Final check: Compare with initial points just in case
    init_radius, _ = get_exact_covering_radius(init_points)
    final_radius, _ = get_exact_covering_radius(current_points)
    
    if final_radius <= init_radius:
        best_points = current_points
    else:
        best_points = init_points

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
