import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Variables: x0, y0, r0, x1, y1, r1, ... x25, y25, r25
    # Total 3 * N = 78 variables
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    def objective(v):
        # v is a flat array of length 3*N
        # Reshape to (N, 3)
        circles = v.reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Penalties
        # 1. Boundary violations
        # x < r  -> r - x > 0
        # x > 1-r -> x - (1-r) > 0
        pen_x_low = np.sum(np.maximum(0, r - x)**2)
        pen_x_high = np.sum(np.maximum(0, x - (1 - r))**2)
        pen_y_low = np.sum(np.maximum(0, r - y)**2)
        pen_y_high = np.sum(np.maximum(0, y - (1 - r))**2)
        
        # 2. Overlap violations
        # dist_sq < (ri + rj)^2
        # dist_sq = (xi-xj)^2 + (yi-yj)^2
        # We can compute pairwise distances
        # For efficiency, use broadcasting
        # x_diff = x[:, None] - x[None, :]
        # dist_sq = x_diff**2 + y_diff**2
        # sum_r_ij = r[:, None] + r[None, :]
        # overlap = np.maximum(0, sum_r_ij**2 - dist_sq)
        # pen_overlap = np.sum(overlap**2)
        
        # To avoid N^2 cost in every eval if possible? 
        # N=26 is small, N^2 = 676. 78 variables. 
        # 676 operations per eval is fine.
        
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r = r[:, None] + r[None, :]
        # Only consider i < j to avoid double counting and diagonal
        # But np.maximum(0, ...) on full matrix is easier, just divide by 2 or ignore diagonal
        # Diagonal is 0 distance, sum_r = 2*ri, so (2ri)^2 - 0 > 0 always? 
        # Wait, dist_sq[i,i] = 0. sum_r[i,i] = 2*ri. 
        # (2ri)^2 - 0 = 4*ri^2. This is a huge penalty for self-overlap? 
        # No, self-overlap is not a constraint. We only care about i != j.
        # We should mask the diagonal.
        
        mask = ~np.eye(N, dtype=bool)
        # We can just sum over the upper triangle or use the mask
        overlap_terms = np.maximum(0, sum_r**2 - dist_sq)
        # Zero out diagonal
        np.fill_diagonal(overlap_terms, 0)
        pen_overlap = np.sum(overlap_terms**2)
        
        total_penalty = pen_x_low + pen_x_high + pen_y_low + pen_y_high + pen_overlap
        return -sum_r + 10.0 * total_penalty

    # Run Differential Evolution
    # popsize=10, maxiter=100, tol=1e-6
    # 78 vars * 10 pop = 780 evals per gen? No, popsize is multiple of dim? 
    # scipy DE popsize is a float, actual pop = int(popsize * dim)
    # Let's try popsize=5, maxiter=50
    # Time check: 50 * 5 * 78 = 19500 evals. 
    # Each eval is fast. Should be well under 140s.
    
    try:
        result_de = differential_evolution(
            objective, 
            bounds, 
            popsize=5, 
            maxiter=50, 
            tol=1e-6, 
            seed=42,
            polish=False # We will do our own polish
        )
    except Exception as e:
        # Fallback to grid if DE fails
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    best_v = result_de.x
    best_val = result_de.fun
    
    # Refine with L-BFGS-B
    # Use a smoother penalty for local optimization
    def objective_smooth(v):
        circles = v.reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        sum_r = np.sum(r)
        
        # Smooth penalties
        # Using a quadratic penalty is already smooth, but we can adjust weight
        pen_x_low = np.sum(np.maximum(0, r - x)**2)
        pen_x_high = np.sum(np.maximum(0, x - (1 - r))**2)
        pen_y_low = np.sum(np.maximum(0, r - y)**2)
        pen_y_high = np.sum(np.maximum(0, y - (1 - r))**2)
        
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r[:, None] + r[None, :]
        overlap_terms = np.maximum(0, sum_r_mat**2 - dist_sq)
        np.fill_diagonal(overlap_terms, 0)
        pen_overlap = np.sum(overlap_terms**2)
        
        total_penalty = pen_x_low + pen_x_high + pen_y_low + pen_y_high + pen_overlap
        return -sum_r + 1.0 * total_penalty # Lower weight for local refine? 
        # Actually, for L-BFGS-B, we want to satisfy constraints strictly.
        # A higher weight ensures constraints are met, but might lead to smaller radii if stuck.
        # Let's keep weight 10.0 for consistency or increase it.
        return -sum_r + 10.0 * total_penalty

    # Check time
    if time.time() < deadline - 20:
        try:
            result_local = minimize(
                objective_smooth, 
                best_v, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_local.fun < best_val:
                best_v = result_local.x
                best_val = result_local.fun
        except Exception:
            pass

    # Final polish with very tight tolerance
    if time.time() < deadline - 5:
        try:
            result_polish = minimize(
                objective_smooth, 
                best_v, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result_polish.fun < best_val:
                best_v = result_polish.x
                best_val = result_polish.fun
        except Exception:
            pass

    # Extract result
    circles = best_v.reshape(N, 3)
    
    # Ensure validity: clip to bounds and ensure no overlaps (best effort)
    # The optimizer should have handled it, but let's be safe
    # If penalty is non-zero, we might have slight violations.
    # We can do a simple repair: shrink radii slightly if needed?
    # Or just return as is, the evaluator checks validity.
    # "an invalid result ... scores 0".
    # We must ensure disjoint interiors.
    
    # Check for violations
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Boundary check
    # If x < r, set x = r? No, that might cause overlap.
    # Better to shrink r if it violates boundary.
    # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    r_min_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    r = np.minimum(r, r_min_bound)
    
    # Overlap check
    # If overlap, shrink radii.
    # This is a heuristic repair.
    # A more robust way: run a few iterations of a simple gradient descent on radii only, 
    # holding x,y fixed, to satisfy overlaps?
    # Or just shrink all radii by a small factor if any overlap is detected.
    
    # Let's check overlaps
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    dist_sq = x_diff**2 + y_diff**2
    sum_r_mat = r[:, None] + r[None, :]
    
    # Check if (ri+rj)^2 > dist_sq
    # We need dist >= ri+rj
    # If not, we have a problem.
    
    # Simple repair: scale down radii uniformly until no overlaps?
    # This might reduce the score significantly if overlaps are large.
    # But the optimizer should have minimized overlaps.
    
    # Let's do a quick iterative shrink if necessary.
    max_iter_repair = 10
    for _ in range(max_iter_repair):
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r[:, None] + r[None, :]
        
        # Find max violation
        # violation = (ri+rj)^2 - dist_sq
        # We want violation <= 0
        # If violation > 0, we need to reduce ri or rj.
        # Simplest: scale all r by (1 - epsilon)
        
        # Check if any violation
        # dist = sqrt(dist_sq)
        # dist_safe = np.where(dist_sq > 0, np.sqrt(dist_sq), np.inf)
        # required_r_sum = dist_safe
        # current_r_sum = sum_r_mat
        # If current_r_sum > dist_safe, overlap.
        
        # Let's just check if penalty is zero (or very small)
        overlap_terms = np.maximum(0, sum_r_mat**2 - dist_sq)
        np.fill_diagonal(overlap_terms, 0)
        pen_overlap = np.sum(overlap_terms**2)
        
        if pen_overlap < 1e-10:
            break
        
        # Shrink radii
        r *= 0.99
        # Re-check boundary
        r_min_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, r_min_bound)
        
    circles[:, 2] = r
    return circles
