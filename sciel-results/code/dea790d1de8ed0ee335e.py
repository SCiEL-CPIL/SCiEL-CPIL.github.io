import numpy as np
import time
from scipy.optimize import differential_evolution

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute pair indices for vectorized operations
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Bounds for optimization variables:
    # 0..N-1: x coordinates (0, 1)
    # N..2N-1: y coordinates (0, 1)
    # 2N..3N-1: radii (0, 0.5)
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    
    def penalty_function(params):
        """
        Minimize the penalty.
        penalty = -sum(r) + lambda * (overlap_penalty + boundary_penalty)
        We want to maximize sum(r), so minimizing -sum(r).
        The penalty terms enforce constraints.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        obj = -np.sum(r)
        
        # Constraints:
        # 1. Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
        # Violations:
        # vx1 = r - x (should be <= 0)
        # vx2 = x + r - 1 (should be <= 0)
        # vy1 = r - y
        # vy2 = y + r - 1
        
        # Use quadratic penalty for smoothness
        lambda_pen = 1000.0
        
        # Boundary penalties
        v_x_left = np.maximum(0, r - x)
        v_x_right = np.maximum(0, x + r - 1)
        v_y_bot = np.maximum(0, r - y)
        v_y_top = np.maximum(0, y + r - 1)
        
        boundary_pen = np.sum(v_x_left**2) + np.sum(v_x_right**2) + \
                       np.sum(v_y_bot**2) + np.sum(v_y_top**2)
        
        # 2. Pairwise: dist(i, j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist_sq = dx*dx + dy*dy
        
        r_sum = r[i_idx] + r[j_idx]
        r_sum_sq = r_sum * r_sum
        
        # Violation: r_sum_sq - dist_sq (should be <= 0)
        v_pair = np.maximum(0, r_sum_sq - dist_sq)
        
        pair_pen = np.sum(v_pair**2)
        
        total_pen = boundary_pen + pair_pen
        
        return obj + lambda_pen * total_pen

    # Initial population: random
    # differential_evolution handles initialization, but we can provide a seed if needed.
    # It uses random initialization by default.
    
    try:
        # maxiter: number of generations. 
        # popsize: multiple of population size.
        # tol: relative tolerance.
        # We want to use most of the time.
        # DE can be slow. Let's limit maxiter and rely on the deadline check inside?
        # DE doesn't easily support early stopping callbacks in all versions, 
        # but we can wrap it or just let it run and catch timeout?
        # Actually, scipy's DE doesn't have a built-in time limit.
        # We can estimate: 
        # Each function evaluation is fast (vectorized).
        # Let's try a moderate number of iterations.
        
        # To be safe with time, we can run DE with a fixed number of iterations
        # and then do a local polish if time permits?
        # Or just run DE with maxiter=50 and popsize=10 (500 evals).
        # 500 evals * 26 circles * 325 pairs = ~4M ops. Should be fast (<1s per 100 evals?).
        # Let's try maxiter=100, popsize=15.
        
        result = differential_evolution(
            penalty_function, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            seed=42,
            tol=1e-8,
            init='sobol' # Better sampling than random
        )
        
        best_params = result.x
    except Exception:
        # Fallback to a simple grid if DE fails or times out?
        # For now, return a valid simple config
        x = np.full(N, 0.5)
        y = np.full(N, 0.5)
        # This will be invalid. We need a valid fallback.
        # Simple grid:
        side = 5
        indices = 0
        for i in range(side):
            for j in range(side):
                if indices < N:
                    x[indices] = (i + 0.5) / side
                    y[indices] = (j + 0.5) / side
                    indices += 1
        r = np.full(N, 0.5 / side * 0.9)
        best_params = np.concatenate([x, y, r])

    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:]
    
    # Post-processing: Ensure strict validity
    # The penalty function minimizes violations, but might not be zero.
    # We need to project to the feasible set.
    
    # 1. Boundary projection
    # For each circle, r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    r = np.minimum(r, np.minimum(x, 1-x))
    r = np.minimum(r, np.minimum(y, 1-y))
    
    # 2. Pairwise projection
    # Iteratively shrink radii to resolve overlaps.
    # This is a conservative projection.
    for _ in range(50):
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1e-9)
        
        # Required: r_i + r_j <= dist
        # Current: r_i + r_j
        # If r_i + r_j > dist, we have a violation.
        # We can shrink both.
        # Let's find the max violation.
        violations = r[i_idx] + r[j_idx] - dist
        max_viol = np.max(violations)
        
        if max_viol < 1e-9:
            break
            
        # Shrink all radii slightly to fix the worst violation
        # A more precise method would adjust specific pairs, but global shrink is safe.
        # Scale factor: (dist - eps) / (r_i + r_j) for the worst pair?
        # Simple global shrink:
        scale = 0.95
        r *= scale
        
        # Re-apply boundary constraints after shrink (shrink doesn't violate boundaries if r was valid, but just in case)
        r = np.minimum(r, np.minimum(x, 1-x))
        r = np.minimum(r, np.minimum(y, 1-y))
        
    # Final check
    # Verify one last time
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx*dx + dy*dy)
    if np.any(r[i_idx] + r[j_idx] > dist + 1e-9):
        # If still invalid, shrink more aggressively
        for _ in range(100):
            dist = np.sqrt((x[i_idx] - x[j_idx])**2 + (y[i_idx] - y[j_idx])**2)
            if np.max(r[i_idx] + r[j_idx] - dist) < 1e-9:
                break
            r *= 0.9
            r = np.minimum(r, np.minimum(x, 1-x))
            r = np.minimum(r, np.minimum(y, 1-y))

    final_state = np.column_stack([x, y, r])
    return final_state

# EVOLVE-BLOCK-END
