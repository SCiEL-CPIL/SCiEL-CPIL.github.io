"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses Simulated Annealing to explore the configuration space.
    """
    n = N
    time_limit = 110.0 # Leave buffer for overhead
    start_time = time.time()
    
    # Helper function to calculate cost
    def calculate_cost(x, y, r, penalty_weight):
        # Boundary violations
        bound_viol = np.maximum(0, r - x) + np.maximum(0, x + r - 1) + \
                     np.maximum(0, r - y) + np.maximum(0, y + r - 1)
        bound_pen = np.sum(bound_viol ** 2)
        
        # Overlap violations
        # Vectorized calculation of pairwise distances
        # X, Y, R are 1D arrays
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        overlap = np.maximum(0, min_dist - dist)
        # Only count upper triangle to avoid double counting
        overlap_pen = np.sum(overlap[~np.eye(n, dtype=bool)] ** 2)
        
        penalty = bound_pen + overlap_pen
        objective = -np.sum(r) + penalty_weight * penalty
        return objective, penalty

    # Helper function to check if a configuration is valid
    def is_valid(x, y, r, tol=1e-5):
        if np.any(r < 0):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        overlap = min_dist - dist
        if np.any(overlap[~np.eye(n, dtype=bool)] > tol):
            return False
        return True

    # Initialize with a grid
    side = int(np.ceil(np.sqrt(n)))
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx < n:
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                r[idx] = 0.5 / side * 0.9 # Slightly smaller to allow movement
                idx += 1
    
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_val = np.sum(best_r)
    
    T = 1.0
    T_min = 1e-5
    alpha = 0.995 # Cooling rate
    penalty_weight = 100.0
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Perturb a random circle
        i = np.random.randint(n)
        
        # Move
        x_new = x.copy()
        y_new = y.copy()
        r_new = r.copy()
        
        # Gaussian perturbation
        step_size = 0.01
        x_new[i] += np.random.normal(0, step_size)
        y_new[i] += np.random.normal(0, step_size)
        
        # Radius perturbation
        r_step = 0.005
        r_new[i] += np.random.normal(0, r_step)
        r_new[i] = max(r_new[i], 1e-4)
        
        # Calculate costs
        cost_old, pen_old = calculate_cost(x, y, r, penalty_weight)
        cost_new, pen_new = calculate_cost(x_new, y_new, r_new, penalty_weight)
        
        delta_cost = cost_new - cost_old
        
        # Acceptance probability
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            x, y, r = x_new, y_new, r_new
            
            # Check if valid and better
            if is_valid(x, y, r):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
        
        # Cool down
        T *= alpha
        
        # Dynamically adjust penalty weight: decrease it as we get closer to valid solutions
        # to allow more aggressive radius expansion
        if pen_old < 1e-6:
             penalty_weight *= 0.99

    # Final check and cleanup
    if is_valid(best_x, best_y, best_r):
        return np.column_stack([best_x, best_y, best_r])
    
    # Fallback: return the initial grid if something went wrong
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
