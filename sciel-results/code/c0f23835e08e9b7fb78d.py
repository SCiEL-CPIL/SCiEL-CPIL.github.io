import numpy as np
import random

N = 26

# EVOLVE-BLOCK-START
def construct():
    # Parameters for Simulated Annealing
    n_iter = 50000
    T_start = 0.01
    T_end = 1e-6
    alpha = (T_end / T_start) ** (1.0 / n_iter)
    
    # Initial configuration: Random
    # Ensure initial radii are small enough to likely not overlap significantly
    # or start with a grid?
    # Let's try a grid initialization for better start
    # 5x5 grid = 25 circles.
    # Add 1 more.
    
    # Grid positions
    xs = []
    ys = []
    side = 5
    for i in range(side):
        for j in range(side):
            xs.append((i + 0.5) / side)
            ys.append((j + 0.5) / side)
            
    # Add 26th circle
    # Place it in a gap, e.g., center of a 2x2 block
    # Or just random
    xs.append(0.5)
    ys.append(0.1) # Near bottom edge, between x=0.3 and 0.7?
    
    # Actually, let's just use random for simplicity and robustness,
    # but initialize radii small.
    xs = np.random.uniform(0.1, 0.9, N)
    ys = np.random.uniform(0.1, 0.9, N)
    rs = np.full(N, 0.05)
    
    x = xs.copy()
    y = ys.copy()
    r = rs.copy()
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def get_cost(x, y, r):
        # Calculate overlaps
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        
        # Overlap penalty: sum of squared overlaps
        overlaps = np.maximum(0, sum_r - dist)
        overlap_penalty = np.sum(overlaps * overlaps)
        
        # Boundary penalty
        bx = np.maximum(0, r - x) + np.maximum(0, x + r - 1)
        by = np.maximum(0, r - y) + np.maximum(0, y + r - 1)
        boundary_penalty = np.sum(bx*bx + by*by)
        
        total_penalty = overlap_penalty + boundary_penalty
        
        # Cost = -sum(r) + penalty
        cost = -np.sum(r) + total_penalty
        return cost
    
    current_cost = get_cost(x, y, r)
    best_cost = current_cost
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    T = T_start
    step_size = 0.05
    
    for i in range(n_iter):
        idx = np.random.randint(N)
        old_x = x[idx]
        old_y = y[idx]
        old_r = r[idx]
        
        new_x = old_x + np.random.normal(0, step_size)
        new_y = old_y + np.random.normal(0, step_size)
        new_r = old_r + np.random.normal(0, step_size * 0.5)
        new_r = max(new_r, 0.01)
        
        x[idx] = new_x
        y[idx] = new_y
        r[idx] = new_r
        
        new_cost = get_cost(x, y, r)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        else:
            x[idx] = old_x
            y[idx] = old_y
            r[idx] = old_r
            
        T *= alpha
        
    # Final validity check and adjustment
    # Shrink radii if any constraint is violated
    for _ in range(1000):
        dx = best_x[i_idx] - best_x[j_idx]
        dy = best_y[i_idx] - best_y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = best_r[i_idx] + best_r[j_idx]
        overlaps = np.maximum(0, sum_r - dist)
        
        bx = np.maximum(0, best_r - best_x) + np.maximum(0, best_x + best_r - 1)
        by = np.maximum(0, best_r - best_y) + np.maximum(0, best_y + best_r - 1)
        boundary_violation = np.sum(bx + by)
        
        if np.sum(overlaps) < 1e-8 and boundary_violation < 1e-8:
            break
        best_r *= 0.999
        
    result = np.column_stack((best_x, best_y, best_r))
    return result
# EVOLVE-BLOCK-END
