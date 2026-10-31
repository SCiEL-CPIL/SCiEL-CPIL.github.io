"""Global optimization circle packing using Differential Evolution."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Objective function for minimization: -sum(r) + penalty
    # We want to maximize sum(r) subject to constraints.
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    
    def objective(params):
        # Reshape params to (N, 3)
        # params: [x1, y1, r1, x2, y2, r2, ...]
        # It is more efficient to keep it flat or reshape carefully
        # Let's reshape to (N, 3)
        # Actually, standard DE works on 1D arrays.
        # Indexing: i*3, i*3+1, i*3+2
        penalty = 0.0
        total_r = 0.0
        
        # Check wall constraints and sum radii
        for i in range(N):
            x_i = params[i*3]
            y_i = params[i*3+1]
            r_i = params[i*3+2]
            
            # Radii must be positive
            if r_i < 0:
                penalty += 1e6 * (-r_i)
                r_i = 0 # For calculation stability, though penalty handles it
            
            total_r += r_i
            
            # Wall constraints: circle must be inside [0, 1]^2
            # x_i >= r_i, x_i <= 1 - r_i
            # y_i >= r_i, y_i <= 1 - r_i
            
            if x_i < r_i:
                penalty += 1e4 * (r_i - x_i)
            if x_i > 1.0 - r_i:
                penalty += 1e4 * (x_i - (1.0 - r_i))
            if y_i < r_i:
                penalty += 1e4 * (r_i - y_i)
            if y_i > 1.0 - r_i:
                penalty += 1e4 * (y_i - (1.0 - r_i))
        
        # Check pairwise overlap constraints
        # dist(i, j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # If violated, add penalty proportional to violation
        for i in range(N):
            xi = params[i*3]
            yi = params[i*3+1]
            ri = params[i*3+2]
            
            for j in range(i+1, N):
                xj = params[j*3]
                yj = params[j*3+1]
                rj = params[j*3+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                sum_r = ri + rj
                sum_r_sq = sum_r * sum_r
                
                if dist_sq < sum_r_sq:
                    # Overlap exists
                    # Penalty: (sum_r - dist)^2 or similar
                    # Using linear penalty on distance difference is often more stable for DE
                    dist = np.sqrt(dist_sq) if dist_sq > 0 else 0.0
                    overlap = sum_r - dist
                    if overlap > 0:
                        penalty += 1e4 * overlap * overlap
        
        return -total_r + penalty

    # Bounds for each variable
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius in unit square is 0.5, but realistically smaller for 26 circles)
    # Let's use [0, 0.5] for r to be safe, DE will find smaller values.
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Initial population strategy
    # DE can be slow if starting from scratch. 
    # We can provide a good initial population based on a grid or random valid packing.
    # However, `differential_evolution` in scipy accepts `init` as 'latinhypercube' or 'sobol'.
    # 'sobol' is generally better for high dimensions.
    
    # Strategy: Run DE with a limited number of generations and maxiter to stay within time.
    # N=26, 78 variables.
    
    # We will run a few restarts if time permits, but DE is stochastic.
    # A single run with high maxiter is usually better than many short runs for DE.
    
    # Set maxiter based on time.
    # Each objective call is O(N^2) ~ 676 operations.
    # Population size (popsize) default is 15 * 78 ~ 1170.
    # Generations: 1000?
    # 1170 * 1000 = 1.17M calls.
    # Each call ~ 10-20 microseconds? Maybe 100us in Python.
    # 1.17M * 100us = 117 seconds. This is too slow for Python loops inside objective.
    
    # Optimization: Vectorize the objective function using NumPy.
    
    def objective_vec(params):
        # params is shape (78,)
        pts = params.reshape((N, 3))
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        
        penalty = 0.0
        total_r = np.sum(r)
        
        # Wall constraints
        # x < r -> penalty
        # x > 1-r -> penalty
        wall_pen_x_left = np.maximum(0, r - x)
        wall_pen_x_right = np.maximum(0, x - (1.0 - r))
        wall_pen_y_left = np.maximum(0, r - y)
        wall_pen_y_right = np.maximum(0, y - (1.0 - r))
        
        wall_penalty = np.sum(wall_pen_x_left) + np.sum(wall_pen_x_right) + \
                       np.sum(wall_pen_y_left) + np.sum(wall_pen_y_right)
        
        # Pairwise constraints
        # Compute distance matrix
        # diff = x[:, None] - x[None, :]
        # This is 26x26 matrix.
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx*dx + dy*dy
        
        r_sum = r[:, None] + r[None, :]
        r_sum_sq = r_sum * r_sum
        
        # We only care about i < j
        # Create mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Overlap check
        # dist < r_sum  <=> dist_sq < r_sum_sq (since both positive)
        # We need dist for penalty calculation if we use linear penalty.
        # Let's use squared distance penalty to avoid sqrt if possible?
        # No, geometric constraint is on dist.
        # Let's calculate dist only where needed.
        
        # For efficiency, we can compute dist for all pairs, then apply mask.
        dist = np.sqrt(dist_sq)
        
        # Overlap amount: r_sum - dist
        # Only for i < j
        overlap = np.maximum(0, r_sum - dist)
        overlap = overlap[mask]
        
        pair_penalty = np.sum(overlap * overlap)
        
        # Total penalty
        # Weighting: Wall violations are "hard" boundaries. Pairwise overlaps are "hard" constraints.
        # If we use quadratic penalty, it's smooth.
        penalty = 100.0 * (wall_penalty * wall_penalty + pair_penalty)
        
        # Note: If we use squared penalty for walls too, it's consistent.
        # Let's re-evaluate wall penalty as squared.
        wall_pen_sq = np.sum(wall_pen_x_left**2) + np.sum(wall_pen_x_right**2) + \
                      np.sum(wall_pen_y_left**2) + np.sum(wall_pen_y_right**2)
        
        penalty = 100.0 * (wall_pen_sq + pair_penalty)
        
        return -total_r + penalty

    # Run Differential Evolution
    # popsize: 15 * dim is default. 15 * 78 = 1170.
    # maxiter: 100
    # tol: 1e-8
    # seed: 42
    
    # We need to be careful with time.
    # Let's estimate speed of objective_vec.
    # Vectorized operations on 26x26 arrays are fast.
    # 100 generations * 1170 pop = 117,000 calls.
    # Should be well under 1 second.
    # We can afford more iterations.
    
    # Let's run DE with a generous maxiter, but monitor time.
    # Since we can't easily interrupt DE inside, we'll set a conservative maxiter 
    # and then use remaining time for local refinement or more DE runs if needed.
    # Or just set maxiter high and hope it's fast.
    
    # Let's try maxiter=200, popsize=10.
    
    try:
        result = differential_evolution(
            objective_vec,
            bounds,
            maxiter=300,
            popsize=10,
            seed=42,
            tol=1e-10,
            polish=False, # Polish uses L-BFGS-B, which might be slow or get stuck.
            # Actually, polish=True might help refine the final result.
            # Let's try polish=True with a small maxiter for polish?
            # Default polish uses L-BFGS-B.
            # Let's keep polish=False for now to control time strictly.
        )
        
        best_params = result.x
        best_val = result.fun
        
        # Extract best solution
        pts = best_params.reshape((N, 3))
        x_best = pts[:, 0]
        y_best = pts[:, 1]
        r_best = pts[:, 2]
        
        # Validate and clean up
        # If penalty is not 0, the solution is invalid.
        # We should check if the penalty is small enough.
        # If invalid, we might need to project or return a valid fallback.
        
        # Check validity
        valid = True
        # Walls
        if np.any(x_best < r_best) or np.any(x_best > 1-r_best) or np.any(y_best < r_best) or np.any(y_best > 1-r_best):
            # Try to project? Or just mark invalid.
            # For now, if DE finds a solution with penalty > 1e-6, it's likely invalid.
            if objective_vec(best_params) > 1e-6: # Re-evaluate to be sure
                valid = False
            # Also check explicit constraints
            if np.any(x_best - r_best < -1e-9) or np.any(1-r_best - x_best < -1e-9) or np.any(y_best - r_best < -1e-9) or np.any(1-r_best - y_best < -1e-9):
                valid = False
        
        if valid:
            return np.column_stack((x_best, y_best, r_best))
        else:
            # Fallback to a known valid grid if DE fails to find a valid packing
            # This ensures we don't return an invalid result.
            # Grid of 5x5 = 25 circles. We need 26.
            # Use 6x5 grid? Or just scale down a 5x5 grid and add one?
            # Simplest valid fallback: 5x5 grid of 25 circles + 1 tiny circle in a gap?
            # Or just 26 circles in a 6x5 grid, scaled to fit.
            # 6x5 grid: width 6, height 5.
            # Max radius limited by smallest dimension / max(cells).
            # Let's use a 6x5 grid.
            nx, ny = 6, 5
            x_grid, y_grid = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
            # This puts centers at 0, 1/5, 1/2... 
            # Radius should be 0.5 / max(nx, ny) ?
            # Distance between centers is 1/5 = 0.2.
            # So radius 0.1 fits.
            # 26 circles.
            # Flatten grid
            x_flat = x_grid.flatten()
            y_flat = y_grid.flatten()
            r_flat = np.full(30, 0.1) # 6*5=30, take 26
            # Check boundaries:
            # x=0, r=0.1 -> x-r = -0.1 < 0. Invalid.
            # Shift grid inside.
            # x in [r, 1-r].
            # Let's use linspace(r, 1-r, n)
            
            # Re-calculate grid properly
            n_x, n_y = 6, 5
            # Total 30 spots, take 26.
            # To maximize radius, we want them as large as possible.
            # In a grid, r = 0.5 / max(n_x, n_y) is the max radius if we center them?
            # No, if we have n_x circles in a row of width 1, max r = 1/(2*n_x).
            # 1/(2*6) = 1/12 = 0.0833.
            # Let's use r = 0.08.
            r_val = 0.08
            x_coords = np.linspace(r_val, 1-r_val, n_x)
            y_coords = np.linspace(r_val, 1-r_val, n_y)
            
            x_f, y_f = np.meshgrid(x_coords, y_coords)
            x_f = x_f.flatten()
            y_f = y_f.flatten()
            r_f = np.full(n_x*n_y, r_val)
            
            # Take first 26
            x_ret = x_f[:N]
            y_ret = y_f[:N]
            r_ret = r_f[:N]
            
            return np.column_stack((x_ret, y_ret, r_ret))
            
    except Exception as e:
        # Fallback to grid on error
        n_x, n_y = 6, 5
        r_val = 0.08
        x_coords = np.linspace(r_val, 1-r_val, n_x)
        y_coords = np.linspace(r_val, 1-r_val, n_y)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        x_f = x_f.flatten()
        y_f = y_f.flatten()
        r_f = np.full(n_x*n_y, r_val)
        return np.column_stack((x_f[:N], y_f[:N], r_f[:N]))

# EVOLVE-BLOCK-END
