"""Circle packing in unit square n=32 with hexagonal init and smooth optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 32

# EVOLVE-BLOCK-START

def construct():
    """
    Pack 32 circles in the unit square, maximizing sum of radii.
    Strategy:
      1. Hexagonal lattice initialization (close to known optimal geometry for many N).
      2. Smooth penalty optimization using L-BFGS-B to grow radii and adjust positions.
      3. Time-guarded: returns best valid result found before deadline.
    """
    t0 = time.time()
    DEADLINE = 150.0  # wall-clock limit from evaluator; use 145s to be safe
    TIME_LIMIT = DEADLINE - 5.0  # 145s budget

    # --- Initialization: hexagonal grid ---
    def hex_init(n_circles=32):
        # Generate a hexagonal lattice that covers the unit square.
        # For N=32, a reasonable layout is roughly 6 columns x 6 rows with
        # some removed, or a 7x5 arrangement. We generate a large grid and
        # pick the best 32.
        
        # Hex grid parameters
        # Horizontal spacing: 2*r
        # Vertical spacing: sqrt(3)*r
        # Alternate rows offset by r
        
        # We want to fit ~32 circles. Let's try a few configurations.
        # A 6x6 grid has 36, remove 4 corners = 32.
        # A 7x5 grid has 35, remove 3 = 32.
        
        # Generate grid points for a 6x6 hex arrangement
        cols = 6
        rows = 6
        
        # Estimate radius
        # Width = (cols-1)*2*r + 2*r = cols*2*r
        # Height = (rows-1)*sqrt(3)*r + 2*r
        # We want both to fit in [0,1]
        # 2*cols*r <= 1 => r <= 1/(2*cols)
        # (rows-1)*sqrt(3)*r + 2*r <= 1 => r*(2 + (rows-1)*sqrt(3)) <= 1
        # => r <= 1/(2 + (rows-1)*sqrt(3))
        
        r_w = 1.0 / (2.0 * cols)
        r_h = 1.0 / (2.0 + (rows - 1) * np.sqrt(3))
        r_init = min(r_w, r_h) * 0.95  # slight margin
        
        # Generate all grid points
        points = []
        for j in range(rows):
            for i in range(cols):
                x = r_init + i * 2 * r_init + (j % 2) * r_init
                y = r_init + j * np.sqrt(3) * r_init
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append((x, y, r_init))
        
        # If we have more than n_circles, keep the most central ones
        if len(points) > n_circles:
            cx, cy = 0.5, 0.5
            points.sort(key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2)
            points = points[:n_circles]
        elif len(points) < n_circles:
            # Fallback: add more points or adjust
            # For safety, just use what we have and pad
            while len(points) < n_circles:
                points.append((0.5, 0.5, r_init * 0.1))
        
        return np.array(points[:n_circles])
    
    best_result = hex_init(N)
    best_score = np.sum(best_result[:, 2])
    
    # --- Smooth penalty objective ---
    def objective(params, softness=100.0):
        """
        Minimize: -sum(r_i) + penalty for constraint violations.
        params: array of shape (3*N,) = [x1, y1, r1, x2, y2, r2, ...]
        """
        n = N
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        
        # Ensure positive radii
        r_safe = np.maximum(r, 1e-6)
        
        # Boundary constraints: 0 <= x - r, x + r <= 1, 0 <= y - r, y + r <= 1
        boundary_viol = (
            np.maximum(0, r_safe - x)**2 +
            np.maximum(0, r_safe - (1 - x))**2 +
            np.maximum(0, r_safe - y)**2 +
            np.maximum(0, r_safe - (1 - y))**2
        )
        boundary_pen = np.sum(boundary_viol)
        
        # Pairwise non-overlap: dist(i,j) >= r_i + r_j
        # Use vectorized computation
        ix = np.repeat(x, n)
        iy = np.repeat(y, n)
        ir = np.repeat(r_safe, n)
        jx = np.tile(x, n)
        jy = np.tile(y, n)
        jr = np.tile(r_safe, n)
        
        dist = np.sqrt((ix - jx)**2 + (iy - jy)**2)
        min_dist = ir + jr
        
        # Only consider i < j to avoid double counting and self-pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_viol = np.maximum(0, min_dist[i_idx, j_idx] - dist[i_idx, j_idx])**2
        pair_pen = np.sum(pair_viol)
        
        # Soft penalty: use exp scaling for smoothness
        total_penalty = np.exp(softness * boundary_pen) - 1 + np.exp(softness * pair_pen) - 1
        
        # Negative sum of radii (we minimize)
        return -np.sum(r_safe) + total_penalty
    
    # --- Optimization loop with restarts ---
    max_time_per_opt = TIME_LIMIT / 6.0  # allow up to 6 optimization calls
    n_restarts = 4
    
    for restart in range(n_restarts):
        if time.time() - t0 > TIME_LIMIT:
            break
        
        # Initialize from previous best or hex init with perturbation
        if restart == 0:
            params_init = best_result.flatten().copy()
        else:
            # Perturb the best solution
            params_init = best_result.flatten().copy()
            rng = np.random.RandomState(restart * 42)
            perturbation = 0.01 * rng.randn(3 * N)
            # Scale perturbation per variable type
            perturbation[0::3] *= 1.0   # x
            perturbation[1::3] *= 1.0   # y
            perturbation[2::3] *= 0.5   # r
            params_init += perturbation
            # Ensure basic feasibility
            params_init[0::3] = np.clip(params_init[0::3], 0, 1)
            params_init[1::3] = np.clip(params_init[1::3], 0, 1)
            params_init[2::3] = np.maximum(params_init[2::3], 1e-6)
        
        # Time limit for this optimization call
        t_start = time.time()
        time_allowance = min(max_time_per_opt, TIME_LIMIT - (time.time() - t0))
        
        if time_allowance < 1.0:
            break
        
        try:
            # Use L-BFGS-B with time-based stopping
            result = minimize(
                objective,
                params_init,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'maxfun': 1000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'disp': False
                }
            )
            
            # Extract solution
            params_opt = result.x
            x_opt = params_opt[0:N]
            y_opt = params_opt[N:2*N]
            r_opt = np.maximum(params_opt[2*N:3*N], 1e-6)
            
            # Validate: check if all constraints are satisfied
            valid = True
            
            # Check boundaries
            if np.any(x_opt - r_opt < -1e-6) or np.any(x_opt + r_opt > 1 + 1e-6):
                valid = False
            if np.any(y_opt - r_opt < -1e-6) or np.any(y_opt + r_opt > 1 + 1e-6):
                valid = False
            
            # Check pairwise
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2)
                        if d < r_opt[i] + r_opt[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x_opt, y_opt, r_opt])
            
        except Exception:
            pass
        
        # Check time
        if time.time() - t0 > TIME_LIMIT:
            break
    
    # Final validation of best_result
    x_final = best_result[:, 0]
    y_final = best_result[:, 1]
    r_final = np.maximum(best_result[:, 2], 1e-6)
    
    # If the final result has constraint violations, try to fix them
    # by clipping and reducing radii
    r_adjusted = np.minimum(r_final, np.minimum(x_final, 1 - x_final))
    r_adjusted = np.minimum(r_adjusted, np.minimum(y_final, 1 - y_final))
    
    # Check pairwise and reduce radii if needed
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x_final[i] - x_final[j])**2 + (y_final[i] - y_final[j])**2)
            max_r = d / 2.0
            if r_adjusted[i] + r_adjusted[j] > max_r:
                # Scale down both
                excess = (r_adjusted[i] + r_adjusted[j]) - max_r
                r_adjusted[i] -= excess / 2.0
                r_adjusted[j] -= excess / 2.0
                r_adjusted[i] = max(r_adjusted[i], 1e-6)
                r_adjusted[j] = max(r_adjusted[j], 1e-6)
    
    best_result[:, 2] = r_adjusted
    
    return best_result

# EVOLVE-BLOCK-END
