"""
Strategy: Direct Non-Linear Optimization with Multi-Start
Instead of fixing topologies and doing 1D aspect ratio search, we optimize 
all variables (positions, radii, W, H) simultaneously using L-BFGS-B.
We use a smooth penalty function for overlaps and walls to guide the gradient.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None

    def objective(params):
        """
        Objective function for minimization.
        We want to MAXIMIZE sum(r). 
        We MINIMIZE: -sum(r) + penalty(overlaps) + penalty(walls) + penalty(perimeter)
        
        params layout:
        [0]: W
        [1]: H
        [2:2+N]: x coordinates
        [2+N:2+2N]: y coordinates
        [2+2N:2+3N]: radii
        """
        W = params[0]
        H = params[1]
        x = params[2:2+N]
        y = params[2+N:2+2*N]
        r = params[2+2*N:2+3*N]
        
        # Enforce positive dimensions and radii (soft penalty if needed, but bounds handle this mostly)
        if W <= 0.1 or H <= 0.1:
            return 1e6
        
        # Perimeter constraint: 2(W+H) = 4 => W+H=2.
        # We optimize W, and set H = 2-W? 
        # No, the prompt says "variable rectangle of perimeter 4". 
        # It's better to optimize W and H with a penalty for deviation from W+H=2,
        # OR simply fix H = 2-W. 
        # Fixing H=2-W reduces dimensionality and ensures constraint satisfaction.
        # Let's stick to optimizing W only for dimensions, and derive H.
        # But wait, the current interface returns x, y, r. The evaluator checks perimeter <= 4.
        # If we fix H = 2-W, we are strictly on the boundary. This is optimal.
        # So params: [W, x1...xN, y1...yN, r1...rN]
        
        # Re-interpret params for this function (assuming caller passes correct structure)
        # Actually, let's define the variable vector clearly in the caller.
        # Let's re-define the objective to take a flat array:
        # [W, x1..xN, y1..yN, r1..rN]
        
        W_val = params[0]
        H_val = 2.0 - W_val
        
        if H_val <= 0.1:
            return 1e6

        x_val = params[1:1+N]
        y_val = params[1+N:1+2*N]
        r_val = params[1+2*N:1+3*N]
        
        # 1. Maximize sum of radii -> Minimize -sum(r)
        obj = -np.sum(r_val)
        
        # 2. Wall Penalties
        # Circles must be inside: x_i >= r_i, x_i <= W - r_i, etc.
        # Violation: max(0, r_i - x_i), max(0, r_i - (W - x_i)), etc.
        wall_viol_x_left = np.maximum(0, r_val - x_val)
        wall_viol_x_right = np.maximum(0, r_val - (W_val - x_val))
        wall_viol_y_bottom = np.maximum(0, r_val - y_val)
        wall_viol_y_top = np.maximum(0, r_val - (H_val - y_val))
        
        wall_penalty = np.sum(wall_viol_x_left**2) + np.sum(wall_viol_x_right**2) + \
                       np.sum(wall_viol_y_bottom**2) + np.sum(wall_viol_y_top**2)
        
        # 3. Overlap Penalties
        # dist(i,j) >= r_i + r_j
        # Violation: max(0, r_i + r_j - dist_ij)^2
        overlap_penalty = 0.0
        # Vectorized distance calculation
        # Create meshgrid of positions
        # To save memory/time for N=21, we can do it in chunks or just loop if N is small.
        # N=21 is small enough for a simple double loop in Python if not too frequent, 
        # but vectorization is better.
        # Let's use scipy.spatial.distance.pdist for efficiency if available, or manual.
        
        # Manual vectorized for N=21:
        # diff_x = x_val[:, None] - x_val[None, :]
        # diff_y = y_val[:, None] - y_val[None, :]
        # dist = np.sqrt(diff_x**2 + diff_y**2)
        # sum_r = r_val[:, None] + r_val[None, :]
        # violations = np.maximum(0, sum_r - dist)
        # overlap_penalty = np.sum(violations**2)
        
        # To avoid diagonal (0,0) which is 0 violation, we can just sum all, it's fine.
        diff_x = x_val[:, np.newaxis] - x_val[np.newaxis, :]
        diff_y = y_val[:, np.newaxis] - y_val[np.newaxis, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        
        sum_r = r_val[:, np.newaxis] + r_val[np.newaxis, :]
        
        # Mask diagonal to avoid numerical noise, though max(0, 0-0)=0
        np.fill_diagonal(dist, np.inf)
        
        violations = np.maximum(0, sum_r - dist)
        overlap_penalty = np.sum(violations**2)
        
        # Weighting: 
        # The penalties need to be strong enough to keep circles disjoint.
        # We want to maximize sum(r) subject to constraints.
        # Lagrangian approach: Min -sum(r) + lambda * (penalties)
        # Let's use a large lambda to enforce constraints strictly.
        # However, if lambda is too high, the optimizer might just shrink radii to 0 to satisfy constraints easily.
        # We want to push radii UP until they hit constraints.
        
        # A better objective: 
        # If constraints are satisfied, return -sum(r).
        # If not, return a large penalty + -sum(r).
        
        penalty_total = wall_penalty + overlap_penalty
        
        if penalty_total > 1e-6:
             # Heavily penalize violations to guide back to feasibility
             return 100.0 * penalty_total - np.sum(r_val)
        else:
             return -np.sum(r_val)

    def generate_initial_guess(W_init, H_init, mode="random"):
        """
        Generate initial positions and radii.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        if mode == "grid":
            # Rough grid
            rows = int(np.sqrt(N * H_init / W_init))
            cols = int(np.ceil(N / rows))
            # Adjust W, H to fit grid better? No, just place them.
            for i in range(N):
                row = i // cols
                col = i % cols
                x[i] = (col + 0.5) * W_init / cols
                y[i] = (row + 0.5) * H_init / rows
                r[i] = 0.4 * min(W_init/cols, H_init/rows)
        
        elif mode == "hex":
            # Hexagonal packing approximation
            # 5 rows: 4, 5, 4, 5, 3
            counts = [4, 5, 4, 5, 3]
            idx = 0
            for i, count in enumerate(counts):
                for j in range(count):
                    if idx >= N: break
                    x[idx] = (j + 0.5) * W_init / count
                    y[idx] = (i + 0.5) * H_init / len(counts)
                    if i % 2 == 1:
                        x[idx] += W_init / (2 * count)
                    r[idx] = 0.4 * min(W_init/count, H_init/len(counts))
                    idx += 1
        else: # random
            x = np.random.uniform(0, W_init, N)
            y = np.random.uniform(0, H_init, N)
            # Small initial radii to avoid massive overlap penalties initially?
            # Or large? If large, penalty pushes them apart.
            r = np.full(N, 0.05)
            
        return x, y, r

    def try_optimization(W_init, H_init, mode, seed):
        np.random.seed(seed)
        
        x0, y0, r0 = generate_initial_guess(W_init, H_init, mode)
        
        # Variable vector: [W, x1..xN, y1..yN, r1..rN]
        params0 = np.concatenate(([W_init], x0, y0, r0))
        
        # Bounds
        # W in [0.5, 1.5] (aspect ratio 1:3 to 3:1)
        # x in [0, W] -> but W changes. This is tricky for bounds.
        # L-BFGS-B requires fixed bounds. 
        # We can't easily bound x by W if W is a variable.
        # Solution: 
        # 1. Normalize x, y to [0,1] and scale by W, H in the objective?
        #    If we optimize normalized coords u, v in [0,1], then x = u*W, y = v*H.
        #    Bounds for u, v are [0,1]. Bounds for W are [0.5, 1.5].
        #    This is much cleaner!
        
        # Let's redefine the optimization variables:
        # [W, u1..uN, v1..vN, r1..rN]
        # Where x = u*W, y = v*H = v*(2-W).
        
        # We need to rewrite the objective to take this new structure.
        
        def objective_normalized(params):
            W_val = params[0]
            H_val = 2.0 - W_val
            
            if H_val <= 0.1 or W_val <= 0.1:
                return 1e6

            u = params[1:1+N]
            v = params[1+N:1+2*N]
            r_val = params[1+2*N:1+3*N]
            
            x_val = u * W_val
            y_val = v * H_val
            
            obj = -np.sum(r_val)
            
            # Wall Penalties
            # x_i >= r_i => u_i * W_val >= r_i => u_i >= r_i / W_val
            # x_i <= W - r_i => u_i * W_val <= W_val - r_i => u_i <= 1 - r_i / W_val
            # Same for y.
            
            # It is easier to compute violations in physical space (x,y,r)
            wall_viol_x_left = np.maximum(0, r_val - x_val)
            wall_viol_x_right = np.maximum(0, r_val - (W_val - x_val))
            wall_viol_y_bottom = np.maximum(0, r_val - y_val)
            wall_viol_y_top = np.maximum(0, r_val - (H_val - y_val))
            
            wall_penalty = np.sum(wall_viol_x_left**2) + np.sum(wall_viol_x_right**2) + \
                           np.sum(wall_viol_y_bottom**2) + np.sum(wall_viol_y_top**2)
            
            # Overlap Penalties
            diff_x = x_val[:, np.newaxis] - x_val[np.newaxis, :]
            diff_y = y_val[:, np.newaxis] - y_val[np.newaxis, :]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            
            sum_r = r_val[:, np.newaxis] + r_val[np.newaxis, :]
            np.fill_diagonal(dist, np.inf)
            
            violations = np.maximum(0, sum_r - dist)
            overlap_penalty = np.sum(violations**2)
            
            penalty_total = wall_penalty + overlap_penalty
            
            if penalty_total > 1e-8:
                 return 100.0 * penalty_total - np.sum(r_val)
            else:
                 return -np.sum(r_val)

        # Initial params in normalized space
        u0 = x0 / W_init
        v0 = y0 / H_init
        
        params0_norm = np.concatenate(([W_init], u0, v0, r0))
        
        # Bounds
        # W: [0.4, 1.6]
        # u, v: [0, 1]
        # r: [0, 0.5] (max possible radius in a 1x1 box is 0.5, here W,H vary but 0.5 is a safe upper bound for individual r if W,H >= 0.4? 
        # If W=0.4, H=1.6, max r could be 0.2. If W=1.6, H=0.4, max r 0.2. 
        # If W=1, H=1, max r 0.5.
        # Let's use [0, 0.5] for r.
        
        bounds = [(0.4, 1.6)] + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 0.5)] * N
        
        # Optimization
        try:
            res = minimize(
                objective_normalized, 
                params0_norm, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < 0: # If it improved or succeeded
                # Extract solution
                W_opt = res.x[0]
                H_opt = 2.0 - W_opt
                u_opt = res.x[1:1+N]
                v_opt = res.x[1+N:1+2*N]
                r_opt = res.x[1+2*N:1+3*N]
                
                x_opt = u_opt * W_opt
                y_opt = v_opt * H_opt
                
                # Verify validity and get true sum
                # Check constraints strictly
                valid = True
                # Walls
                if np.any(x_opt < r_opt - 1e-6) or np.any(x_opt > W_opt - r_opt + 1e-6):
                    valid = False
                if np.any(y_opt < r_opt - 1e-6) or np.any(y_opt > H_opt - r_opt + 1e-6):
                    valid = False
                
                # Overlaps
                diff_x = x_opt[:, np.newaxis] - x_opt[np.newaxis, :]
                diff_y = y_opt[:, np.newaxis] - y_opt[np.newaxis, :]
                dist = np.sqrt(diff_x**2 + diff_y**2)
                np.fill_diagonal(dist, np.inf)
                sum_r = r_opt[:, np.newaxis] + r_opt[np.newaxis, :]
                if np.any(sum_r > dist + 1e-6):
                    valid = False
                
                if valid:
                    sum_r_val = np.sum(r_opt)
                    if sum_r_val > best_sum_r:
                        return sum_r_val, x_opt, y_opt, r_opt, W_opt, H_opt
                else:
                    # If invalid, try to project back or just skip
                    # For now, skip invalid results from optimizer
                    pass
                    
        except Exception:
            pass
            
        return None

    # Multi-start loop
    # We will try several aspect ratios and modes.
    
    # Aspect ratios to try: 1:1, 2:1, 1:2, 1.5:1, 1:1.5
    aspect_ratios = [1.0, 1.5, 2.0, 0.5, 0.66]
    modes = ["grid", "hex", "random"]
    
    seed = 0
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        # W/H = ar => W = ar * H. W+H=2 => H(ar+1)=2 => H=2/(ar+1), W=2ar/(ar+1)
        W_init = 2.0 * ar / (ar + 1.0)
        H_init = 2.0 - W_init
        
        for mode in modes:
            if time.time() > deadline - 5:
                break
                
            result = try_optimization(W_init, H_init, mode, seed)
            seed += 1
            
            if result:
                sum_r, x, y, r, W, H = result
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_config = np.column_stack((x, y, r))
                    
                    # If we found a good solution, we can try to refine it further with a tighter tolerance
                    # Or just accept it.
                    
    # If no valid config found, fallback to simple grid
    if best_config is None:
        # Simple valid config: 5x4 grid in 1x1 square? 
        # W=1, H=1. 
        # 5 cols, 4 rows. 
        # r = 0.5/5 = 0.1? No, spacing is 0.2. r=0.1.
        # Sum r = 21 * 0.1 = 2.1.
        # Let's create a valid one.
        W = 1.0
        H = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        rows = 4
        cols = 5
        for i in range(N):
            row = i // cols
            col = i % cols
            x[i] = (col + 0.5) / cols
            y[i] = (row + 0.5) / rows
            r[i] = 0.5 / cols # 0.1
        best_config = np.column_stack((x, y, r))
        best_sum_r = np.sum(r)

    return best_config

# EVOLVE-BLOCK-END
