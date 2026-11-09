"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 144.0  # Safety margin for 150s limit
    
    def calc_radius(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        wall_dists = 0.5 - np.max(np.abs(p), axis=1)
        min_wall_dist = np.min(wall_dists)
        r = min(min_pair_dist / 2.0, min_wall_dist)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_wall_idx = np.argmin(wall_dists)
        return r, min_pair_idx, min_wall_idx

    def objective_and_grad(flat):
        p = flat.reshape(N, 2)
        # Soft-min surrogate for smooth gradients
        # f = -min(min_pair/2, min_wall)
        # We maximize r, so minimize -r.
        # r = min(min_pair_dist/2, min_wall_dist)
        # To make it differentiable, we use log-sum-exp (soft-min)
        # soft_min(a, b) = -logsumexp(-a, -b)
        # But we need gradient of the whole thing.
        # Let's just use the exact objective for L-BFGS-B if we can compute gradient.
        # Gradient of min is 0 if not at boundary? No, L-BFGS-B handles non-smooth poorly.
        # Let's use a soft-min approximation.
        beta = 50.0 # Temperature for soft-min
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = 0.5 - np.max(np.abs(p), axis=1)
        
        # Candidate radii
        r_pairs = dists / 2.0
        r_walls = wall_dists
        
        # Flatten all constraints into a vector of 'violations' or 'margins'
        # We want to maximize min(all margins).
        # Margins: dists/2 - r  >= 0  =>  r <= dists/2
        #          wall_dists - r >= 0 => r <= wall_dists
        
        # Let's define the objective as -soft_min(all margins)
        # Soft min of a set of values v_i:
        # s = -1/beta * log(sum(exp(-beta * v_i)))
        
        all_vals = np.concatenate([r_pairs.flatten(), r_walls])
        # Filter out inf? No, inf in exp(-beta*inf) -> 0.
        # But exp(-beta * 100) is tiny.
        # logsumexp is numerically stable.
        
        # We want to maximize the minimum.
        # Objective to minimize: -soft_min(all_vals)
        # soft_min(v) = -1/beta * log(sum(exp(-beta * v)))
        # -soft_min(v) = 1/beta * log(sum(exp(-beta * v)))
        
        # Gradient of -soft_min(v) w.r.t v_i:
        # d/dv_i (1/beta * log(sum(exp(-beta * v_j))))
        # = 1/beta * (1/sum) * sum_j (d/dv_i exp(-beta * v_j))
        # = 1/beta * (1/sum) * exp(-beta * v_i) * (-beta)
        # = -exp(-beta * v_i) / sum(exp(-beta * v_j))
        # = - p_i  where p_i = exp(-beta * v_i) / sum(...)
        
        # So grad_obj_wrt_v = -p
        
        # Now we need grad_v w.r.t p.
        
        # This is getting complicated to vectorize fully.
        # Let's stick to a simpler approach:
        # Use exact objective but with a finite difference gradient?
        # Or just use L-BFGS-B with exact objective and let it use finite differences?
        # L-BFGS-B can handle non-smooth to some extent, but might get stuck.
        
        # Alternative: Use the soft-min objective.
        # Let's implement the gradient for the soft-min objective.
        
        # Values:
        # 1. Pair distances: d_ij = ||p_i - p_j|| / 2
        # 2. Wall distances: w_i = 0.5 - max(|p_ix|, |p_iy|)
        
        # Let's compute weights w_k for each constraint k.
        # w_k = exp(-beta * v_k) / sum(exp(-beta * v_j))
        
        # For pairs:
        # v_ij = d_ij
        # grad d_ij w.r.t p_i = (p_i - p_j) / ||p_i - p_j||
        # grad d_ij w.r.t p_j = (p_j - p_i) / ||p_i - p_j||
        
        # For walls:
        # v_i = w_i
        # grad w_i w.r.t p_i = -sign(p_ik) for the k with max |p_ik|
        
        # Let's just use finite differences for the gradient in the minimize call.
        # It's slower per step but robust.
        # Or, let's try to compute the soft-min gradient properly.
        
        # Given time constraints, let's use a hybrid:
        # 1. SA to get a good start.
        # 2. L-BFGS-B with exact objective (finite diff grad) for polishing.
        
        # For now, in this function, we just return the objective value.
        # We will call minimize separately.
        return -min(np.min(r_pairs), np.min(r_walls))
    
    def sa_polish(pos, time_limit):
        """Run SA and return best position."""
        current_r, current_min_pair, current_min_wall = calc_radius(pos)
        best_r = current_r
        best_pos = pos.copy()
        
        T = 0.005
        T_end = 1e-5
        # Estimate steps based on time
        # Assume ~1000 steps/sec?
        steps = 50000
        
        alpha = (T_end / T) ** (1.0 / steps)
        current_pos = pos.copy()
        
        for step in range(steps):
            if time.time() > time_limit:
                break
                
            i = np.random.randint(N)
            
            move_size = 0.02 * (T / 0.005)
            move = np.random.uniform(-move_size, move_size, 2)
            
            new_pos = current_pos.copy()
            new_pos[i] += move
            
            if np.any(new_pos[i] > 0.5) or np.any(new_pos[i] < -0.5):
                continue
            
            # Fast check
            # If i is not in min pair/wall, we can be faster
            i_in_pair = (current_min_pair[0] == i or current_min_pair[1] == i)
            i_is_wall = (current_min_wall == i)
            
            if i_in_pair or i_is_wall:
                new_r, new_min_pair, new_min_wall = calc_radius(new_pos)
            else:
                dists_i = np.sqrt(np.sum((new_pos[i] - current_pos)**2, axis=1))
                dists_i[i] = np.inf
                min_pair_dist_new = np.min(dists_i)
                wall_dist_i = 0.5 - np.max(np.abs(new_pos[i]))
                
                new_r = min(current_r, min_pair_dist_new / 2.0, wall_dist_i)
                
                if new_r < current_r:
                    if min_pair_dist_new / 2.0 < current_r and min_pair_dist_new / 2.0 <= wall_dist_i:
                        j = np.argmin(dists_i)
                        new_min_pair = (i, j) if i < j else (j, i)
                        new_min_wall = current_min_wall
                    elif wall_dist_i < current_r:
                        new_min_wall = i
                        new_min_pair = current_min_pair
                    else:
                        new_min_pair = current_min_pair
                        new_min_wall = current_min_wall
                else:
                    new_min_pair = current_min_pair
                    new_min_wall = current_min_wall
            
            delta_r = new_r - current_r
            if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
                current_pos = new_pos
                current_r = new_r
                current_min_pair = new_min_pair
                current_min_wall = new_min_wall
                
                if current_r > best_r:
                    best_r = current_r
                    best_pos = current_pos.copy()
            
            T *= alpha
            
        return best_pos

    def lbfgs_polish(pos, time_limit):
        """Run L-BFGS-B to polish the position."""
        if time.time() > time_limit:
            return pos
            
        # Flatten
        x0 = pos.flatten()
        
        def obj(flat):
            p = flat.reshape(N, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            wall_dists = 0.5 - np.max(np.abs(p), axis=1)
            min_wall_dist = np.min(wall_dists)
            r = min(min_pair_dist / 2.0, min_wall_dist)
            return -r
        
        # Constraints: bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(
                obj, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return res.x.reshape(N, 2)
        except Exception:
            return pos

    # 1. Initialization
    np.random.seed(42)
    
    best_overall_r = -np.inf
    best_overall_pos = None
    
    # Generate several diverse initial configurations
    configs = []
    
    # Grid
    m = 8
    n = 7
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    pos_grid = grid[:N].copy()
    pos_grid += np.random.uniform(-0.01, 0.01, pos_grid.shape)
    pos_grid = np.clip(pos_grid, -0.5, 0.5)
    configs.append(pos_grid)
    
    # Random
    pos_rand = np.random.uniform(-0.4, 0.4, (N, 2))
    configs.append(pos_rand)
    
    # Honeycomb-like
    # Hexagonal packing
    h = np.sqrt(3)/2
    pos_hex = np.zeros((N, 2))
    idx = 0
    for row in range(10):
        for col in range(10):
            if idx >= N: break
            x = col * 0.11 - 0.5
            y = row * 0.095 - 0.5
            if row % 2 == 1:
                x += 0.055
            pos_hex[idx] = [x, y]
            idx += 1
    pos_hex += np.random.uniform(-0.005, 0.005, pos_hex.shape)
    pos_hex = np.clip(pos_hex, -0.5, 0.5)
    configs.append(pos_hex)
    
    # Run optimization on each config
    for i, config in enumerate(configs):
        if time.time() > deadline - 10:
            break
            
        # SA Phase
        # Allocate time based on remaining time
        sa_time_limit = time.time() + 30.0
        pos_sa = sa_polish(config, sa_time_limit)
        
        # L-BFGS Phase
        lbfgs_time_limit = time.time() + 10.0
        pos_final = lbfgs_polish(pos_sa, lbfgs_time_limit)
        
        r, _, _ = calc_radius(pos_final)
        
        if r > best_overall_r:
            best_overall_r = r
            best_overall_pos = pos_final.copy()
            
        # If we have time, do a second round of SA + LBFGS on the best result
        if time.time() < deadline - 20:
            # Refine the best result further
            sa_time_limit2 = time.time() + 20.0
            pos_sa2 = sa_polish(best_overall_pos, sa_time_limit2)
            
            lbfgs_time_limit2 = time.time() + 10.0
            pos_final2 = lbfgs_polish(pos_sa2, lbfgs_time_limit2)
            
            r2, _, _ = calc_radius(pos_final2)
            if r2 > best_overall_r:
                best_overall_r = r2
                best_overall_pos = pos_final2.copy()
                
    if best_overall_pos is None:
        # Fallback
        return configs[0]
        
    return best_overall_pos
# EVOLVE-BLOCK-END
