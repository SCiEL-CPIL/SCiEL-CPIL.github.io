import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute pair indices for vectorized operations
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack([i_idx, j_idx])
    num_pairs = len(pairs)
    
    def get_forces_and_energy(x, y, r):
        """
        Compute repulsive forces and potential energy for overlapping circles.
        Uses a simple spring-like model for overlaps: F = k * (penetration).
        Boundary forces push circles back into [0,1]^2.
        """
        # Boundary violations
        # We want x >= r, x <= 1-r, y >= r, y <= 1-r
        # Violation if x < r or x > 1-r
        # Let's define violation as distance outside the valid box [r, 1-r]
        
        # Instead of forces, let's just compute a penalty energy for optimization
        # Energy = sum of overlaps^2 + sum of boundary violations^2 - alpha * sum(r)
        # We want to MINIMIZE energy.
        
        # Overlaps
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Required distance: r_i + r_j
        req_dist = r[i_idx] + r[j_idx]
        
        # Overlap amount (positive if overlapping)
        overlap = req_dist - dist
        overlap = np.maximum(overlap, 0)
        
        # Boundary violations
        # x < r  => r - x > 0
        # x > 1-r => x - (1-r) > 0
        v_x_left = np.maximum(r - x, 0)
        v_x_right = np.maximum(x - (1 - r), 0)
        v_y_bot = np.maximum(r - y, 0)
        v_y_top = np.maximum(y - (1 - r), 0)
        
        # Total boundary violation
        v_bound = v_x_left + v_x_right + v_y_bot + v_y_top
        v_bound = np.maximum(v_bound, 0) # Should be non-negative already
        
        # Energy terms
        E_overlap = np.sum(overlap**2)
        E_bound = np.sum(v_bound**2)
        
        # Objective: Minimize penalty - reward
        # We want to maximize sum(r), so minimize -sum(r)
        # But we must ensure validity.
        # A common trick: if any violation > 0, return a huge penalty.
        # Otherwise, return -sum(r).
        # However, for gradient-based methods, we need smoothness.
        # Let's use a large weight for violations.
        
        W = 1000.0
        energy = W * (E_overlap + E_bound) - np.sum(r)
        
        return energy
    
    def project_valid(x, y, r):
        """
        Project the configuration to the nearest valid configuration.
        This is a heuristic projection.
        1. Clip positions to [r, 1-r].
        2. If overlaps remain, shrink radii.
        """
        # Clip to boundaries
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        
        # Check for overlaps and shrink if necessary
        for _ in range(20):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            dist = np.maximum(dist, 1e-9)
            
            overlap = r[i_idx] + r[j_idx] - dist
            if np.max(overlap) < 1e-8:
                break
            
            # Shrink radii proportionally to the overlap
            # Simple approach: shrink all radii by a factor
            # Better: shrink the involved circles
            # For now, global shrink is safer for validity
            factor = 1.0 - np.max(overlap) * 0.5
            r *= factor
            r = np.maximum(r, 1e-6)
            
            # Re-clip
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
        
        return x, y, r
    
    def local_refine(x, y, r):
        """
        Use L-BFGS-B to refine the positions and radii.
        Variables: x, y, r (flattened)
        """
        def objective(v):
            x_ = v[:N]
            y_ = v[N:2*N]
            r_ = v[2*N:]
            
            # Check validity
            # If invalid, return high penalty
            # We rely on the bounds in L-BFGS-B to keep r > 0
            # But we need to handle overlaps.
            
            # Simple check: if energy is high, return it
            return get_forces_and_energy(x_, y_, r_)
        
        v0 = np.concatenate([x, y, r])
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5] (max radius in unit square)
        # Actually, r is limited by position. 
        # L-BFGS-B bounds are box constraints. 
        # We can set r bounds to [1e-6, 0.5] and x,y to [0, 1].
        # The objective function will penalize invalid configs.
        
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(1e-6, 0.5)] * N
        
        try:
            res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                return x_opt, y_opt, r_opt
        except Exception:
            pass
            
        return x, y, r

    best_score = 0.0
    best_state = None
    
    # 1. Differential Evolution for Global Search
    # We need to define the parameter space for DE.
    # DE works on flat vectors.
    # We can use x, y, r directly, but the search space is huge.
    # Alternative: Use a structured initialization for DE?
    # DE is good for global, but 78 dimensions is a lot.
    # Let's try DE on a reduced space? 
    # Or just use DE on x, y, r with a good initial population?
    
    # Since N=26, 78 variables is manageable for DE if we limit iterations.
    # Bounds for DE:
    # x, y in [0.05, 0.95] (keep away from edges initially)
    # r in [0.01, 0.2] (reasonable range)
    
    bounds_de = [(0.05, 0.95)] * N + [(0.05, 0.95)] * N + [(0.01, 0.2)] * N
    
    # Objective for DE
    def de_objective(v):
        x_ = v[:N]
        y_ = v[N:2*N]
        r_ = v[2*N:]
        return get_forces_and_energy(x_, y_, r_)
        
    # Run DE
    # popsize=10, maxiter=50, tol=1e-4
    # This might be slow. 26*3 = 78 vars.
    # Let's try a smaller population and fewer iterations.
    
    try:
        de_res = differential_evolution(
            de_objective, 
            bounds_de, 
            seed=42, 
            maxiter=30, 
            popsize=8, 
            tol=1e-4,
            update_interval=1,
            polish=False # We will do our own polish
        )
        
        if de_res.success or de_res.fun < 1e6:
            x_de = de_res.x[:N]
            y_de = de_res.x[N:2*N]
            r_de = de_res.x[2*N:]
            
            # Project to valid
            x_de, y_de, r_de = project_valid(x_de, y_de, r_de)
            
            # Local Refine
            x_opt, y_opt, r_opt = local_refine(x_de, y_de, r_de)
            x_opt, y_opt, r_opt = project_valid(x_opt, y_opt, r_opt)
            
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_state = (x_opt, y_opt, r_opt)
                
    except Exception as e:
        print(f"DE failed: {e}")
        
    # 2. Fallback: Random Restarts + Local Refine
    # If DE didn't find anything good, or as a complement, do some random restarts.
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Random initialization
        x_rand = np.random.uniform(0.1, 0.9, N)
        y_rand = np.random.uniform(0.1, 0.9, N)
        r_rand = np.random.uniform(0.05, 0.15, N)
        
        # Project
        x_r, y_r, r_r = project_valid(x_rand, y_rand, r_rand)
        
        # Local Refine
        x_opt, y_opt, r_opt = local_refine(x_r, y_r, r_r)
        x_opt, y_opt, r_opt = project_valid(x_opt, y_opt, r_opt)
        
        score = np.sum(r_opt)
        if score > best_score:
            best_score = score
            best_state = (x_opt, y_opt, r_opt)
            
    # 3. If still no valid state, use a simple grid
    if best_state is None:
        # Simple grid fallback
        side = 5
        x_f = np.zeros(N)
        y_f = np.zeros(N)
        r_f = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx < N:
                    x_f[idx] = (i + 0.5) / side
                    y_f[idx] = (j + 0.5) / side
                    r_f[idx] = 0.5 / side * 0.9
                    idx += 1
        best_state = (x_f, y_f, r_f)
        best_score = np.sum(r_f)
        
    bx, by, br = best_state
    
    # Final validity check and projection
    bx, by, br = project_valid(bx, by, br)
    
    # Ensure strict validity for the final output
    # 1. Boundary
    br = np.minimum(br, np.minimum(bx, 1-bx))
    br = np.minimum(br, np.minimum(by, 1-by))
    
    # 2. Pairwise
    # If any overlap, shrink radii iteratively
    for _ in range(100):
        dx = bx[i_idx] - bx[j_idx]
        dy = by[i_idx] - by[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        violations = br[i_idx] + br[j_idx] - dist
        if np.max(violations) < 1e-9:
            break
        # Shrink all radii slightly
        br *= 0.98
        br = np.minimum(br, np.minimum(bx, 1-bx))
        br = np.minimum(br, np.minimum(by, 1-by))
        
    final_state = np.column_stack([bx, by, br])
    return final_state

# EVOLVE-BLOCK-END
