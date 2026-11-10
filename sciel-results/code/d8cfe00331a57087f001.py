import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def construct():
    N = 24
    start_time = time.time()
    max_time = 115.0
    
    # Precompute high-resolution grid for initial peak finding
    # We need sufficient resolution to not miss peaks.
    # For N=24, the polynomial can oscillate.
    # 2048 points might be enough for initial detection, but let's use 4096 for safety.
    M_grid = 4096
    theta_grid = np.linspace(0, 2 * np.pi, M_grid, endpoint=False)
    z_grid = np.exp(1j * theta_grid)
    V_grid = np.vander(z_grid, N=N, increasing=True)
    
    def exact_sup_norm(c):
        """
        Compute the maximum of |P(z)| for |z|=1 with high accuracy.
        """
        # 1. Evaluate on grid
        p_vals = V_grid @ c
        mags = np.abs(p_vals)
        idx_max = np.argmax(mags)
        max_val = mags[idx_max]
        
        # 2. Refine the peak
        # The peak is at theta_grid[idx_max].
        # We refine in a small neighborhood.
        # h is the grid spacing.
        h = 2 * np.pi / M_grid
        
        # Define objective function to maximize |P(z)|
        # We use minimize_scalar on -|P(z)|
        
        def obj(t):
            # t is offset from theta_grid[idx_max]
            z_loc = np.exp(1j * (theta_grid[idx_max] + t))
            # Compute P(z_loc) efficiently
            # Since N is small, direct dot product is fine
            # But we can precompute powers? No, z_loc changes.
            # Horner's method or direct sum.
            # Direct sum: c[0] + c[1]z + ...
            # We can use np.polyval? No, coeffs are c_k z^k, not c_k z^{N-1-k}.
            # Let's just use dot product with powers.
            # To speed up, we can reuse the grid evaluation logic if we were on grid, 
            # but we are off-grid.
            
            # Optimization: Precompute z_loc powers once per call?
            # Inside obj, z_loc is scalar.
            z_powers = np.ones(N, dtype=complex)
            for k in range(1, N):
                z_powers[k] = z_powers[k-1] * z_loc
            val = np.dot(c, z_powers)
            return -np.abs(val)
        
        # Bounds for refinement: [-h, h]
        # Handle wrap-around?
        # theta_grid[idx_max] + t might exceed 2pi or go below 0.
        # exp(1j * x) is periodic, so it's fine.
        
        res = minimize_scalar(obj, bounds=(-h, h), method='bounded', options={'xatol': 1e-10})
        
        if res.success:
            refined_val = -res.fun
            # It's possible the refined value is slightly less than grid max due to numerical noise,
            # but usually it's higher. We take the max just in case.
            return max(max_val, refined_val)
        else:
            return max_val

    def greedy_local_search(c):
        """
        Perform greedy local search.
        Repeatedly flip the coefficient that yields the largest improvement in sup_norm.
        Stop when no single flip improves the score.
        """
        current_c = c.copy()
        current_score = exact_sup_norm(current_c)
        
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_idx = -1
            best_new_score = current_score
            
            for k in range(N):
                # Flip k
                old_val = current_c[k]
                current_c[k] = -old_val
                
                new_score = exact_sup_norm(current_c)
                delta = current_score - new_score
                
                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_idx = k
                    best_new_score = new_score
                
                # Revert immediately to test next k
                current_c[k] = old_val
            
            if best_idx != -1:
                # Apply the best flip
                current_c[best_idx] = -current_c[best_idx]
                current_score = best_new_score
                improved = True
        
        return current_c, current_score

    best_c = None
    best_score = float('inf')
    
    # Multi-start strategy
    # Run greedy local search from multiple random starts
    # Estimate time for one run:
    # exact_sup_norm is ~1ms.
    # Greedy loop: ~24 evals/pass. ~10-20 passes. ~240-480 evals. ~0.5s per run.
    # We can run ~200 starts in 100s.
    
    num_starts = 300
    
    for i in range(num_starts):
        if time.time() - start_time > max_time:
            break
            
        # Random initial configuration
        c = np.random.choice([-1, 1], N).astype(float)
        
        # Run greedy search
        c_opt, score = greedy_local_search(c)
        
        if score < best_score:
            best_score = score
            best_c = c_opt.copy()
            
        # Optional: If we find the known optimum, we can stop early?
        # The task says "exhaustive optimum is 1.2583758".
        # If we get close to this, we might stop to save time for more diverse starts?
        # Or just keep going to ensure we found it.
        # Let's not stop early to be safe, but if we hit 1.2583758, we are likely done.
        if best_score < 1.25838:
            # We found a very good solution. 
            # We could break, but let's allow a few more to be sure it's the global min.
            # Actually, if we find the global min, further searches won't improve it.
            # But we don't know for sure if it's the global min without proof.
            # However, 1.2583758 is the known optimum.
            # Let's break if we are extremely close.
            if best_score < 1.258376:
                break

    # If best_c is still None (shouldn't happen), return random
    if best_c is None:
        return [int(x) for x in np.random.choice([-1, 1], N)]
        
    return [int(x) for x in best_c]
