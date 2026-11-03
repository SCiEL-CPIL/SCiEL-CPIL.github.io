import numpy as np
import time

N = 26

def construct():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Uses Simulated Annealing with a penalty method for overlaps.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_sum = 0.0
    best_x = None
    best_y = None
    best_r = None
    
    # Helper: Calculate penalty for overlaps and boundary violations
    # Returns (total_score, is_valid)
    def evaluate(x, y, r):
        # Boundary check
        # x, y must be in [r, 1-r]
        # If not, it's invalid. We handle this by projecting or penalizing.
        # For SA, we can just penalize hard.
        
        # Vectorized distance matrix for N circles
        # d[i, j] = distance between i and j
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        # Sum of radii matrix
        r_sum = r[:, None] + r[None, :]
        
        # Overlap penalty: max(0, r_i + r_j - d_ij)^2
        # We only need upper triangle to avoid double counting, but full matrix is fine for simplicity
        # Mask diagonal (self-distance is 0, r_i+r_i > 0, so self-overlap would be penalized if not masked)
        # Actually, d[i,i] = 0. r_i+r_i > 0. So we must mask i==j.
        mask = ~np.eye(N, dtype=bool)
        
        overlap = np.where(mask, np.maximum(0, r_sum - d), 0)
        penalty = np.sum(overlap**2)
        
        # Boundary penalty
        # x < r or x > 1-r
        bound_pen_x = np.sum(np.maximum(0, r - x)**2) + np.sum(np.maximum(0, x - (1 - r))**2)
        bound_pen_y = np.sum(np.maximum(0, r - y)**2) + np.sum(np.maximum(0, y - (1 - r))**2)
        
        total_penalty = penalty + bound_pen_x + bound_pen_y
        
        score = np.sum(r) - 10.0 * total_penalty # Penalty weight
        
        # Validity check for final result
        is_valid = (penalty == 0) and (bound_pen_x == 0) and (bound_pen_y == 0)
        
        return score, is_valid

    def is_valid_fast(x, y, r):
        # Fast check for a single state, assuming r > 0
        # Bounds
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return False
        # Overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        # Check if any d < r_sum (excluding diagonal)
        # To avoid floating point issues, use a small epsilon
        mask = ~np.eye(N, dtype=bool)
        overlaps = np.any((d < r_sum - 1e-9) & mask)
        return not overlaps

    # SA Parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.9995 # Cooling rate
    
    num_restarts = 100 # Max restarts, limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Strategy: Start with small random radii and random positions
        # Or start with a grid and perturb?
        # Random is safer to avoid symmetry traps.
        x = np.random.uniform(0, 1, N)
        y = np.random.uniform(0, 1, N)
        # Start with small radii to ensure validity initially
        r = np.random.uniform(0.01, 0.05, N)
        
        # Project to valid bounds initially
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        # If initial state is invalid due to overlaps (unlikely with small r), shrink radii
        # Simple fix: just start very small
        r = np.full(N, 0.01)
        x = np.random.uniform(0.05, 0.95, N)
        y = np.random.uniform(0.05, 0.95, N)
        
        current_score, _ = evaluate(x, y, r)
        T = T_start
        
        # Local best for this restart
        restart_best_sum = 0
        restart_best_x = x.copy()
        restart_best_y = y.copy()
        restart_best_r = r.copy()
        
        # SA Loop
        steps_per_restart = 20000
        # Adjust steps based on remaining time?
        # Just run a fixed amount, break if time runs out
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Generate Move
            i = np.random.randint(N)
            move_type = np.random.random()
            
            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()
            
            if move_type < 0.7:
                # Move position
                # Step size proportional to temperature?
                step_size = 0.05 * (T + 0.1)
                x_new[i] += np.random.normal(0, step_size)
                y_new[i] += np.random.normal(0, step_size)
                # Clip to [0,1] roughly, penalty will handle rest
            else:
                # Change radius
                # Try to increase radius mostly
                if np.random.random() < 0.6:
                    r_new[i] += np.random.uniform(0, 0.02)
                else:
                    r_new[i] -= np.random.uniform(0, 0.02)
                r_new[i] = max(0.001, r_new[i])
                # Ensure radius doesn't exceed 0.5 (max possible in unit square)
                r_new[i] = min(r_new[i], 0.5)
            
            new_score, _ = evaluate(x_new, y_new, r_new)
            delta = new_score - current_score
            
            # Metropolis Acceptance
            if delta > 0 or np.random.random() < np.exp(delta / T):
                x = x_new
                y = y_new
                r = r_new
                current_score = new_score
                
                # Check if this is a valid improvement for this restart
                # We only track valid states as "best"
                # But checking validity every step is expensive (O(N^2))
                # evaluate() already computed penalty. If penalty is 0, it's valid.
                # We can re-use the penalty calculation from evaluate?
                # evaluate returns score. score = sum(r) - 10*penalty.
                # If score == sum(r), then penalty is 0.
                if abs(current_score - np.sum(r)) < 1e-9:
                    if np.sum(r) > restart_best_sum:
                        restart_best_sum = np.sum(r)
                        restart_best_x = x.copy()
                        restart_best_y = y.copy()
                        restart_best_r = r.copy()
                        
            # Cool
            T *= alpha
            if T < T_end:
                T = T_end
                
        # End of restart
        if restart_best_sum > best_sum:
            best_sum = restart_best_sum
            best_x = restart_best_x
            best_y = restart_best_y
            best_r = restart_best_r
            
        # Optional: Use the best found so far to initialize the next restart (warm start)
        # This can help convergence but might get stuck.
        # Let's stick to independent restarts for diversity, but maybe perturb the best?
        # For now, independent.

    # Final Validation and Formatting
    if best_x is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Ensure final validity with a strict check
    if not is_valid_fast(best_x, best_y, best_r):
        # If invalid, try to fix by shrinking radii slightly?
        # Or just return seed.
        # Let's try to fix by reducing radii until valid
        scale = 1.0
        while not is_valid_fast(best_x, best_y, best_r * scale) and scale > 0.1:
            scale *= 0.95
        best_r *= scale
        
    result = np.column_stack((best_x, best_y, best_r))
    return result
