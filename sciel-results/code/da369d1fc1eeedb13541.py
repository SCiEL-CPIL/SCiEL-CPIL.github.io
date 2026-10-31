import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute pair indices for vectorized operations
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack([i_idx, j_idx])
    num_pairs = len(pairs)
    
    def penalty_objective(z):
        """
        z is a vector of length 3N: [x_1..x_N, y_1..y_N, r_1..r_N]
        Returns a scalar penalty value. We want to minimize this.
        The objective is -sum(r) + penalty_terms.
        """
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]
        
        # Ensure radii are positive for stability in gradients
        r_safe = np.maximum(r, 1e-6)
        
        # 1. Objective: Maximize sum of radii -> Minimize -sum(r)
        obj = -np.sum(r_safe)
        
        # 2. Boundary Constraints:
        # x_i - r_i >= 0  => r_i - x_i <= 0
        # 1 - x_i - r_i >= 0 => r_i + x_i - 1 <= 0
        # y_i - r_i >= 0  => r_i - y_i <= 0
        # 1 - y_i - r_i >= 0 => r_i + y_i - 1 <= 0
        
        # We use a smooth penalty: sum(max(0, violation)^2)
        # To make it differentiable, we can use softplus or just squared max.
        # L-BFGS-B handles bounds, but we are using a penalty method because 
        # the pairwise constraints are non-linear and hard to express as simple bounds.
        
        # Boundary violations
        v_left = r_safe - x
        v_right = r_safe + x - 1.0
        v_bottom = r_safe - y
        v_top = r_safe + y - 1.0
        
        # Only penalize if violation > 0
        # Using np.maximum(v, 0)**2 is not smooth at 0, but L-BFGS-B often copes.
        # Better: use softplus approximation or just rely on the fact that 
        # we initialize validly and stay close. 
        # Let's use a high weight on squared violations.
        
        pen_boundary = np.sum(np.maximum(v_left, 0)**2) + \
                       np.sum(np.maximum(v_right, 0)**2) + \
                       np.sum(np.maximum(v_bottom, 0)**2) + \
                       np.sum(np.maximum(v_top, 0)**2)
        
        # 3. Pairwise Constraints:
        # dist(i,j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        # violation = (r_i + r_j)^2 - dist^2
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist_sq = dx*dx + dy*dy
        
        r_sum = r_safe[i_idx] + r_safe[j_idx]
        r_sum_sq = r_sum * r_sum
        
        violation_pairs = r_sum_sq - dist_sq
        
        # Penalize overlaps
        # If dist_sq < r_sum_sq, violation is positive.
        pen_pairs = np.sum(np.maximum(violation_pairs, 0)**2)
        
        # Total penalty
        # Weighting: The pairwise constraint is the most critical for validity.
        # We want the optimizer to prioritize fixing overlaps over growing radii.
        # If we just do obj + lambda * pen, the optimizer might keep radii small to avoid penalty.
        # We want to maximize radii SUBJECT TO validity.
        # A common trick: minimize (1 - sum(r)) + M * penalty, where M is large.
        # Or, minimize -sum(r) + M * penalty.
        
        M = 1000.0
        total = obj + M * (pen_boundary + pen_pairs)
        
        return total

    def local_optimize_scipy(z0, time_limit):
        """
        Runs L-BFGS-B from initial state z0.
        Returns the best valid state and score.
        """
        # Initial projection to ensure we start reasonably valid
        x0 = z0[:N]
        y0 = z0[N:2*N]
        r0 = z0[2*N:]
        
        # Basic boundary clipping
        r0 = np.minimum(r0, np.minimum(np.minimum(x0, 1-x0), np.minimum(y0, 1-y0)))
        r0 = np.maximum(r0, 1e-4)
        
        z0 = np.concatenate([x0, y0, r0])
        
        # Bounds for L-BFGS-B:
        # x, y in [0, 1]
        # r in [0, 0.5] (max possible radius in unit square)
        bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
        
        # We will run multiple short minimizations or one long one with time check.
        # SciPy's minimize doesn't have a built-in time limit, so we rely on 
        # maxiter and maxfun. 
        # To respect the time limit strictly, we might need to interrupt, 
        # but for 120s limit and ~5s per start, we can set generous maxiter 
        # and check time after each start.
        
        best_z = z0.copy()
        best_score = 0.0
        current_penalty = 0.0
        
        # We'll do a few steps of minimization
        try:
            res = minimize(
                penalty_objective, 
                z0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'maxfun': 2000,
                    'ftol': 1e-9,
                    'gtol': 1e-6
                }
            )
            best_z = res.x
        except Exception:
            pass
        
        # Evaluate the result
        x = best_z[:N]
        y = best_z[N:2*N]
        r = best_z[2*N:]
        
        # Calculate validity and score
        # If there are still significant overlaps, we must shrink to make it valid.
        # The penalty method might leave small overlaps if M isn't infinite.
        
        # Project to validity:
        # 1. Boundary
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, max_r_bound)
        
        # 2. Pairwise
        # Iterative shrink if overlaps exist
        for _ in range(20):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r[i_idx] + r[j_idx]
            violations = sum_r - dist
            if np.max(violations) < 1e-9:
                break
            # Shrink all radii slightly
            r *= 0.95
            r = np.minimum(r, max_r_bound)
        
        score = np.sum(r)
        
        # Compare with initial score just in case optimization diverged
        x0 = z0[:N]
        y0 = z0[N:2*N]
        r0 = z0[2*N:]
        max_r_bound0 = np.minimum(np.minimum(x0, 1-x0), np.minimum(y0, 1-y0))
        r0 = np.minimum(r0, max_r_bound0)
        # Check validity of initial
        valid0 = True
        for _ in range(10):
            dx = x0[i_idx] - x0[j_idx]
            dy = y0[i_idx] - y0[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r0[i_idx] + r0[j_idx]
            if np.max(sum_r - dist) < 1e-9:
                break
            r0 *= 0.95
            r0 = np.minimum(r0, max_r_bound0)
        
        if np.sum(r0) > score:
            score = np.sum(r0)
            best_z = np.concatenate([x0, y0, r0])
        
        return best_z, score

    best_global_state = None
    best_global_score = 0.0
    
    # Multi-Start Configurations
    # We generate diverse initial states.
    # 1. Grid
    # 2. Random
    # 3. Ring (Center + 2 rings)
    # 4. Hexagonal-like
    
    num_starts = 15
    time_per_start = (deadline - start_time) / num_starts
    time_per_start = max(time_per_start, 5.0) # Minimum 5s per start
    
    for k in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        start_k = time.time()
        
        # Generate Initial State z0 (shape 3N)
        if k % 4 == 0:
            # Grid
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < N:
                        x[count] = (i + 0.5) / side
                        y[count] = (j + 0.5) / side
                        r[count] = 0.08
                        count += 1
        elif k % 4 == 1:
            # Random
            x = np.random.rand(N) * 0.8 + 0.1
            y = np.random.rand(N) * 0.8 + 0.1
            r = np.full(N, 0.05)
        elif k % 4 == 2:
            # Ring: 1 center, 8 inner, 17 outer
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.full(N, 0.05)
            
            # Center
            x[0] = 0.5
            y[0] = 0.5
            
            # Inner ring (8)
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            rad1 = 0.15
            x[1:9] = 0.5 + rad1 * np.cos(angles1)
            y[1:9] = 0.5 + rad1 * np.sin(angles1)
            
            # Outer ring (17)
            angles2 = np.linspace(0, 2*np.pi, 17, endpoint=False)
            rad2 = 0.35
            x[9:] = 0.5 + rad2 * np.cos(angles2)
            y[9:] = 0.5 + rad2 * np.sin(angles2)
            
        else:
            # Hexagonal packing approximation
            # 5x5 grid offset
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.full(N, 0.06)
            count = 0
            rows = 5
            cols = 5
            spacing = 0.18
            offset = 0.09
            for i in range(rows):
                for j in range(cols):
                    if count < N:
                        x[count] = offset + j * spacing + (i % 2) * (spacing / 2)
                        y[count] = offset + i * spacing
                        count += 1
            
        # Add noise to break symmetry
        x += np.random.normal(0, 0.005, N)
        y += np.random.normal(0, 0.005, N)
        
        # Clip to [0.01, 0.99] to stay away from boundaries initially
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        z0 = np.concatenate([x, y, r])
        
        # Run optimization
        candidate_z, candidate_score = local_optimize_scipy(z0, start_k + time_per_start)
        
        if candidate_score > best_global_score:
            best_global_score = candidate_score
            best_global_state = candidate_z.copy()
            
    # Fallback if no valid state found
    if best_global_state is None:
        x = np.full(N, 0.5)
        y = np.full(N, 0.5)
        r = np.full(N, 0.01)
        best_global_state = np.concatenate([x, y, r])
        best_global_score = np.sum(r)

    # Final safety projection
    x = best_global_state[:N]
    y = best_global_state[N:2*N]
    r = best_global_state[2*N:]
    
    # Boundary
    max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    r = np.minimum(r, max_r_bound)
    
    # Pairwise
    for _ in range(50):
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        violations = sum_r - dist
        if np.max(violations) < 1e-9:
            break
        # Shrink radii to resolve overlap
        # Simple global shrink
        r *= 0.98
        r = np.minimum(r, max_r_bound)
    
    # Construct output array
    final_state = np.column_stack([x, y, r])
    
    return final_state

# EVOLVE-BLOCK-END
