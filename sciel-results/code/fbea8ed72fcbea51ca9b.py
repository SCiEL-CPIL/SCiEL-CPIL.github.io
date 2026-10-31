import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin for 150s limit
    
    # Precompute pair indices for vectorized distance checks
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    def penalty_objective(z):
        """
        Objective function for optimization.
        z is a vector of length 3*N: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        We maximize sum(r) subject to:
          1. 0 <= x_i <= 1, 0 <= y_i <= 1
          2. r_i >= 0
          3. r_i <= min(x_i, 1-x_i, y_i, 1-y_i)  (boundary)
          4. dist(i,j) >= r_i + r_j for all i != j (non-overlap)
        
        We use a penalty formulation:
        f(z) = -sum(r) + lambda * (penalty_terms)
        
        Penalty terms:
        - boundary_violation = sum(max(0, r_i - min(x_i,1-x_i,y_i,1-y_i))^2)
        - overlap_violation = sum(max(0, r_i + r_j - dist(i,j))^2)
        - negative_r = sum(max(0, -r_i)^2)
        """
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Boundary violation
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        boundary_viol = np.maximum(0, r - max_r_bound)
        boundary_penalty = np.sum(boundary_viol ** 2)
        
        # Negative radius violation
        neg_r = np.maximum(0, -r)
        neg_penalty = np.sum(neg_r ** 2)
        
        # Overlap violation
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        overlap_viol = np.maximum(0, sum_r - dist)
        overlap_penalty = np.sum(overlap_viol ** 2)
        
        total_penalty = boundary_penalty + neg_penalty + overlap_penalty
        
        # Objective: minimize -sum(r) + lambda * total_penalty
        # Use adaptive lambda
        lambda_val = 50.0
        return -np.sum(r) + lambda_val * total_penalty
    
    def is_valid(z, tol=1e-6):
        """Check if a solution is valid."""
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Boundary
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        if np.any(r > max_r_bound + tol):
            return False
        
        # Non-negative
        if np.any(r < -tol):
            return False
        
        # Overlap
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        if np.any(sum_r > dist + tol):
            return False
        
        return True
    
    def project_to_valid(z):
        """Project an invalid state to a valid one by scaling radii down."""
        x = z[0::3].copy()
        y = z[1::3].copy()
        r = z[2::3].copy()
        
        # Clip positions to [0,1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Clip radii to non-negative
        r = np.maximum(r, 0)
        
        # Iteratively scale down to resolve violations
        for _ in range(20):
            # Boundary check
            max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
            if np.any(r > max_r_bound):
                ratios = max_r_bound / np.maximum(r, 1e-12)
                scale = np.min(ratios) * 0.999
                r *= scale
            
            # Overlap check
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r[i_idx] + r[j_idx]
            violations = sum_r - dist
            if np.max(violations) < 1e-9:
                break
            # Scale down
            r *= 0.95
            
            # Re-check boundary after scale
            max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
            if np.any(r > max_r_bound):
                ratios = max_r_bound / np.maximum(r, 1e-12)
                scale = np.min(ratios) * 0.999
                r *= scale
        
        return np.concatenate([x, y, r])
    
    def refine_with_l_bfgs(z0, time_limit):
        """Refine a candidate using L-BFGS-B on the penalty objective."""
        start = time.time()
        
        # Bounds for L-BFGS-B
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(N):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.0, 0.5))   # r
        
        try:
            res = minimize(
                penalty_objective,
                z0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > time_limit:
                return None
            
            z_refined = res.x
            # Project to valid
            z_valid = project_to_valid(z_refined)
            return z_valid
            
        except Exception:
            return None
    
    # Multi-start strategy:
    # 1. Use differential evolution for global search (a few runs)
    # 2. Use L-BFGS-B for local refinement from diverse starts
    
    best_z = None
    best_score = 0.0
    
    # --- Phase 1: Differential Evolution (global search) ---
    # DE bounds: x,y in [0.05, 0.95], r in [0.01, 0.4]
    de_bounds = [(0.05, 0.95), (0.05, 0.95), (0.01, 0.4)] * N
    
    # Run DE with limited iterations to find good starting points
    de_time_budget = min(40.0, (deadline - start_time) * 0.3)
    
    try:
        de_result = differential_evolution(
            penalty_objective,
            bounds=de_bounds,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False,
            update_interval=1,
            callback=lambda xk, convergence: (time.time() > start_time + de_time_budget)
        )
        
        if de_result.x is not None:
            z_candidate = project_to_valid(de_result.x)
            score = np.sum(z_candidate[2::3])
            if score > best_score:
                best_score = score
                best_z = z_candidate.copy()
                
    except Exception as e:
        pass
    
    # --- Phase 2: L-BFGS-B refinement from multiple diverse starts ---
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        if best_z is not None:
            return np.column_stack([best_z[0::3], best_z[1::3], best_z[2::3]])
        # Fallback
        z_fallback = np.zeros(3*N)
        z_fallback[0::3] = 0.5
        z_fallback[1::3] = 0.5
        z_fallback[2::3] = 0.05
        return np.column_stack([z_fallback[0::3], z_fallback[1::3], z_fallback[2::3]])
    
    # Generate diverse starting points
    starts = []
    
    # 1. If DE found a good point, use it
    if best_z is not None:
        starts.append(best_z.copy())
    
    # 2. Grid-based init
    side = int(np.ceil(np.sqrt(N)))
    grid_state = np.zeros(3*N)
    for i in range(N):
        row = i // side
        col = i % side
        grid_state[3*i] = (col + 0.5) / side
        grid_state[3*i+1] = (row + 0.5) / side
        grid_state[3*i+2] = 0.08
    starts.append(grid_state)
    
    # 3. Ring-based init (1 center + 2 rings)
    ring_state = np.zeros(3*N)
    ring_state[0:2] = [0.5, 0.5]
    ring_state[2] = 0.15
    
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for k in range(8):
        idx = 1 + k
        ring_state[3*idx] = 0.5 + 0.25 * np.cos(angles1[k])
        ring_state[3*idx+1] = 0.5 + 0.25 * np.sin(angles1[k])
        ring_state[3*idx+2] = 0.1
    
    angles2 = np.linspace(0, 2*np.pi, 17, endpoint=False)
    for k in range(17):
        idx = 9 + k
        ring_state[3*idx] = 0.5 + 0.42 * np.cos(angles2[k])
        ring_state[3*idx+1] = 0.5 + 0.42 * np.sin(angles2[k])
        ring_state[3*idx+2] = 0.08
    starts.append(ring_state)
    
    # 4. Random init
    random_state = np.zeros(3*N)
    random_state[0::3] = np.random.uniform(0.1, 0.9, N)
    random_state[1::3] = np.random.uniform(0.1, 0.9, N)
    random_state[2::3] = 0.05
    starts.append(random_state)
    
    # 5. Another random with different seed
    np.random.seed(123)
    random_state2 = np.zeros(3*N)
    random_state2[0::3] = np.random.uniform(0.1, 0.9, N)
    random_state2[1::3] = np.random.uniform(0.1, 0.9, N)
    random_state2[2::3] = 0.06
    starts.append(random_state2)
    
    # 6. Hexagonal-ish packing
    hex_state = np.zeros(3*N)
    hex_spacing = 0.18
    count = 0
    for row in range(8):
        for col in range(8):
            if count >= N:
                break
            x_pos = 0.1 + col * hex_spacing + (0.09 if row % 2 == 1 else 0)
            y_pos = 0.1 + row * hex_spacing * 0.866
            if x_pos < 0.9 and y_pos < 0.9:
                hex_state[3*count] = x_pos
                hex_state[3*count+1] = y_pos
                hex_state[3*count+2] = 0.09
                count += 1
            if count >= N:
                break
    if count < N:
        # Fill remaining
        for i in range(count, N):
            hex_state[3*i] = np.random.uniform(0.1, 0.9)
            hex_state[3*i+1] = np.random.uniform(0.1, 0.9)
            hex_state[3*i+2] = 0.05
    starts.append(hex_state)
    
    # Time budget per L-BFGS-B refinement
    time_per_refine = max(1.0, (deadline - time.time() - 3.0) / len(starts))
    
    for s_idx, start_z in enumerate(starts):
        if time.time() > deadline - 3.0:
            break
        
        # Project start to valid first
        start_valid = project_to_valid(start_z)
        
        # Try L-BFGS-B refinement
        refined = refine_with_l_bfgs(start_valid, deadline - 3.0)
        
        if refined is not None:
            score = np.sum(refined[2::3])
            if score > best_score:
                best_score = score
                best_z = refined.copy()
        
        # Also try the projected start itself
        score_start = np.sum(start_valid[2::3])
        if score_start > best_score:
            best_score = score_start
            best_z = start_valid.copy()
    
    # Final safety: ensure best_z is valid
    if best_z is None:
        # Trivial fallback
        best_z = np.zeros(3*N)
        best_z[0::3] = 0.5
        best_z[1::3] = 0.5
        best_z[2::3] = 0.01
        best_z = project_to_valid(best_z)
    else:
        best_z = project_to_valid(best_z)
    
    # Convert to (N, 3) array
    result = np.column_stack([best_z[0::3], best_z[1::3], best_z[2::3]])
    
    return result

# EVOLVE-BLOCK-END
