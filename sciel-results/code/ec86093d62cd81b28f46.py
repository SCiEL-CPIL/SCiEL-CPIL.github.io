import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Initialization: Better hexagonal-like packing
    np.random.seed(42)
    
    # Use a denser initial pattern
    # Start with a grid and perturb
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    # Place circles in a roughly hexagonal pattern
    # 5 columns, varying rows
    col_width = 1.0 / 5.0
    row_height = 1.0 / 5.0
    
    idx = 0
    for i in range(5):
        x_base = (i + 0.5) * col_width
        if i % 2 == 0:
            for j in range(5):
                if idx >= N:
                    break
                xs[idx] = x_base
                ys[idx] = (j + 0.5) * row_height
                rs[idx] = 0.04
                idx += 1
        else:
            for j in range(5):
                if idx >= N:
                    break
                xs[idx] = x_base
                ys[idx] = (j + 0.5) * row_height
                rs[idx] = 0.04
                idx += 1
    
    # If we have more than N, trim
    xs = xs[:N]
    ys = ys[:N]
    rs = rs[:N]
    
    # Add small random perturbation
    xs += np.random.uniform(-0.02, 0.02, N)
    ys += np.random.uniform(-0.02, 0.02, N)
    xs = np.clip(xs, 0.05, 0.95)
    ys = np.clip(ys, 0.05, 0.95)
    
    # State: [x0, y0, r0, x1, y1, r1, ...]
    state = np.zeros(3 * N)
    state[0::3] = xs
    state[1::3] = ys
    state[2::3] = rs

    # Vectorized energy calculation
    def calc_energy(state):
        xs = state[0::3]
        ys = state[1::3]
        rs = state[2::3]
        
        # Boundary penalty (vectorized)
        dist_left = xs
        dist_right = 1 - xs
        dist_bottom = ys
        dist_top = 1 - ys
        min_bound = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_bottom, dist_top))
        bound_viol = np.maximum(0, rs - min_bound)
        pen_bound = np.sum(bound_viol ** 2)
        
        # Overlap penalty (vectorized using broadcasting)
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_mat = np.sqrt(dx**2 + dy**2)
        sum_r = rs[:, None] + rs[None, :]
        overlap_viol = np.maximum(0, sum_r - dist_mat)
        # Only count upper triangle to avoid double counting
        triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pen_overlap = np.sum(overlap_viol[triu_mask] ** 2)
        
        lambda_pen = 10.0
        return -np.sum(rs) + lambda_pen * (pen_bound + pen_overlap)

    # 2. Simulated Annealing with better parameters
    T = 0.5
    cooling = 0.9995
    step_size_pos = 0.03
    step_size_rad = 0.02
    
    best_state = state.copy()
    best_energy = calc_energy(state)
    current_energy = best_energy
    
    # Track best sum of radii separately for tracking
    best_sum_r = np.sum(state[2::3])
    
    while time.time() < deadline - 25.0:  # Leave 25s for refinement
        # Propose move
        new_state = state.copy()
        i = np.random.randint(N)
        
        # Move position
        new_state[i*3] += np.random.uniform(-step_size_pos, step_size_pos)
        new_state[i*3+1] += np.random.uniform(-step_size_pos, step_size_pos)
        
        # Move radius
        new_state[i*3+2] += np.random.uniform(-step_size_rad, step_size_rad)
        
        # Clip to valid ranges
        new_state[i*3] = np.clip(new_state[i*3], 0.01, 0.99)
        new_state[i*3+1] = np.clip(new_state[i*3+1], 0.01, 0.99)
        new_state[i*3+2] = np.clip(new_state[i*3+2], 0.01, 0.4)
        
        new_energy = calc_energy(new_state)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            state = new_state
            current_energy = new_energy
            sum_r = np.sum(state[2::3])
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_state = state.copy()
                best_energy = current_energy
        
        T *= cooling
        if T < 1e-4:
            T = 1e-4

    # 3. Refinement with SLSQP using penalty method instead of explicit constraints
    # This is faster and more robust
    def penalty_objective(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Boundary penalty
        min_bound = np.minimum(np.minimum(xs, 1-xs), np.minimum(ys, 1-ys))
        bound_viol = np.maximum(0, rs - min_bound)
        
        # Overlap penalty
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_mat = np.sqrt(dx**2 + dy**2)
        sum_r = rs[:, None] + rs[None, :]
        overlap_viol = np.maximum(0, sum_r - dist_mat)
        triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        pen = np.sum(bound_viol**2) + np.sum(overlap_viol[triu_mask]**2)
        return -np.sum(rs) + 100.0 * pen
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
        bounds.append((0.01, 0.4))   # r
    
    # Try SLSQP with penalty
    try:
        res = minimize(penalty_objective, best_state, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_energy:
            best_state = res.x
            best_energy = res.fun
    except Exception:
        pass
    
    # Second refinement with tighter penalty
    def penalty_objective_tight(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        min_bound = np.minimum(np.minimum(xs, 1-xs), np.minimum(ys, 1-ys))
        bound_viol = np.maximum(0, rs - min_bound)
        
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_mat = np.sqrt(dx**2 + dy**2)
        sum_r = rs[:, None] + rs[None, :]
        overlap_viol = np.maximum(0, sum_r - dist_mat)
        triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        pen = np.sum(bound_viol**2) + np.sum(overlap_viol[triu_mask]**2)
        return -np.sum(rs) + 500.0 * pen
    
    try:
        res2 = minimize(penalty_objective_tight, best_state, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if res2.fun < best_energy:
            best_state = res2.x
            best_energy = res2.fun
    except Exception:
        pass

    # 4. Final feasibility check and cleanup
    xs = best_state[0::3].copy()
    ys = best_state[1::3].copy()
    rs = best_state[2::3].copy()
    
    # Ensure strict feasibility with iterative repair
    for iteration in range(50):
        valid = True
        
        # Check and fix boundary violations
        for i in range(N):
            min_dist = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
            if rs[i] > min_dist:
                rs[i] = min_dist * 0.999
                valid = False
        
        # Check and fix overlap violations
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    # Shrink the larger radius more
                    total_r = rs[i] + rs[j]
                    if total_r > 0:
                        target = dist * 0.999
                        scale_i = (target * rs[i]) / total_r if total_r > 1e-10 else target / 2
                        scale_j = (target * rs[j]) / total_r if total_r > 1e-10 else target / 2
                        rs[i] = max(0.001, scale_i)
                        rs[j] = max(0.001, scale_j)
                    valid = False
        
        if valid:
            break
    
    # Final clip to ensure no negative radii
    rs = np.maximum(rs, 0.001)
    
    result = np.column_stack((xs, ys, rs))
    return result
