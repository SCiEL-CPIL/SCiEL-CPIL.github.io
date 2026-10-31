"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _energy(pos, radii, penalty_weight=1000.0):
    """
    Calculate the energy (cost) of the configuration.
    Lower is better. We want to minimize penalty and maximize radii.
    Energy = penalty - sum(radii)
    """
    # Boundary violations
    # x - r < 0 => r > x => violation
    # x + r > 1 => r > 1-x => violation
    bv_x_left = np.maximum(0.0, radii - pos[:, 0])
    bv_x_right = np.maximum(0.0, radii - (1.0 - pos[:, 0]))
    bv_y_bottom = np.maximum(0.0, radii - pos[:, 1])
    bv_y_top = np.maximum(0.0, radii - (1.0 - pos[:, 1]))
    
    boundary_viol = bv_x_left + bv_x_right + bv_y_bottom + bv_y_top
    boundary_pen = np.sum(boundary_viol ** 2)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Overlap if dist_sq < min_dist_sq
    # Violation amount = min_dist_sq - dist_sq (squared space)
    # Using linear violation in distance space is more stable for gradients/SA
    dist = np.sqrt(dist_sq)
    overlap_lin = np.maximum(0.0, min_dist - dist)
    overlap_pen = np.sum(overlap_lin[mask] ** 2)
    
    total_pen = boundary_pen + overlap_pen
    score = np.sum(radii)
    
    return penalty_weight * total_pen - score

def _is_valid(pos, radii, tol=1e-8):
    """
    Strict validity check.
    """
    if np.any(pos[:, 0] < -tol) or np.any(pos[:, 0] > 1.0 + tol):
        return False
    if np.any(pos[:, 1] < -tol) or np.any(pos[:, 1] > 1.0 + tol):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # dist < min_dist => dist_sq < min_dist^2
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _shrink_to_valid(pos, radii, tol=1e-6):
    """
    Shrink radii and adjust positions slightly to ensure validity.
    This is a safety net.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Iterative shrinking
    for _ in range(100):
        valid = True
        
        # Check bounds
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + tol:
                radii[i] = max(0.0, max_r - tol)
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - tol:
                    # Shrink both proportionally to fit
                    # dist >= r_i + r_j
                    # k * (r_i + r_j) <= dist => k <= dist / (r_i + r_j)
                    k = max(0.0, dist / (sum_r + 1e-12))
                    k = min(k, 1.0) # Only shrink
                    radii[i] = radii[i] * k
                    radii[j] = radii[j] * k
                    valid = False
        
        if valid:
            break
            
    return pos, radii

def _sa_packing(init_pos, init_radii, steps=20000, seed=0, deadline=None):
    """
    Simulated Annealing to pack circles with adaptive step sizes.
    """
    rng = np.random.RandomState(seed)
    pos = init_pos.copy().astype(float)
    radii = init_radii.copy().astype(float)
    
    current_energy = _energy(pos, radii)
    best_energy = current_energy
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    T = 0.15 # Initial temperature
    min_T = 1e-7
    
    # Adaptive step size parameters
    pos_step = 0.08
    rad_step = 0.02
    
    # Track acceptance rates for adaptive step sizing
    accept_count = 0
    total_count = 0
    last_adjust_step = 0
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Calculate cooling rate based on progress
        progress = step / max(steps, 1)
        T = max(min_T, 0.15 * (1 - progress) ** 2 + 0.001 * (1 - progress))
        
        # Generate a move
        # Randomly choose to move a position or grow/shrink a radius
        if rng.random() < 0.65:
            # Move position of a random circle
            i = rng.randint(N)
            delta_x = rng.normal(0, pos_step)
            delta_y = rng.normal(0, pos_step)
            
            old_x, old_y = pos[i]
            pos[i, 0] += delta_x
            pos[i, 1] += delta_y
            
            # Keep within bounds roughly to avoid huge penalties
            pos[i, 0] = np.clip(pos[i, 0], 0, 1)
            pos[i, 1] = np.clip(pos[i, 1], 0, 1)
            
            new_energy = _energy(pos, radii)
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                current_energy = new_energy
                accept_count += 1
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            else:
                # Reject move
                pos[i, 0] = old_x
                pos[i, 1] = old_y
        else:
            # Grow or shrink radius of a random circle
            i = rng.randint(N)
            # Try to grow more often, especially early on
            grow_prob = 0.7 if progress < 0.5 else 0.5
            if rng.random() < grow_prob:
                delta_r = rng.uniform(0, rad_step)
            else:
                delta_r = rng.uniform(-rad_step, 0)
                
            old_r = radii[i]
            radii[i] = max(1e-4, radii[i] + delta_r)
            
            new_energy = _energy(pos, radii)
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                current_energy = new_energy
                accept_count += 1
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            else:
                # Reject move
                radii[i] = old_r
        
        total_count += 1
        
        # Adaptive step size adjustment every 1000 steps
        if step - last_adjust_step >= 1000:
            accept_rate = accept_count / max(total_count - last_adjust_step, 1)
            # Target acceptance rate ~0.2-0.4
            if accept_rate < 0.15:
                # Too many rejections, increase step size
                pos_step *= 1.2
                rad_step *= 1.2
            elif accept_rate > 0.5:
                # Too many acceptances, decrease step size
                pos_step *= 0.85
                rad_step *= 0.85
            
            # Clamp step sizes
            pos_step = np.clip(pos_step, 0.01, 0.15)
            rad_step = np.clip(rad_step, 0.005, 0.05)
            
            accept_count = 0
            last_adjust_step = step
        
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """
    Final high-precision polishing using L-BFGS-B with multiple penalty weights.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-4, 0.5))
        
    def _objective(x, penalty_weight=1000.0):
        pos = x[:2*N].reshape(N, 2)
        radii = np.maximum(x[2*N:], 1e-6)
        
        # Boundary violations
        bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
             np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
        boundary_pen = np.sum(bv * bv)
        
        # Overlap violations
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        min_dist_sq = min_dist * min_dist
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        viol = np.maximum(0.0, min_dist_sq - dist_sq)
        overlap_pen = np.sum(viol[mask] * viol[mask])
        
        return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

    best_x = x.copy()
    best_obj = _objective(x, 1000.0)
    
    # Try multiple penalty weights to escape local minima
    for pw in [500.0, 1000.0, 2000.0, 5000.0]:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(pw,),
                              options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12,
                                       'maxls': 50})
            obj_val = _objective(result.x, pw)
            if obj_val < best_obj - 1e-12:
                best_obj = obj_val
                best_x = result.x.copy()
                x = result.x.copy()
        except Exception:
            pass
    
    # Final polish with high penalty weight
    try:
        result = minimize(_objective, best_x, method='L-BFGS-B', bounds=bounds,
                          args=(10000.0,),
                          options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
        if _objective(result.x, 10000.0) < _objective(best_x, 10000.0) - 1e-14:
            best_x = result.x.copy()
    except Exception:
        pass
    
    pos_out = best_x[:2*N].reshape(N, 2)
    radii_out = np.maximum(best_x[2*N:], 1e-4)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    inits = []
    
    # 1. Grid Init
    pos_grid = np.zeros((N, 2))
    rad_grid = np.zeros(N)
    spacing = 1.0 / 5.0
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos_grid[idx] = [(i + 0.5) * spacing, (j + 0.5) * spacing]
                rad_grid[idx] = spacing * 0.45
    if N > 25:
        pos_grid[25] = [0.5, 0.5]
        rad_grid[25] = 0.05
    inits.append((pos_grid, rad_grid))
    
    # 2. Random Init
    rng = np.random.RandomState(42)
    pos_rand = rng.uniform(0.1, 0.9, (N, 2))
    rad_rand = rng.uniform(0.05, 0.1, N)
    inits.append((pos_rand, rad_rand))
    
    # 3. Hexagonal Init
    pos_hex = np.zeros((N, 2))
    rad_hex = np.zeros(N)
    r = 0.11
    dx = r * np.sqrt(3) * 1.05
    dy = r * 1.5 * 1.05
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = r + offset + col * dx
            if 0 <= x <= 1 and 0 <= y <= 1:
                pos_hex[k] = [x, y]
                rad_hex[k] = r
                k += 1
        if k >= N:
            break
    while k < N:
        pos_hex[k] = [0.5, 0.5]
        rad_hex[k] = 0.01
        k += 1
    inits.append((pos_hex, rad_hex))
    
    # 4. Corner-focused Init (often good for packing)
    pos_corner = np.zeros((N, 2))
    rad_corner = np.zeros(N)
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for i, c in enumerate(corners):
        pos_corner[i] = c
        rad_corner[i] = 0.12
    # Fill rest with edge-focused positions
    edges = [
        [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5],
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.5, 0.5]
    ]
    for i in range(4, N):
        if i - 4 < len(edges):
            pos_corner[i] = edges[i - 4]
            rad_corner[i] = 0.06
        else:
            pos_corner[i] = rng.uniform(0.2, 0.8, 2)
            rad_corner[i] = 0.05
    inits.append((pos_corner, rad_corner))
    
    # 5. Dense corner cluster init
    pos_dense = np.zeros((N, 2))
    rad_dense = np.zeros(N)
    # Place 4 large circles in corners
    corner_pos = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for i, cp in enumerate(corner_pos):
        pos_dense[i] = cp
        rad_dense[i] = 0.14
    # Place remaining circles in between
    mid_positions = [
        [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
        [0.5, 0.5]
    ]
    for i in range(4, N):
        if i - 4 < len(mid_positions):
            pos_dense[i] = mid_positions[i - 4]
            rad_dense[i] = 0.04
        else:
            pos_dense[i] = rng.uniform(0.2, 0.8, 2)
            rad_dense[i] = 0.03
    inits.append((pos_dense, rad_dense))
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        # Run SA
        # Adjust steps based on remaining time
        time_left = deadline - time.time()
        steps = int(min(30000, time_left * 600)) # Rough estimate
        
        pos_sa, rad_sa = _sa_packing(pos_init, rad_init, steps=steps, seed=idx*100 + 42, deadline=deadline - 8)
        
        # Polish
        if time.time() < deadline - 4:
            pos_pol, rad_pol = _polish(pos_sa, rad_sa, deadline)
            
            # Validate and shrink
            pos_val, rad_val = _shrink_to_valid(pos_pol, rad_pol)
            
            if _is_valid(pos_val, rad_val):
                score = np.sum(rad_val)
                if score > best_score:
                    best_score = score
                    best_pos = pos_val.copy()
                    best_radii = rad_val.copy()
                    
    # If no valid solution found, fallback to a simple grid
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                if i*5+j < N-1:
                    pos[i*5+j] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[i*5+j] = 0.09
        pos[N-1] = [0.5, 0.02]
        radii[N-1] = 0.01
        best_pos, best_radii = _shrink_to_valid(pos, radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
