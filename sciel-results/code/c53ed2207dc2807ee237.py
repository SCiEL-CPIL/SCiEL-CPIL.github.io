"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(20):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii to make them just touch
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_greedy_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    This creates a valid, dense configuration that serves as a strong starting point.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample random points for better coverage
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98

    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_hexagonal():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.12
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(5):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            # Clip to unit square
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            pos[k] = [x, y]
            radii[k] = r
            k += 1
            
    # If we have fewer than N, fill remaining with small circles in corners
    while k < N:
        if k == 25:
             pos[k] = [0.5, 0.5] # Center
             radii[k] = 0.05
        else:
             pos[k] = [0.1, 0.1]
             radii[k] = 0.05
        k += 1

    # Ensure validity
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_anneal(pos, radii, max_steps=3000, initial_temp=0.05, final_temp=0.001, deadline=None):
    """
    Force-directed simulated annealing to improve packing density.
    Moves circles to resolve overlaps and then grows radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Precompute pairs for efficiency? No, N is small enough.
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
            
        # Cooling schedule
        temp = initial_temp * (final_temp / initial_temp) ** (step / max_steps)
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    # Force magnitude proportional to overlap
                    force = 10.0 * overlap
                    fx[i] -= force * dx / dist
                    fy[i] -= force * dy / dist
                    fx[j] += force * dx / dist
                    fy[j] += force * dy / dist
        
        # Boundary forces
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            # Left wall
            if x < r:
                fx[i] += 5.0 * (r - x)
            # Right wall
            if x > 1 - r:
                fx[i] -= 5.0 * (x - (1 - r))
            # Bottom wall
            if y < r:
                fy[i] += 5.0 * (r - y)
            # Top wall
            if y > 1 - r:
                fy[i] -= 5.0 * (y - (1 - r))
        
        # Update positions
        step_size = temp
        pos += step_size * fx
        pos += step_size * fy
        
        # Clamp to valid range (soft clamp to allow some flexibility for force calculation in next iter)
        # But for validity, we must be inside.
        # Let's clamp strictly to [r, 1-r]
        for i in range(N):
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r, 1 - r)
            pos[i, 1] = np.clip(pos[i, 1], r, 1 - r)
            
        # Check validity and try to grow radii
        is_valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if is_valid:
            # Try to grow radii
            # Calculate max possible growth for each circle
            growths = np.zeros(N)
            for i in range(N):
                max_growth = 0.0
                # Wall constraint
                max_growth = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1]) - radii[i]
                
                # Neighbor constraint
                for j in range(N):
                    if i == j: continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # dist >= r_i + r_j + growth_i + growth_j
                    # If we assume others don't grow, max growth is dist - r_i - r_j
                    # But we want to grow all simultaneously.
                    # Conservative approach: grow by a fraction of the minimum slack.
                    slack = dist - radii[i] - radii[j]
                    if slack < max_growth:
                        max_growth = slack
                        
                # Limit growth to a small step size to maintain stability
                max_growth = np.clip(max_growth, 0.0, 0.01 * temp + 0.0001)
                growths[i] = max_growth
            
            new_radii = radii + growths
            
            # Check if new radii are valid
            valid_growth = True
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-9:
                        valid_growth = False
                        break
                if not valid_growth:
                    break
            
            if valid_growth:
                radii = new_radii
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            else:
                # If growth failed, maybe shrink slightly to allow more movement?
                # Or just keep current radii and let forces resolve overlaps if any appeared due to numerical error
                pass
        else:
            # If invalid, forces will push them apart in next iteration
            pass
            
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # List of initializers to try
    initializers = [
        _initial_greedy_best_fit,
        _initial_hexagonal,
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
        lambda: _initial_random(4),
        lambda: _initial_random(5),
    ]
    
    # Bounds for optimizer
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
        
    # Constraint functions for SLSQP
    def _boundary_constraints(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        cons_x_left = pos[:, 0] - radii
        cons_x_right = 1.0 - pos[:, 0] - radii
        cons_y_bottom = pos[:, 1] - radii
        cons_y_top = 1.0 - pos[:, 1] - radii
        return np.concatenate([cons_x_left, cons_x_right, cons_y_bottom, cons_y_top])

    def _overlap_constraints(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                sum_r = radii[i] + radii[j]
                cons.append(dist_sq - sum_r * sum_r)
        return np.array(cons)

    def constraints_func(x):
        b_cons = _boundary_constraints(x)
        o_cons = _overlap_constraints(x)
        return np.concatenate([b_cons, o_cons])

    def _objective(x):
        radii = x[2*N:]
        return -np.sum(radii)

    # Try each initializer
    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 30:
            break
            
        try:
            pos, radii = init_func()
            x0 = np.concatenate([pos.flatten(), radii])
            
            # Check if this initial state is already good
            if _is_valid(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            
            # Run SLSQP with hard constraints
            if time.time() > deadline - 25:
                continue
                
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 200, 'ftol': 1e-9})
            
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
        except Exception as e:
            continue
            
    if best_pos is None:
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
        best_score = np.sum(best_radii)

    # Refinement Stage 1: Force Annealing on the best result
    if time.time() < deadline - 20:
        try:
            pos_fa, radii_fa = _force_anneal(best_pos, best_radii, max_steps=2000, initial_temp=0.05, final_temp=0.001, deadline=deadline - 5)
            pos_fa, radii_fa = _post_process(pos_fa, radii_fa)
            score_fa = np.sum(radii_fa)
            if _is_valid(pos_fa, radii_fa) and score_fa > best_score:
                best_score = score_fa
                best_pos = pos_fa.copy()
                best_radii = radii_fa.copy()
        except Exception:
            pass

    # Refinement Stage 2: Higher precision SLSQP on the best result
    if time.time() < deadline - 10:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 500, 'ftol': 1e-12})
            
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Refinement Stage 3: Final short Force Annealing pass
    if time.time() < deadline - 2:
        try:
            pos_fa2, radii_fa2 = _force_anneal(best_pos, best_radii, max_steps=1000, initial_temp=0.02, final_temp=0.0005, deadline=deadline - 1)
            pos_fa2, radii_fa2 = _post_process(pos_fa2, radii_fa2)
            score_fa2 = np.sum(radii_fa2)
            if _is_valid(pos_fa2, radii_fa2) and score_fa2 > best_score:
                best_score = score_fa2
                best_pos = pos_fa2.copy()
                best_radii = radii_fa2.copy()
        except Exception:
            pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
