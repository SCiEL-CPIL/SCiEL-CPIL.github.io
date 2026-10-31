"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations (Smooth Squared Penalty)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Pairwise Overlap Violations (Vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Smoother penalty: (violation)^2 / (min_dist^2 + epsilon)
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom
    
    overlap_penalty = np.sum(smooth_overlap[mask])
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _is_valid(pos, radii, tol=1e-7):
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
    
    for _ in range(200):
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

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
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
        
        # Sample 3000 random points for better coverage
        for _ in range(3000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
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
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_lattice():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
            
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        placed = False
        for _ in range(100):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_anneal(pos, radii, steps=500, start_temp=0.05, end_temp=0.001, rng=None):
    """
    Force-directed annealing to escape local optima.
    Applies repulsive forces between overlapping circles and attractive forces
    to pull circles towards the center of available space.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(steps):
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Calculate forces
        forces = np.zeros((N, 2))
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r and dist > 1e-10:
                    # Repulsive force proportional to overlap
                    overlap = sum_r - dist
                    force_mag = overlap * 10.0
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Attractive forces to center (weak)
        for i in range(N):
            forces[i, 0] += (0.5 - pos[i, 0]) * 0.1
            forces[i, 1] += (0.5 - pos[i, 1]) * 0.1
        
        # Update positions with noise
        new_pos = pos + forces * 0.01 + rng.normal(0, temp, (N, 2))
        
        # Keep within bounds
        for i in range(N):
            new_pos[i, 0] = np.clip(new_pos[i, 0], radii[i], 1.0 - radii[i])
            new_pos[i, 1] = np.clip(new_pos[i, 1], radii[i], 1.0 - radii[i])
        
        # Try to expand radii
        new_radii = radii.copy()
        for i in range(N):
            # Max radius allowed by walls
            max_r_wall = min(new_pos[i, 0], 1.0-new_pos[i, 0], new_pos[i, 1], 1.0-new_pos[i, 1])
            
            # Max radius allowed by other circles
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = new_pos[i, 0] - new_pos[j, 0]
                dy = new_pos[i, 1] - new_pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - new_radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            # Expand slightly if possible
            if max_r_circ > new_radii[i]:
                new_radii[i] = new_radii[i] * 0.99 + max_r_circ * 0.01
        
        # Check validity
        valid = True
        for i in range(N):
            if new_pos[i, 0] < new_radii[i] or new_pos[i, 0] > 1.0 - new_radii[i]:
                valid = False
                break
            if new_pos[i, 1] < new_radii[i] or new_pos[i, 1] > 1.0 - new_radii[i]:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = new_pos[i, 0] - new_pos[j, 0]
                    dy = new_pos[i, 1] - new_pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
            pos = new_pos
            radii = new_radii
        else:
            # Reject move
            pass
            
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Strategy: Multi-start with different initializations + force-directed annealing
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Random (4 variations)
    for i in range(4):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    # Multi-stage optimization
    penalty_stages = [50.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        for p_w in penalty_stages:
            if time.time() > deadline - 15:
                break
                
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
                    
            except Exception:
                pass
            
            # Check validity and update best
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
        # Apply force-directed annealing to the best result from this init
        if time.time() < deadline - 20:
            pos_fa, rad_fa = _force_directed_anneal(pos_pp, radii_pp, steps=300, start_temp=0.03, end_temp=0.001)
            pos_fa, rad_fa = _post_process(pos_fa, rad_fa)
            score_fa = np.sum(rad_fa)
            
            if _is_valid(pos_fa, rad_fa) and score_fa > best_score:
                best_score = score_fa
                best_pos = pos_fa.copy()
                best_radii = rad_fa.copy()
                
        # Final polish with high penalty
        if time.time() < deadline - 10:
            x_polish = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x_polish, method='L-BFGS-B', bounds=bounds,
                                  args=(5000.0,),
                                  options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
                
                pos_pol = result.x[:2*N].reshape(N, 2)
                rad_pol = np.maximum(result.x[2*N:], 1e-4)
                pos_pol, rad_pol = _post_process(pos_pol, rad_pol)
                score_pol = np.sum(rad_pol)
                
                if _is_valid(pos_pol, rad_pol) and score_pol > best_score:
                    best_score = score_pol
                    best_pos = pos_pol.copy()
                    best_radii = rad_pol.copy()
                    
            except Exception:
                pass

    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback if invalid
    if not _is_valid(best_pos, best_radii):
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
