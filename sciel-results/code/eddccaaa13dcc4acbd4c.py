"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_violations(pos, radii):
    """
    Compute boundary and overlap violations.
    Returns total violation magnitude (for penalty) and individual terms if needed.
    """
    # Boundary violations
    # x - r < 0  => r - x > 0
    # x + r > 1  => x - (1-r) > 0
    bv_left = np.maximum(0, radii - pos[:, 0])
    bv_right = np.maximum(0, pos[:, 0] + radii - 1.0)
    bv_bottom = np.maximum(0, radii - pos[:, 1])
    bv_top = np.maximum(0, pos[:, 1] + radii - 1.0)
    
    boundary_pen = np.sum(bv_left**2) + np.sum(bv_right**2) + np.sum(bv_bottom**2) + np.sum(bv_top**2)
    
    # Overlap violations
    # d_ij < r_i + r_j  => (r_i + r_j)^2 - d_ij^2 > 0
    # We use squared distance to avoid sqrt in the main loop for speed, 
    # but for gradient-based polish we might need actual distances.
    # For the penalty objective, squared violation is fine.
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    # Only consider i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Violation is positive if min_dist_sq > dist_sq
    viol_sq = np.maximum(0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol_sq[mask] ** 2) # Squaring again to make penalty very steep
    
    return boundary_pen + overlap_pen

def _de_objective(x):
    """
    Objective function for Differential Evolution.
    Minimizes -sum(radii) + large_penalty * violations.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    viol = _compute_violations(pos, radii)
    
    # Penalty weight is high to force validity
    penalty_weight = 1000.0
    score = -np.sum(radii) + penalty_weight * viol
    
    return score

def _polish_configuration(pos, radii, time_limit=5.0):
    """
    Local search to maximize sum of radii starting from a valid or near-valid configuration.
    Strategy:
    1. Ensure validity by shrinking radii if necessary.
    2. Iteratively try to increase radii and adjust positions to maintain validity.
    """
    pos = pos.copy()
    radii = radii.copy()
    start_time = time.time()
    
    # Step 1: Strictly enforce validity by shrinking radii
    def enforce_validity(pos, radii):
        # Boundary
        for i in range(N):
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.99999
        
        # Overlaps
        # Iterate a few times to resolve overlaps
        for _ in range(50):
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    sum_r = radii[i] + radii[j]
                    if dist < sum_r:
                        # Shrink both radii proportionally to fit
                        # dist = r_i + r_j
                        # scale = dist / sum_r
                        scale = dist / sum_r * 0.99999
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
            if not changed:
                break
        return pos, radii

    pos, radii = enforce_validity(pos, radii)
    
    # Step 2: Gradient-like local search
    # We want to maximize sum(r). 
    # We can do this by:
    # 1. Try to increase a random radius.
    # 2. If it causes violation, move positions to alleviate it.
    # 3. If it still causes violation, shrink it back.
    
    rng = np.random.RandomState(42)
    
    for _ in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a circle to expand
        i = rng.randint(N)
        
        # Try to expand radius slightly
        old_r = radii[i]
        # Adaptive step size
        step = 0.001 * (1.0 + old_r) 
        new_r = old_r + step
        
        # Check if new radius is valid with current positions
        # Boundary check
        max_r_wall = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if new_r > max_r_wall:
            new_r = max_r_wall * 0.99999
            if new_r <= old_r:
                continue
        
        # Overlap check
        valid = True
        for j in range(N):
            if i == j: continue
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < new_r + radii[j]:
                valid = False
                break
        
        if valid:
            radii[i] = new_r
        else:
            # Attempt to move circle i to create space
            # Calculate repulsion forces from overlapping circles
            fx, fy = 0.0, 0.0
            for j in range(N):
                if i == j: continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dist = 1e-6
                    dx = rng.uniform(-0.01, 0.01)
                    dy = rng.uniform(-0.01, 0.01)
                
                overlap = (new_r + radii[j]) - dist
                if overlap > 0:
                    # Push away
                    fx += (dx / dist) * overlap
                    fy += (dy / dist) * overlap
            
            # Also push away from walls
            if pos[i,0] - new_r < 0:
                fx += (new_r - pos[i,0])
            if pos[i,0] + new_r > 1:
                fx -= (pos[i,0] + new_r - 1)
            if pos[i,1] - new_r < 0:
                fy += (new_r - pos[i,1])
            if pos[i,1] + new_r > 1:
                fy -= (pos[i,1] + new_r - 1)
            
            # Move
            move_step = 0.01
            pos[i,0] += fx * move_step
            pos[i,1] += fy * move_step
            
            # Clamp to bounds
            pos[i,0] = np.clip(pos[i,0], 0.0, 1.0)
            pos[i,1] = np.clip(pos[i,1], 0.0, 1.0)
            
            # After moving, try to set radius to max possible
            # This is a greedy approach: expand as much as possible
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            for j in range(N):
                if i == j: continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            
            if max_r > radii[i]:
                radii[i] = max_r * 0.99999
            else:
                # If we can't even fit current radius, shrink it
                radii[i] = max_r * 0.99999 if max_r > 0 else 0.001

    # Final strict validity check and shrink
    pos, radii = enforce_validity(pos, radii)
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # 1. Global Search using Differential Evolution
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    for _ in range(N):
        bounds.append((0.0, 0.5)) # r
        
    # DE parameters
    # popsize 10-20 is usually good for 78 vars
    # maxiter kept low because we will polish
    # tol for stopping
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Run DE for a significant portion of the time
    de_time_limit = 60.0
    de_deadline = start_time + de_time_limit
    
    try:
        # We use 'best1bin' strategy
        result = differential_evolution(
            _de_objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6,
            seed=42,
            init='latinhypercube',
            # Using a callback to check time is not directly supported in simple way,
            # but we can estimate. 
            # Better to just run a fixed number of iterations that fits in time.
            # 200 iterations * 15 pop * 78 vars is manageable.
        )
        
        if result.success or result.fun < 0: # result.fun is -sum(r) + penalty
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            
            # Polish this result
            pos_pol, rad_pol = _polish_configuration(pos_opt, radii_opt, time_limit=deadline - time.time() - 5)
            
            # Validate
            # Check if valid
            valid = True
            for i in range(N):
                if pos_pol[i,0] - rad_pol[i] < -1e-9 or pos_pol[i,0] + rad_pol[i] > 1.0 + 1e-9:
                    valid = False
                    break
                if pos_pol[i,1] - rad_pol[i] < -1e-9 or pos_pol[i,1] + rad_pol[i] > 1.0 + 1e-9:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos_pol[i,0] - pos_pol[j,0]
                        dy = pos_pol[i,1] - pos_pol[j,1]
                        dist_sq = dx*dx + dy*dy
                        sum_r = rad_pol[i] + rad_pol[j]
                        if dist_sq < sum_r*sum_r - 1e-9:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                score = np.sum(rad_pol)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pol
                    best_radii = rad_pol
                    
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Fallback / Alternative: If DE didn't produce a good result or failed,
    # use a simpler multi-start local search.
    
    if best_score < 1.0: # Threshold for "bad" result (sum of 26 circles of r~0.1 is 2.6)
         # Multi-start L-BFGS
         bounds_lbfgs = [(0.0, 1.0), (0.0, 1.0)] * N + [(0.0, 0.5)] * N
         
         for seed in range(5):
             if time.time() > deadline - 5:
                 break
             rng = np.random.RandomState(seed)
             x0 = np.concatenate([rng.uniform(0.1, 0.9, 2*N), rng.uniform(0.05, 0.15, N)])
             
             try:
                 res = minimize(_de_objective, x0, method='L-BFGS-B', bounds=bounds_lbfgs,
                                options={'maxiter': 200, 'ftol': 1e-10})
                 if res.fun < _de_objective(x0):
                     x_opt = res.x
                     pos_opt = x_opt[:2*N].reshape(N, 2)
                     radii_opt = x_opt[2*N:]
                     
                     pos_pol, rad_pol = _polish_configuration(pos_opt, radii_opt, time_limit=min(5.0, deadline - time.time() - 2))
                     
                     # Quick validity check
                     valid = True
                     for i in range(N):
                         if pos_pol[i,0] - rad_pol[i] < -1e-9 or pos_pol[i,0] + rad_pol[i] > 1.0 + 1e-9:
                             valid = False
                             break
                         if pos_pol[i,1] - rad_pol[i] < -1e-9 or pos_pol[i,1] + rad_pol[i] > 1.0 + 1e-9:
                             valid = False
                             break
                     if valid:
                         for i in range(N):
                             for j in range(i+1, N):
                                 dx = pos_pol[i,0] - pos_pol[j,0]
                                 dy = pos_pol[i,1] - pos_pol[j,1]
                                 dist_sq = dx*dx + dy*dy
                                 sum_r = rad_pol[i] + rad_pol[j]
                                 if dist_sq < sum_r*sum_r - 1e-9:
                                     valid = False
                                     break
                             if not valid: break
                     
                     if valid:
                         score = np.sum(rad_pol)
                         if score > best_score:
                             best_score = score
                             best_pos = pos_pol
                             best_radii = rad_pol
             except Exception:
                 pass

    # Final Fallback: Grid
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        # Last one
        if k < N:
             pos[k] = [0.5, 0.5]
             radii[k] = 0.01
        best_pos = pos
        best_radii = radii
        # Polish even the fallback
        best_pos, best_radii = _polish_configuration(best_pos, best_radii, time_limit=5.0)

    # Final strict validation and shrink
    def final_enforce(pos, radii):
        # Boundary
        for i in range(N):
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.99999999
        
        # Overlaps
        for _ in range(100):
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    sum_r = radii[i] + radii[j]
                    if dist < sum_r:
                        scale = dist / sum_r * 0.99999999
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
            if not changed:
                break
        return pos, radii

    best_pos, best_radii = final_enforce(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
