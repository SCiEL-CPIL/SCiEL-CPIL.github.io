"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    # Vectorized distance check
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Check if any pair violates dist < min_dist
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist (since both positive)
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
    
    for _ in range(50):
        valid = True
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            
            # Check against walls
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        # Find min clearance
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale down both radii proportionally to fit
                    # We want r_i' + r_j' <= dist
                    # Let r_i' = r_i * k, r_j' = r_j * k
                    # k * (r_i + r_j) = dist => k = dist / (r_i + r_j)
                    k = dist / (sum_r + 1e-12) * 0.999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_greedy_best_fit():
    """
    Greedy best-fit initialization:
    Place circles one by one at the location that allows the largest possible radius,
    given the already placed circles and the boundaries.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Define a grid of potential centers
    grid_size = 20
    xs = np.linspace(0.05, 0.95, grid_size)
    ys = np.linspace(0.05, 0.95, grid_size)
    
    placed_pos = []
    placed_radii = []
    
    for _ in range(N):
        best_r = -1
        best_pos = None
        
        # Sample points on a fine grid
        for x in xs:
            for y in ys:
                # Calculate max radius allowed by boundaries
                r_bound = min(x, 1.0-x, y, 1.0-y)
                
                # Check against placed circles
                r_max = r_bound
                for px, py, pr in zip(placed_pos, placed_radii):
                    dx = x - px
                    dy = y - py
                    dist = np.sqrt(dx*dx + dy*dy)
                    # If dist > pr, we can fit a circle of radius dist - pr
                    # But we are limited by boundaries too.
                    # Actually, the max radius is min(r_bound, dist - pr) if dist > pr
                    # If dist <= pr, this point is invalid (inside existing circle)
                    if dist > pr:
                        r_candidate = dist - pr
                        if r_candidate < r_max:
                            r_max = r_candidate
                    else:
                        r_max = -1
                        break
                
                if r_max > best_r:
                    best_r = r_max
                    best_pos = (x, y)
        
        if best_r > 0:
            pos[_] = best_pos
            radii[_] = best_r
            placed_pos.append(best_pos[0])
            placed_radii.append(best_pos[1])
            placed_radii.append(best_r)
        else:
            # Fallback: place a tiny circle
            pos[_] = (0.5, 0.5)
            radii[_] = 0.01
            placed_pos.append(0.5)
            placed_radii.append(0.5)
            placed_radii.append(0.01)
            
    return pos, radii

def _initial_hexagonal():
    """
    Hexagonal grid initialization.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069
    # For 26 circles in unit square, average area ~ 1/26 ~ 0.038
    # r ~ sqrt(0.038/pi) ~ 0.11
    # But we need to fit them in.
    
    # Let's try a simple grid first, then hexagonal adjustment
    # 5x5 grid = 25 circles + 1
    # Spacing s. 5s <= 1 => s <= 0.2. r = s/2 = 0.1.
    # 25 * 0.1 = 2.5.
    
    # Hexagonal:
    # Row 1: 5 circles
    # Row 2: 4 circles (shifted)
    # Row 3: 5 circles
    # Row 4: 4 circles
    # Row 5: 5 circles
    # Total 23. Need 3 more.
    
    # Let's just use a jittered grid for simplicity and robustness
    rng = np.random.RandomState(42)
    for i in range(N):
        # Map i to a grid position
        row = i // 5
        col = i % 5
        x = (col + 0.5) / 5.0
        y = (row + 0.5) / 5.0
        pos[i] = [x, y]
        radii[i] = 0.09 # Start small
        
    return pos, radii

def _objective(x, penalty_weight=1000.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Pairwise Overlap Violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = np.maximum(min_dist_sq, 1e-12)
    
    smooth_overlap = (violation ** 2) / (4.0 * denom)
    overlap_penalty = np.sum(smooth_overlap[mask])
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _vectorized_refine(pos, radii, start_time, deadline):
    """
    Vectorized local refinement using gradient ascent on radii and positions.
    This is a fast, derivative-free-ish approach that iteratively pushes circles apart
    and grows radii.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Parameters
    step_size = 0.01
    decay = 0.999
    
    while time.time() < deadline - 5:
        # Calculate forces/gradients
        grad_pos = np.zeros_like(pos)
        grad_radii = np.zeros_like(radii)
        
        # 1. Boundary constraints (push inside)
        # If x < r, push x right (grad x += 1, grad r -= 1)
        # If x > 1-r, push x left (grad x -= 1, grad r -= 1)
        # If y < r, push y up
        # If y > 1-r, push y down
        
        left_viol = np.maximum(0, radii - pos[:, 0])
        right_viol = np.maximum(0, pos[:, 0] - (1.0 - radii))
        bottom_viol = np.maximum(0, radii - pos[:, 1])
        top_viol = np.maximum(0, pos[:, 1] - (1.0 - radii))
        
        grad_pos[:, 0] += left_viol - right_viol
        grad_pos[:, 1] += bottom_viol - top_viol
        grad_radii -= (left_viol + right_viol + bottom_viol + top_viol)
        
        # 2. Overlap constraints
        # For each pair (i, j) where dist < r_i + r_j
        # Push apart: grad_pos[i] += (pos[i]-pos[j])/dist * overlap
        # Grad_radii[i] -= overlap
        
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq + 1e-12)
        
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0, min_dist - dist)
        
        # Force magnitude
        force_mag = overlap / (dist + 1e-12)
        
        # Sum forces
        # grad_pos[i] = sum_j force_mag[i,j] * (pos[i]-pos[j])/dist[i,j]
        # This is equivalent to summing force_mag * (dx/dx, dy/dy)
        
        # force_mag is symmetric, dx is antisymmetric.
        # grad_pos_x[i] = sum_j force_mag[i,j] * dx[i,j] / dist[i,j]
        # Note: dx[i,j] = pos[i,x] - pos[j,x]
        
        force_x = force_mag * dx
        force_y = force_mag * dy
        
        # Sum over j (axis 1) for each i
        grad_pos[:, 0] += np.sum(force_x, axis=1)
        grad_pos[:, 1] += np.sum(force_y, axis=1)
        
        # Grad radii
        # grad_radii[i] -= sum_j overlap[i,j]
        grad_radii -= np.sum(overlap, axis=1)
        
        # Update
        pos += step_size * grad_pos
        radii += step_size * grad_radii
        
        # Clip
        for i in range(N):
            max_r = min(pos[i, 0], 1.0-pos[i, 0], pos[i, 1], 1.0-pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
            if radii[i] < 1e-4:
                radii[i] = 1e-4
        
        # Check validity and update best
        if _is_valid(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # If invalid, shrink radii slightly to recover
            radii *= 0.99
        
        # Decay step size
        step_size *= decay
        
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # 1. Generate multiple initializations
    pos1, radii1 = _initial_greedy_best_fit()
    pos2, radii2 = _initial_hexagonal()
    
    # Select the better one
    score1 = np.sum(radii1)
    score2 = np.sum(radii2)
    
    pos, radii = (pos1, radii1) if score1 > score2 else (pos2, radii2)
    
    # 2. Vectorized Refinement (Fast, robust)
    pos_v, radii_v, score_v = _vectorized_refine(pos, radii, start_time, deadline)
    
    if score_v > np.sum(radii):
        pos, radii = pos_v, radii_v
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 3. Scipy Optimization (L-BFGS-B)
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-4, 0.5))
        
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          args=(1000.0,),
                          options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.fun < _objective(x0, 1000.0) - 1e-10:
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
        pass
        
    # 4. Powell for fine tuning
    if time.time() < deadline - 10:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              args=(1000.0,),
                              options={'maxiter': 3000, 'ftol': 1e-12})
            
            if result.fun < _objective(x0, 1000.0) - 1e-10:
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
            pass
            
    # 5. Final Polish with lower penalty
    if time.time() < deadline - 5:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(200.0,),
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < _objective(x0, 200.0) - 1e-10:
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
            pass

    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback
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
