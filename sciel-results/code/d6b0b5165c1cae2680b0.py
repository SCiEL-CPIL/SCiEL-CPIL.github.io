"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + smooth penalty for violations.
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
    
    # 2. Pairwise Overlap Violations (Vectorized, smooth)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Smooth penalty: (violation)^2 / (min_dist^2 + epsilon)
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom
    
    overlap_penalty = np.sum(smooth_overlap[mask])
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _objective_grad(x, penalty_weight=500.0):
    """
    Compute objective and gradient for L-BFGS-B.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    radii_clipped = np.maximum(radii, 1e-6)
    
    # Gradient of -sum(radii)
    grad = np.zeros_like(x)
    grad[2*N:] = -1.0
    
    # Boundary penalties
    left_viol = np.maximum(0.0, radii_clipped - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii_clipped))
    bottom_viol = np.maximum(0.0, radii_clipped - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii_clipped))
    
    # d/dx_i of left_viol_i = -1 if viol > 0
    # d/dr_i of left_viol_i = 1 if viol > 0
    # Penalty is sum(viol^2), so d/dvar = 2*viol * d(viol)/dvar
    
    # x-coords
    mask_left = left_viol > 0
    mask_right = right_viol > 0
    mask_bottom = bottom_viol > 0
    mask_top = top_viol > 0
    
    # grad[x_i] from left: -2 * left_viol (if > 0)
    # grad[x_i] from right: +2 * right_viol (if > 0)
    grad[:N] += -2.0 * np.where(mask_left, left_viol, 0.0)
    grad[:N] += 2.0 * np.where(mask_right, right_viol, 0.0)
    # grad[y_i]
    grad[N:2*N] += -2.0 * np.where(mask_bottom, bottom_viol, 0.0)
    grad[N:2*N] += 2.0 * np.where(mask_top, top_viol, 0.0)
    # grad[r_i]
    grad[2*N:] += 2.0 * (np.where(mask_left, left_viol, 0.0) + 
                          np.where(mask_right, right_viol, 0.0) +
                          np.where(mask_bottom, bottom_viol, 0.0) +
                          np.where(mask_top, top_viol, 0.0))
    
    # Pairwise overlap
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    dist_sq_safe = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq_safe)
    
    min_dist = radii_clipped[:, np.newaxis] + radii_clipped[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom
    
    # Gradient of smooth_overlap[i,j] w.r.t. pos and radii
    # f_ij = v_ij^2 / D_ij where v_ij = max(0, M_ij^2 - D_ij^2), D_ij = M_ij^2 + eps
    # df/dv = 2v/D - v^2/D^2
    # dv/du = -1 where u = dist_sq
    # du/d(x_i) = 2(x_i - x_j)
    # So df/d(x_i) = (2v/D - v^2/D^2) * (-1) * 2(x_i - x_j)
    
    v_pos = np.maximum(0.0, min_dist_sq - dist_sq)
    active = v_pos > 0
    active = active & mask
    
    # Coefficient for dist_sq gradient: c_ij = -(2*v_pos/D - v_pos^2/D^2)
    c = np.zeros((N, N))
    if np.any(active):
        D_a = min_dist_sq[active] + 1e-8
        v_a = v_pos[active]
        c[active] = -(2.0 * v_a / D_a - v_a**2 / D_a**2)
    
    # Make symmetric for the full matrix product
    C = c + c.T  # symmetric
    
    # grad[x_i] += sum_j C_ij * 2 * (x_i - x_j)
    # = 2 * (x_i * sum_j C_ij - sum_j C_ij * x_j)
    sum_c_row = np.sum(C, axis=1)
    grad_x_vec = 2.0 * (x[:N, np.newaxis] * sum_c_row[:, np.newaxis] - C.dot(x[:N]))
    grad_y_vec = 2.0 * (x[N:2*N, np.newaxis] * sum_c_row[:, np.newaxis] - C.dot(x[N:2*N]))
    
    grad[:N] += grad_x_vec
    grad[N:2*N] += grad_y_vec
    
    # Gradient w.r.t. radii
    # df/dr_i = (2v/D - v^2/D^2) * dv/dr_i
    # dv/dr_i = 2*M_ij where M_ij = r_i + r_j
    D_safe = min_dist_sq + 1e-8
    H = np.zeros((N, N))
    if np.any(active):
        v_a = v_pos[active]
        M_a = min_dist[active]
        D_a = D_safe[active]
        val = (2.0 * v_a / D_a - v_a**2 / D_a**2) * 2.0 * M_a
        H[active] = val
    
    # Make symmetric
    H_sym = H + H.T
    grad_r = np.sum(H_sym, axis=1)
    grad[2*N:] += grad_r
    
    # Apply penalty weight to the penalty part only
    val = _objective(x, penalty_weight)
    
    # The gradient currently has:
    # grad = [-1 for radii] + penalty_grad (unweighted)
    # We need: grad_total = [-1 for radii] + penalty_weight * penalty_grad
    # So: grad_total = grad + (penalty_weight - 1) * penalty_grad
    # But we mixed them. Let's restructure.
    
    # Actually, let's just compute the penalty gradient separately and add it weighted.
    # For simplicity, let's just return the value and let scipy use numerical gradients.
    # The explicit gradient is complex and error-prone.
    
    return val

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
    Shrink radii gently to ensure strict validity.
    Uses proportional shrinkage based on how much each circle violates constraints.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        # 1. Boundary fix - shrink only the violating dimension proportionally
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                # Shrink to just fit, with small margin
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        
        # 2. Overlap fix - shrink the larger circle more
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Compute how much to shrink each circle
                    # Shrink proportionally to their radii, with margin
                    overlap = sum_r - dist
                    if overlap > 0:
                        # Shrink both circles to eliminate overlap with margin
                        margin = overlap * 0.5 + 1e-6
                        # Distribute shrinkage proportionally
                        total_r = radii[i] + radii[j]
                        if total_r > 1e-8:
                            shrink_i = margin * (radii[i] / total_r)
                            shrink_j = margin * (radii[j] / total_r)
                            radii[i] = max(radii[i] - shrink_i, 1e-6)
                            radii[j] = max(radii[j] - shrink_j, 1e-6)
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
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
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
            best_x, best_y, best_r = 0.5, 0.5, 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98
        
    return pos, radii

def _initial_lattice():
    """
    Generate an initial packing based on a lattice structure.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    points = []
    for i in range(5):
        for j in range(5):
            points.append([0.1 + i*0.2, 0.1 + j*0.2])
            
    for i in range(min(N, len(points))):
        pos[i] = points[i]
        radii[i] = 0.095
        
    if N > len(points):
        for i in range(len(points), N):
            best_x, best_y, best_r = 0.5, 0.5, 0.0
            rng = np.random.RandomState(i * 777)
            for _ in range(500):
                x = rng.uniform(0.1, 0.9)
                y = rng.uniform(0.1, 0.9)
                max_r_wall = min(x, 1.0-x, y, 1.0-y)
                max_r_circ = max_r_wall
                for j in range(i):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    possible_r = dist - radii[j]
                    if possible_r < max_r_circ:
                        max_r_circ = possible_r
                if max_r_circ > best_r:
                    best_r = max_r_circ
                    best_x = x
                    best_y = y
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.98
            
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_anneal(pos, radii, steps=1000, start_temp=0.01, end_temp=0.001, time_limit=None):
    """
    Force-directed annealing to improve packing.
    Adaptive push magnitudes and weights toward smaller circles.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for step in range(steps):
        if time_limit and time.time() > time_limit:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces - stronger for larger violations
        for i in range(N):
            if pos[i, 0] < radii[i]:
                viol = radii[i] - pos[i, 0]
                forces[i, 0] += viol * 20
            if pos[i, 0] > 1.0 - radii[i]:
                viol = pos[i, 0] - (1.0 - radii[i])
                forces[i, 0] -= viol * 20
            if pos[i, 1] < radii[i]:
                viol = radii[i] - pos[i, 1]
                forces[i, 1] += viol * 20
            if pos[i, 1] > 1.0 - radii[i]:
                viol = pos[i, 1] - (1.0 - radii[i])
                forces[i, 1] -= viol * 20
        
        # Overlap forces - adaptive magnitude
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-12
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    overlap = sum_r - dist
                    # Adaptive force: stronger for larger overlaps
                    force_mag = overlap * 10
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    # Weight toward smaller circle (it should move more)
                    w_i = radii[j] / (radii[i] + radii[j] + 1e-8)
                    w_j = radii[i] / (radii[i] + radii[j] + 1e-8)
                    forces[i, 0] += fx * w_i
                    forces[i, 1] += fy * w_i
                    forces[j, 0] -= fx * w_j
                    forces[j, 1] -= fy * w_j
        
        # Update positions with noise
        noise = np.random.normal(0, temp, pos.shape)
        pos += forces * 0.08 + noise
        
        # Clip to bounds
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        
        # Post-process to ensure validity
        pos, radii = _post_process(pos, radii)
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 148.0 # 2s safety margin
    
    # Strategy: Multi-start with different initializations
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Perturbed Best Fit
    for i in range(4):
        pos_p, rad_p = _initial_random(100 + i)
        inits.append((pos_p, rad_p))
        
    # 3. One more random
    pos_r, rad_r = _initial_random(999)
    inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))
        
    penalty_stages = [100.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: Coarse optimization
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(penalty_stages[0],),
                              options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-6})
            x0 = result.x
        except Exception:
            pass
        
        # Stage 2: Medium penalty
        if time.time() < deadline - 30:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1],),
                                  options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3: High penalty
        if time.time() < deadline - 40:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3.5: Very high penalty for tighter packing
        if time.time() < deadline - 35:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(5000.0,),
                                  options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12})
                x0 = result.x
            except Exception:
                pass
            
            x_opt = x0
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Force-directed anneal on the best result
        if idx == 0 and time.time() < deadline - 50:
            pos_fa, rad_fa = _force_directed_anneal(best_pos, best_radii, steps=800, time_limit=deadline - 50)
            score = np.sum(rad_fa)
            if _is_valid(pos_fa, rad_fa) and score > best_score:
                best_score = score
                best_pos = pos_fa.copy()
                best_radii = rad_fa.copy()

    # Final high-precision polish - Pass 1
    if time.time() < deadline - 15:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(10000.0,),
                              options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-14})
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
        
        # Polish - Pass 2: Powell
        if time.time() < deadline - 8:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x0, method='Powell', bounds=bounds,
                                  args=(10000.0,),
                                  options={'maxiter': 3000, 'ftol': 1e-16})
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
        
        # Polish - Pass 3: Final L-BFGS-B with even higher penalty
        if time.time() < deadline - 4:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(50000.0,),
                                  options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
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
        
        # Polish - Pass 4: Ultra-high penalty for final tightening
        if time.time() < deadline - 2:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(200000.0,),
                                  options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
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

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
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
