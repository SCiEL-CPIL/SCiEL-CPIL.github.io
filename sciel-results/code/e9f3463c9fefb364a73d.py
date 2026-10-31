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
    
    # Use a smoother penalty: (violation)^2 / (min_dist^2 + epsilon)
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
    # Let v = min_dist_sq - dist_sq
    # f = v^2 / denom
    # df/dv = 2v/denom - v^2/denom^2 = v(2denom - v)/denom^2
    # df/d(dist_sq) = -df/dv
    # df/d(x_i) = -df/d(dist_sq) * d(dist_sq)/d(x_i) = df/dv * 2*dx_ij
    
    # This is complex. Let's use a simpler penalty gradient or just use L-BFGS-B without explicit grad
    # Actually, for robustness, let's stick to the function-only version but improve the optimization strategy.
    # The previous version used function-only. Let's try to improve the search strategy instead.
    
    # For now, return just the value, let L-BFGS-B use numerical gradients if needed (slower but robust)
    # Or implement a simpler gradient.
    
    # Let's implement the gradient for the overlap part properly.
    # f_ij = (v_ij)^2 / D_ij where v_ij = max(0, M_ij^2 - D_ij^2), D_ij^2 = dist_sq_ij, M_ij = r_i + r_j
    # If v_ij > 0:
    # df/d(dist_sq_ij) = -2 v_ij / D_ij + v_ij^2 / D_ij^2  (derivative of v^2/D wrt D^2 is -2v/D + v^2/D^2? No)
    # Let u = dist_sq. v = M^2 - u. f = v^2 / (M^2 + eps).
    # df/du = d/du [(M^2-u)^2 / (M^2+eps)] = -2(M^2-u) / (M^2+eps)
    # d(u)/d(x_i) = 2(x_i - x_j)
    # So df/d(x_i) = -2(M^2-u)/(M^2+eps) * 2(x_i-x_j)
    
    # Let's compute this.
    v_pos = np.maximum(0.0, min_dist_sq - dist_sq)
    # Only where v_pos > 0
    active = v_pos > 0
    active = active & mask
    
    # Coefficient for dist_sq gradient: c_ij = -2 * v_pos_ij / (min_dist_sq_ij + 1e-8)
    # Note: v_pos is already max(0, ...), so for inactive pairs it's 0.
    c = np.zeros((N, N))
    c[active] = -2.0 * v_pos[active] / (min_dist_sq[active] + 1e-8)
    
    # c is symmetric? No, v_pos is symmetric, min_dist_sq is symmetric. So c is symmetric.
    # But we only want upper triangle in the sum.
    # The objective sums over i<j.
    # grad[x_i] += sum_{j>i} c_ij * 2 * (x_i - x_j) + sum_{j<i} c_ji * 2 * (x_i - x_j)
    # Since c is symmetric, c_ji = c_ij.
    # So grad[x_i] += 2 * sum_{j != i} c_ij * (x_i - x_j)
    
    # Let's vectorize this.
    # grad_x_i = 2 * sum_j c_ij (x_i - x_j) = 2 * (sum_j c_ij x_i - sum_j c_ij x_j)
    # sum_j c_ij x_i = x_i * sum_j c_ij
    # sum_j c_ij x_j is matrix-vector product C @ x
    
    C = c  # N x N
    sum_c_row = np.sum(C, axis=1)  # N
    grad_x_vec = 2.0 * (x[:N, np.newaxis] * sum_c_row[:, np.newaxis] - C.dot(x[:N]))
    grad_y_vec = 2.0 * (x[N:2*N, np.newaxis] * sum_c_row[:, np.newaxis] - C.dot(x[N:2*N]))
    
    grad[:N] += grad_x_vec
    grad[N:2*N] += grad_y_vec
    
    # Gradient w.r.t. radii
    # df/dr_i:
    # v_ij = (r_i+r_j)^2 - dist_sq
    # df/dv = 2v/D - v^2/D^2 = v(2D-v)/D^2 where D = min_dist_sq + 1e-8
    # dv/dr_i = 2(r_i+r_j)
    # So df/dr_i = v_ij(2D_ij - v_ij)/D_ij^2 * 2(r_i+r_j)
    
    # Sum over j != i
    # Let's compute a matrix H where H_ij = contribution of pair (i,j) to grad r_i
    # H_ij = v_ij * (2*min_dist_sq_ij + 1e-8 - v_ij) / (min_dist_sq_ij + 1e-8)^2 * 2 * min_dist_ij
    # This is only for active pairs.
    
    D_safe = min_dist_sq + 1e-8
    H = np.zeros((N, N))
    active_idx = np.where(active)
    if len(active_idx[0]) > 0:
        v_a = v_pos[active_idx]
        M_a = min_dist[active_idx]
        D_a = D_safe[active_idx]
        # df/dv = (2*D - v) / D * (v/D) ? No.
        # f = v^2 / D
        # df/dv = 2v/D - v^2/D^2
        # dv/dr_i = 2 M
        # df/dr_i = (2v/D - v^2/D^2) * 2M
        
        val = (2.0 * v_a / D_a - v_a**2 / D_a**2) * 2.0 * M_a
        # Scatter back
        H[active_idx] = val
    
    # grad[r_i] += sum_{j != i} H_ij
    grad_r = np.sum(H, axis=1)
    grad[2*N:] += grad_r
    
    # Apply penalty weight
    val = _objective(x, penalty_weight)
    grad *= penalty_weight
    
    # Wait, the boundary penalty gradient was computed without penalty weight in the intermediate steps?
    # No, I added the raw gradients. I need to multiply the whole gradient by penalty_weight.
    # But the -sum(radii) part is not penalized.
    # So: grad_total = -1 (for radii) + penalty_weight * (grad_penalty)
    
    # Let's restructure.
    grad_penalty = np.zeros_like(x)
    # ... fill grad_penalty with the terms above ...
    
    # Actually, the code above mixed them. Let's just return val and let scipy handle it, 
    # or implement this correctly in the final version.
    # Given time constraints, I will use the function-only version but optimize the search strategy.
    
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
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(100):
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
    """
    pos = pos.copy()
    radii = radii.copy()
    current_time = time.time()
    
    for step in range(steps):
        if time_limit and time.time() > time_limit:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 10
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i])) * 10
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 10
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i])) * 10
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-12
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    overlap = sum_r - dist
                    fx = overlap * dx / dist * 5
                    fy = overlap * dy / dist * 5
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update positions with noise
        noise = np.random.normal(0, temp, pos.shape)
        pos += forces * 0.1 + noise
        
        # Clip to bounds
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        
        # Post-process to ensure validity
        pos, radii = _post_process(pos, radii)
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
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
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_stages = [100.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
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
        if time.time() < deadline - 20:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1],),
                                  options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3: High penalty
        if time.time() < deadline - 25:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
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
        if idx == 0 and time.time() < deadline - 30:
            pos_fa, rad_fa = _force_directed_anneal(best_pos, best_radii, steps=500, time_limit=deadline - 30)
            score = np.sum(rad_fa)
            if _is_valid(pos_fa, rad_fa) and score > best_score:
                best_score = score
                best_pos = pos_fa.copy()
                best_radii = rad_fa.copy()

    # Final high-precision polish
    if time.time() < deadline - 10:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
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
        
        # Additional Powell polish
        if time.time() < deadline - 5:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x0, method='Powell', bounds=bounds,
                                  args=(5000.0,),
                                  options={'maxiter': 2000, 'ftol': 1e-16})
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
