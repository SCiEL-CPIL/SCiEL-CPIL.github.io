"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0, smoothness=1.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    Uses a smoother penalty formulation with soft-min for overlap.
    """
    # Reshape x into positions (N, 2) and radii (N,)
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
    
    # 2. Pairwise Overlap Violations (Vectorized, smoother)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    # Required distance: r_i + r_j
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Use soft penalty: max(0, min_dist - dist)^2 / (min_dist^2 + eps)
    # This is smoother than the squared-distance formulation
    dist = np.sqrt(dist_sq + 1e-12)
    violation = np.maximum(0.0, min_dist - dist)
    denom = min_dist * min_dist + 1e-10
    smooth_overlap = (violation ** 2) / denom
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
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

def _adaptive_penalty_weight(x, base_weight=500.0, min_weight=50.0, max_weight=2000.0):
    """
    Adaptively adjust penalty weight based on current violation level.
    Higher weight when violations are large, lower when near-feasible.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Compute total violation magnitude
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_viol = np.sum([left_viol, right_viol, bottom_viol, top_viol])
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap_viol = np.sum(np.maximum(0.0, min_dist - dist))
    
    total_viol = boundary_viol + overlap_viol
    
    # Scale weight: if violation is large, use higher weight; if small, use lower
    if total_viol < 1e-6:
        return min_weight
    elif total_viol < 0.01:
        return min_weight + (base_weight - min_weight) * (total_viol / 0.01)
    else:
        return max_weight

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    Also attempts to grow radii where there is clearance.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # First pass: fix violations
    for _ in range(30):
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
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
    
    # Second pass: try to grow radii where there is clearance
    for _ in range(10):
        improved = False
        for i in range(N):
            x, y = pos[i]
            # Max radius from boundaries
            max_r_boundary = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius from other circles
            max_r_circles = max_r_boundary
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # r_i + r_j <= dist => r_i <= dist - r_j
                max_r_circles = min(max_r_circles, dist - radii[j])
            
            max_r = min(max_r_boundary, max_r_circles)
            if max_r > radii[i] * 1.001:
                radii[i] = max_r * 0.9999
                improved = True
        
        if not improved:
            break
    
    return pos, radii

def _initial_model():
    """
    Construct a geometrically inspired initial layout.
    Strategy: 
    1. Place 4 circles in corners.
    2. Place remaining 22 in a hexagonal-like pattern in the center.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 1. Corners - slightly larger initial radius
    corner_r = 0.12
    corners = [
        (0.12, 0.12),
        (0.88, 0.12),
        (0.12, 0.88),
        (0.88, 0.88)
    ]
    for i in range(4):
        pos[i] = corners[i]
        radii[i] = corner_r
        
    # 2. Hexagonal pattern for the remaining 22 circles
    # Use a hexagonal lattice centered in the square
    # Rows at y = 0.3, 0.45, 0.55, 0.7 (4 rows)
    # Odd rows: 6 circles, even rows: 5 circles (offset by half spacing)
    # Total: 6 + 5 + 6 + 5 = 22
    
    r_center = 0.09
    y_rows = [0.3, 0.45, 0.55, 0.7]
    idx = 4
    
    for row_idx, y in enumerate(y_rows):
        if row_idx % 2 == 0:
            # 6 circles in this row
            x_start = 0.2
            x_spacing = 0.12
            for k in range(6):
                if idx < N:
                    pos[idx] = [x_start + k * x_spacing, y]
                    radii[idx] = r_center
                    idx += 1
        else:
            # 5 circles in this row, offset
            x_start = 0.26
            x_spacing = 0.12
            for k in range(5):
                if idx < N:
                    pos[idx] = [x_start + k * x_spacing, y]
                    radii[idx] = r_center
                    idx += 1
    
    # Small jitter to break symmetry and help escape local minima
    rng = np.random.RandomState(42)
    pos += rng.uniform(-0.005, 0.005, pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    pos, radii = _initial_model()
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds:
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Step 1: L-BFGS-B with adaptive penalty
    try:
        # Use callback to adapt penalty weight
        penalty_state = {'weight': 500.0}
        
        def adaptive_callback(xk):
            penalty_state['weight'] = _adaptive_penalty_weight(xk)
            return False
        
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          args=(penalty_state['weight'],),
                          callback=adaptive_callback,
                          options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.fun < _objective(x0, penalty_state['weight']) - 1e-10:
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
        
    # Step 2: Powell for fine tuning
    if time.time() < deadline - 15:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              args=(500.0,),
                              options={'maxiter': 3000, 'ftol': 1e-12})
            
            if result.fun < _objective(x0, 500.0) - 1e-10:
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
            
    # Step 3: Final high-precision polish with lower penalty weight
    if time.time() < deadline - 8:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            # Lower penalty weight to allow more growth when near-feasible
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(100.0,),
                              options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < _objective(x0, 100.0) - 1e-10:
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
    
    # Step 4: Final micro-polish with very tight tolerance
    if time.time() < deadline - 3:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(50.0,),
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if result.fun < _objective(x0, 50.0) - 1e-12:
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
