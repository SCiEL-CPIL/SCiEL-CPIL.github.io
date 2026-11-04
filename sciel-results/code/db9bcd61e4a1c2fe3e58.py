"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n):
    """
    Compute the maximum valid radius for each circle given positions.
    Uses iterative relaxation to find consistent radii.
    """
    radii = np.zeros(n)
    
    # Initialize with boundary constraints
    for i in range(n):
        x, y = pos[i]
        radii[i] = min(x, y, 1.0 - x, 1.0 - y)
    
    # Iterative relaxation: update each radius to be consistent with all others
    for _ in range(50):
        changed = False
        for i in range(n):
            x, y = pos[i]
            r_bound = min(x, y, 1.0 - x, 1.0 - y)
            r_max = r_bound
            
            for j in range(n):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_candidate = dist - radii[j]
                if r_candidate < r_max:
                    r_max = r_candidate
            
            r_new = max(0.0, r_max)
            if abs(r_new - radii[i]) > 1e-10:
                radii[i] = r_new
                changed = True
        
        if not changed:
            break
    
    return radii

def _soft_objective_and_grad(params, n, beta):
    """
    Soft objective: maximize sum of radii - penalty for overlaps and boundary violations.
    params: (n, 3) array of (x, y, r)
    Returns (objective, gradient)
    """
    pos = params[:, :2]
    radii = params[:, 2]
    
    # Compute pairwise distances
    dx = pos[:, 0:1] - pos[0:1, 0]
    dy = pos[:, 1:2] - pos[0:2, 1]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    
    # Overlap penalty: for each pair i,j, penalty if r_i + r_j > dist_ij
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            excess = radii[i] + radii[j] - dist[i, j]
            if excess > 0:
                overlap[i, j] = excess
                overlap[j, i] = excess
    
    # Boundary penalty
    bound_viol = np.zeros(n)
    for i in range(n):
        bound_viol[i] = max(0, radii[i] - pos[i, 0]) + \
                        max(0, radii[i] - pos[i, 1]) + \
                        max(0, radii[i] - (1 - pos[i, 0])) + \
                        max(0, radii[i] - (1 - pos[i, 1]))
    
    total_overlap = np.sum(overlap)
    total_bound = np.sum(bound_viol)
    
    obj = -np.sum(radii) + beta * (total_overlap**2 + total_bound**2)
    
    # Gradient computation
    grad = np.zeros_like(params)
    
    # Gradient w.r.t. radii: -1 + 2*beta * (sum of overlaps involving i)
    for i in range(n):
        grad[i, 2] = -1.0
        for j in range(n):
            if i != j:
                excess = radii[i] + radii[j] - dist[i, j]
                if excess > 0:
                    grad[i, 2] += 2 * beta * excess
        # Boundary gradient
        if radii[i] > pos[i, 0]:
            grad[i, 2] += 2 * beta * (radii[i] - pos[i, 0])
        if radii[i] > pos[i, 1]:
            grad[i, 2] += 2 * beta * (radii[i] - pos[i, 1])
        if radii[i] > (1 - pos[i, 0]):
            grad[i, 2] += 2 * beta * (radii[i] - (1 - pos[i, 0]))
        if radii[i] > (1 - pos[i, 1]):
            grad[i, 2] += 2 * beta * (radii[i] - (1 - pos[i, 1]))
    
    # Gradient w.r.t. positions
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            excess = radii[i] + radii[j] - dist[i, j]
            if excess > 0:
                # d(dist)/d(x_i) = (x_i - x_j) / dist
                grad[i, 0] -= 2 * beta * excess * (pos[i, 0] - pos[j, 0]) / dist[i, j]
                grad[i, 1] -= 2 * beta * excess * (pos[i, 1] - pos[j, 1]) / dist[i, j]
        
        # Boundary position gradients
        if radii[i] > pos[i, 0]:
            grad[i, 0] += 2 * beta * (radii[i] - pos[i, 0])
        if radii[i] > pos[i, 1]:
            grad[i, 1] += 2 * beta * (radii[i] - pos[i, 1])
        if radii[i] > (1 - pos[i, 0]):
            grad[i, 0] -= 2 * beta * (radii[i] - (1 - pos[i, 0]))
        if radii[i] > (1 - pos[i, 1]):
            grad[i, 1] -= 2 * beta * (radii[i] - (1 - pos[i, 1]))
    
    return obj, grad

def _discrete_best_fit(pos, radii, n, deadline, step_sizes=None):
    """
    Discrete best-fit local search:
    For each circle, try moving it to various candidate positions
    and keep the one that maximizes the sum of radii.
    """
    if step_sizes is None:
        step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    for step_size in step_sizes:
        if time.time() > deadline:
            break
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(n):
                orig_x, orig_y = best_pos[i]
                best_local_sum = best_sum
                best_local_pos = best_pos.copy()
                best_local_radii = best_radii.copy()
                
                # Generate candidate positions
                candidates = []
                # 8 neighbors + center
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        # Keep in bounds
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            candidates.append((new_x, new_y))
                
                # Also try diagonal moves
                diag = step_size * np.sqrt(2) / 2
                for dx in [-diag, diag]:
                    for dy in [-diag, diag]:
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            candidates.append((new_x, new_y))
                
                # Try each candidate
                for cx, cy in candidates:
                    trial_pos = best_local_pos.copy()
                    trial_pos[i] = [cx, cy]
                    trial_radii = _compute_radii(trial_pos, n)
                    trial_sum = np.sum(trial_radii)
                    
                    if trial_sum > best_local_sum + 1e-8:
                        best_local_sum = trial_sum
                        best_local_pos = trial_pos
                        best_local_radii = trial_radii
                
                if best_local_sum > best_sum + 1e-8:
                    best_pos = best_local_pos
                    best_radii = best_local_radii
                    best_sum = best_local_sum
                    improved = True
    
    return best_pos, best_radii

def _gradient_optimize(pos, radii, n, deadline):
    """
    Use L-BFGS-B to optimize the soft objective (maximize sum of radii with overlap penalty).
    """
    if time.time() > deadline:
        return pos, radii
    
    params = np.zeros((n, 3))
    params[:, :2] = pos
    params[:, 2] = radii
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    best_params = params.copy()
    best_score = np.sum(radii)
    
    # Try different beta values
    for beta in [10.0, 50.0, 200.0]:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                _soft_objective_and_grad,
                best_params,
                args=(n, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < 0:
                opt_params = result.x
                opt_pos = opt_params[:, :2].copy()
                opt_radii = opt_params[:, 2].copy()
                
                # Clamp to valid values
                for i in range(n):
                    mr = min(opt_pos[i, 0], opt_pos[i, 1], 1 - opt_pos[i, 0], 1 - opt_pos[i, 1])
                    opt_radii[i] = min(max(0.0, opt_radii[i]), max(0.0, mr))
                
                # Recompute proper radii
                opt_radii = _compute_radii(opt_pos, n)
                score = np.sum(opt_radii)
                
                if score > best_score:
                    best_score = score
                    best_params = opt_params.copy()
                    best_params[:, :2] = opt_pos
                    best_params[:, 2] = opt_radii
                    
        except Exception:
            continue
    
    # Return the best found
    best_pos = best_params[:, :2].copy()
    best_radii = _compute_radii(best_pos, n)
    
    return best_pos, best_radii

def _validate(pos, radii, n, tol=1e-9):
    """Check if packing is valid."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_and_validate(pos, radii, n):
    """
    Ensure validity by clamping radii.
    This is a safety net.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp to boundary
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
        
    # Fix overlaps by reducing radii
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                md = radii[i] + radii[j]
                if d < md - 1e-9:
                    excess = md - d
                    reduction = excess / 2.0 + 1e-6
                    radii[i] = max(0.0, radii[i] - reduction)
                    radii[j] = max(0.0, radii[j] - reduction)
                    changed = True
        if not changed:
            break
            
    # Final boundary clamp
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
        
    if _validate(pos, radii, n):
        return pos, radii, np.sum(radii)
    else:
        return None, None, -1.0

def _initial_grid(n):
    """Create a grid initial configuration."""
    pos = np.zeros((n, 2))
    # 5x6 grid = 30, take first 26
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx >= n:
                break
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 6.0
            pos[idx] = [x, y]
            idx += 1
    return pos

def _initial_hexagonal(n):
    """Create a hexagonal initial configuration."""
    pos = np.zeros((n, 2))
    idx = 0
    row = 0
    while idx < n:
        offset = 0.1 if row % 2 == 1 else 0.0
        for col in range(5):
            if idx >= n:
                break
            x = (col + 0.5) * 0.2 + offset
            y = (row + 0.5) * 0.2
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pos[idx] = [x, y]
            idx += 1
        row += 1
    return pos

def _initial_random(n, seed=0):
    """Create a random initial configuration with minimum separation."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 2))
    valid_pos = []
    attempts = 0
    while len(valid_pos) < n and attempts < 1000:
        p = rng.uniform(0.1, 0.9, size=2)
        ok = True
        for vp in valid_pos:
            if np.linalg.norm(p - vp) < 0.15:
                ok = False
                break
        if ok:
            valid_pos.append(p)
        attempts += 1
    while len(valid_pos) < n:
        valid_pos.append(rng.uniform(0.1, 0.9, size=2))
    pos[:] = valid_pos
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Multiple initial configurations
    configs = []
    
    # Grid
    pos_grid = _initial_grid(N)
    radii_grid = _compute_radii(pos_grid, N)
    configs.append((pos_grid, radii_grid, "grid"))
    
    # Hexagonal
    pos_hex = _initial_hexagonal(N)
    radii_hex = _compute_radii(pos_hex, N)
    configs.append((pos_hex, radii_hex, "hex"))
    
    # Multiple random starts
    for seed in range(8):
        pos_rand = _initial_random(N, seed)
        radii_rand = _compute_radii(pos_rand, N)
        configs.append((pos_rand, radii_rand, f"rand{seed}"))
    
    # Process each configuration
    for pos, radii, name in configs:
        if time.time() > deadline - 20:
            break
        
        try:
            # Phase 1: Gradient-based optimization
            if time.time() < deadline - 15:
                grad_pos, grad_radii = _gradient_optimize(pos, radii, N, deadline - 12)
            else:
                grad_pos, grad_radii = pos, radii
            
            # Phase 2: Discrete best-fit refinement
            opt_pos, opt_radii = _discrete_best_fit(grad_pos, grad_radii, N, deadline - 8)
            
            # Validate and fix
            valid_pos, valid_radii, score = _fix_and_validate(opt_pos, opt_radii, N)
            
            if valid_pos is not None and score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = valid_pos
                result[:, 2] = valid_radii
                best_result = result
        except Exception:
            continue
    
    # Final polish: run gradient + discrete on the best result
    if best_result is not None and time.time() < deadline - 10:
        try:
            polish_pos = best_result[:, :2].copy()
            polish_radii = best_result[:, 2].copy()
            
            # Gradient polish
            gp, gr = _gradient_optimize(polish_pos, polish_radii, N, deadline - 5)
            
            # Discrete polish with finer steps
            step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002]
            fp, fr = _discrete_best_fit(gp, gr, N, deadline - 2, step_sizes)
            
            valid_pos, valid_radii, score = _fix_and_validate(fp, fr, N)
            if valid_pos is not None and score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = valid_pos
                result[:, 2] = valid_radii
                best_result = result
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: small circles in a grid
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
        
    return best_result

# EVOLVE-BLOCK-END
