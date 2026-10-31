"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(v, beta=50.0):
    """Smooth approximation of max(0, v) using log(1+exp(beta*v))/beta."""
    # For numerical stability, use piecewise definition
    result = np.zeros_like(v, dtype=float)
    positive_mask = v > 0
    negative_mask = ~positive_mask
    # For large positive v, softplus ≈ v
    large_pos = v > 20.0 / beta
    small_pos = positive_mask & ~large_pos
    result[large_pos] = v[large_pos]
    result[small_pos] = np.log1p(np.exp(beta * v[small_pos])) / beta
    # For negative v, softplus ≈ exp(beta*v)/beta
    result[negative_mask] = np.exp(beta * v[negative_mask]) / beta
    return result

def _penalty_function(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    Uses softplus for smooth max(0, v).
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # Boundary violations: circle must be inside [0,1]^2
    bv_left = radii - pos[:, 0]
    bv_right = pos[:, 0] + radii - 1.0
    bv_bottom = radii - pos[:, 1]
    bv_top = pos[:, 1] + radii - 1.0
    
    # Use softplus for smooth penalty
    boundary_pen = np.sum(_softplus(bv_left, beta)**2 + _softplus(bv_right, beta)**2 + 
                          _softplus(bv_bottom, beta)**2 + _softplus(bv_top, beta)**2)

    # Overlap violations: dist(i, j) >= r_i + r_j
    # violation: (r_i + r_j)^2 - dist^2
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    # Only consider upper triangle (i < j)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = min_dist_sq[mask] - dist_sq[mask]
    
    overlap_pen = np.sum(_softplus(viol, beta)**2)

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
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
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """
    Ensure strict validity by minimally adjusting radii/positions.
    Uses a gentler approach: only shrink when necessary, and by the minimum amount.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(300):
        changed = False
        for i in range(N):
            # Check boundaries
            max_r_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r_bound:
                radii[i] = max_r_bound * (1.0 - 1e-10)
                changed = True
            
            # Check overlaps with others
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # If overlapping, compute exact shrink needed
                if dist < radii[i] + radii[j]:
                    # We need r_i + r_j <= dist
                    # Shrink proportionally to current radii
                    total_r = radii[i] + radii[j]
                    if total_r > 0:
                        scale = dist / total_r * (1.0 - 1e-10)
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True

        if not changed:
            break

    # Final strict check
    if not _is_valid(pos, radii, tol=1e-9):
        # Very gentle final shrink
        for i in range(N):
            radii[i] *= (1.0 - 1e-8)
        
    return pos, radii

def _best_fit_init(seed=0):
    """
    Greedy Best-Fit initialization.
    Place circles one by one, choosing the location that allows the largest possible radius.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place first circle in center with max possible radius (0.25)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        # Try many random positions
        for _ in range(20000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # Circle i can have radius at most dist - r_j
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if found and best_r > 0.01:
            pos[i] = [best_x, best_y]
            # Apply a small shrink factor to ensure no floating point overlap issues later
            radii[i] = best_r * 0.98 
        else:
            # Fallback: place a tiny circle somewhere
            for _ in range(1000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                min_dist = min(x, 1-x, y, 1-y)
                for j in range(i):
                    d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                    min_dist = min(min_dist, d - radii[j])
                if min_dist > 0.005:
                    pos[i] = [x, y]
                    radii[i] = min_dist * 0.95
                    break
            else:
                # Last resort
                pos[i] = [0.5, 0.5]
                radii[i] = 0.001

    return pos, radii

def _hex_init():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Approximate radius for hex packing in unit square
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break

    # Fill remaining with random best-fit
    rng = np.random.RandomState(123)
    while k < N:
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(5000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(k):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r:
                best_r = max_r_circ
                best_x = x
                best_y = y
        if best_r > 0.01:
            pos[k] = [best_x, best_y]
            radii[k] = best_r * 0.98
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        k += 1
        
    return pos, radii

def _corner_biased_init(seed=42):
    """Corner-biased initialization: place larger circles near corners first."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place 4 circles in corners with moderate radius
    corner_positions = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for i in range(4):
        pos[i] = corner_positions[i]
        radii[i] = 0.14
    
    # Place remaining circles using best-fit
    for i in range(4, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        # Try many random positions
        for _ in range(15000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if found and best_r > 0.005:
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.97
        else:
            # Fallback: place a tiny circle somewhere
            for _ in range(1000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                min_dist = min(x, 1-x, y, 1-y)
                for j in range(i):
                    d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                    min_dist = min(min_dist, d - radii[j])
                if min_dist > 0.003:
                    pos[i] = [x, y]
                    radii[i] = min_dist * 0.95
                    break
            else:
                pos[i] = [0.5, 0.5]
                radii[i] = 0.001

    return pos, radii

def _optimize(pos, radii, deadline, seed=0):
    """
    Optimize the configuration using L-BFGS-B with penalty, then SLSQP polish.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds for positions and radii
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
        
    best_x = x0.copy()
    best_score = np.sum(radii)
    
    # Stage 1: Moderate penalty, find a valid-ish config
    p1 = 100.0
    try:
        res1 = minimize(_penalty_function, x0, method='L-BFGS-B', bounds=bounds,
                        args=(p1,),
                        options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
        if res1.fun < _penalty_function(x0, p1):
            x_curr = res1.x
            pos_curr = x_curr[:2*N].reshape(N, 2)
            rad_curr = np.maximum(x_curr[2*N:], 1e-6)
            pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
            if _is_valid(pos_pp, rad_pp):
                score = np.sum(rad_pp)
                if score > best_score:
                    best_score = score
                    best_x = np.concatenate([pos_pp.flatten(), rad_pp])
    except Exception:
        pass
    
    # Stage 2: High penalty, refine
    if time.time() < deadline - 10:
        p2 = 1000.0
        try:
            res2 = minimize(_penalty_function, best_x, method='L-BFGS-B', bounds=bounds,
                            args=(p2,),
                            options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res2.fun < _penalty_function(best_x, p2):
                x_curr = res2.x
                pos_curr = x_curr[:2*N].reshape(N, 2)
                rad_curr = np.maximum(x_curr[2*N:], 1e-6)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass
            
    # Stage 3: SLSQP with hard constraints for final polish - tighter tolerances
    if time.time() < deadline - 5:
        try:
            ii, jj = np.triu_indices(N, k=1)
            
            def constraint_boundary(x):
                pos = x[:2 * N].reshape(N, 2)
                radii = x[2 * N:]
                c1 = pos[:, 0] - radii
                c2 = 1.0 - pos[:, 0] - radii
                c3 = pos[:, 1] - radii
                c4 = 1.0 - pos[:, 1] - radii
                return np.concatenate([c1, c2, c3, c4])

            def constraint_overlap(x):
                pos = x[:2 * N].reshape(N, 2)
                radii = x[2 * N:]
                dx = pos[ii, 0] - pos[jj, 0]
                dy = pos[ii, 1] - pos[jj, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[ii] + radii[jj]
                return dist_sq - min_dist * min_dist

            def objective(x):
                return -np.sum(x[2*N:])

            constraints = [
                {'type': 'ineq', 'fun': constraint_boundary},
                {'type': 'ineq', 'fun': constraint_overlap}
            ]
            
            res3 = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False})
            
            if res3.success or (res3.fun < objective(best_x)):
                pos_curr = res3.x[:2*N].reshape(N, 2)
                rad_curr = np.maximum(res3.x[2*N:], 1e-6)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    # Stage 4: Second SLSQP polish with even tighter tolerances if time permits
    if time.time() < deadline - 2:
        try:
            ii, jj = np.triu_indices(N, k=1)
            
            def constraint_boundary(x):
                pos = x[:2 * N].reshape(N, 2)
                radii = x[2 * N:]
                c1 = pos[:, 0] - radii
                c2 = 1.0 - pos[:, 0] - radii
                c3 = pos[:, 1] - radii
                c4 = 1.0 - pos[:, 1] - radii
                return np.concatenate([c1, c2, c3, c4])

            def constraint_overlap(x):
                pos = x[:2 * N].reshape(N, 2)
                radii = x[2 * N:]
                dx = pos[ii, 0] - pos[jj, 0]
                dy = pos[ii, 1] - pos[jj, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[ii] + radii[jj]
                return dist_sq - min_dist * min_dist

            def objective(x):
                return -np.sum(x[2*N:])

            constraints = [
                {'type': 'ineq', 'fun': constraint_boundary},
                {'type': 'ineq', 'fun': constraint_overlap}
            ]
            
            res4 = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-16, 'disp': False})
            
            if res4.success or (res4.fun < objective(best_x)):
                pos_curr = res4.x[:2*N].reshape(N, 2)
                rad_curr = np.maximum(res4.x[2*N:], 1e-6)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    pos_out = best_x[:2*N].reshape(N, 2)
    rad_out = best_x[2*N:]
    return pos_out, rad_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_pos = None
    best_rad = None
    best_score = 0.0
    
    # Generate initializations
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _best_fit_init(seed=1)
    inits.append((pos_bf, rad_bf))
    
    # 2. Best Fit with different seed
    pos_bf2, rad_bf2 = _best_fit_init(seed=2)
    inits.append((pos_bf2, rad_bf2))
    
    # 3. Hexagonal
    pos_hex, rad_hex = _hex_init()
    inits.append((pos_hex, rad_hex))
    
    # 4. Corner-biased
    pos_cb, rad_cb = _corner_biased_init(seed=42)
    inits.append((pos_cb, rad_cb))
    
    # Optimize each initialization
    for i, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        # Initial score
        pos_pp, rad_pp = _post_process(pos_init, rad_init)
        if _is_valid(pos_pp, rad_pp):
            score = np.sum(rad_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_rad = rad_pp.copy()
                
        # Run optimizer
        try:
            pos_opt, rad_opt = _optimize(pos_pp, rad_pp, deadline, seed=i)
            pos_pp_opt, rad_pp_opt = _post_process(pos_opt, rad_opt)
            if _is_valid(pos_pp_opt, rad_pp_opt):
                score = np.sum(rad_pp_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp_opt.copy()
                    best_rad = rad_pp_opt.copy()
        except Exception:
            pass

    # If no valid solution found, use a safe fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        # Place 25 circles in 5x5 grid
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    rad[k] = 0.095
                    k += 1
        best_pos, best_rad = _post_process(pos, rad)
        best_score = np.sum(best_rad)

    # Final validation
    best_pos, best_rad = _post_process(best_pos, best_rad)
    
    result = np.column_stack([best_pos, best_rad])
    return result

# EVOLVE-BLOCK-END
