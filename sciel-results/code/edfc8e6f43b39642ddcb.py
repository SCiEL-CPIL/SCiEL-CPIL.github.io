import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_violations_and_score(pos, radii, w, h):
    n = len(radii)
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    overlap_sq = np.zeros_like(d_sq_upper)
    overlap_mask = d_sq_upper < r_sum_upper**2
    if np.any(overlap_mask):
        d_overlap = np.sqrt(d_sq_upper[overlap_mask])
        overlap_val = r_sum_upper[overlap_mask] - d_overlap
        overlap_sq[overlap_mask] = overlap_val**2
        
    circle_viol = np.sum(overlap_sq)
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    valid = total_viol < 1e-8
    return total_viol, sum_rad, valid

def _objective(x, w, h, penalty_weight=1000.0, eps=0.001):
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    # Smooth wall violation using soft-max
    wall_terms = np.maximum(0, radii - d_left) + np.maximum(0, radii - d_right) + \
                 np.maximum(0, radii - d_bottom) + np.maximum(0, radii - d_top)
    # Use a smoother approximation: for each term, use softplus-like
    # Actually let's keep quadratic but with a smooth max for the distance terms
    # Simpler: just use the squared violation as before but with a different penalty structure
    
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    # Smooth overlap penalty using soft-min
    # For each pair, the violation is max(0, r_i + r_j - dist)
    # We approximate max(0, x) with eps * log(1 + exp(x/eps))
    # But this can be numerically unstable. Let's use a different approach.
    # Use: soft_max(0, x) = 0.5 * (x + sqrt(x^2 + eps^2))
    
    d_upper = np.sqrt(np.maximum(d_sq_upper, 1e-12))
    overlap_raw = r_sum_upper - d_upper
    
    # Smooth max(0, overlap_raw) using softplus-like: 0.5*(x + sqrt(x^2 + eps^2))
    smooth_overlap = 0.5 * (overlap_raw + np.sqrt(overlap_raw**2 + eps**2))
    
    circle_viol = np.sum(smooth_overlap**2)
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    
    return penalty_weight * total_viol - sum_rad

def _hexagonal_start(w, h, num_circles, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(42)
        
    r_est = np.sqrt(0.9069 * w * h / (21 * np.pi))
    a = 2 * r_est
    
    margin = a * 2
    xs = np.arange(-margin, w + margin, a * np.sqrt(3)/2)
    ys = np.arange(-margin, h + margin, a)
    
    points = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if i % 2 == 1:
                y_offset = y + a/2
            else:
                y_offset = y
            points.append([x, y_offset])
            
    points = np.array(points)
    
    valid_mask = (points[:, 0] >= 0) & (points[:, 0] <= w) & \
                 (points[:, 1] >= 0) & (points[:, 1] <= h)
    points = points[valid_mask]
    
    if len(points) > num_circles:
        cx, cy = w/2, h/2
        dist_to_center = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        sorted_idx = np.argsort(dist_to_center)
        points = points[sorted_idx[:num_circles]]
    elif len(points) < num_circles:
        missing = num_circles - len(points)
        extra_points = rng.rand(missing, 2) * np.array([w, h])
        points = np.vstack([points, extra_points])
        
    order = rng.permutation(num_circles)
    points = points[order]
    
    radii = np.full(num_circles, r_est * 0.95)
    
    return points, radii

def _random_start(w, h, num_circles, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(0)
    
    pos = rng.rand(num_circles, 2) * np.array([w, h])
    max_r = min(w, h) / 2.0
    radii = rng.rand(num_circles) * 0.1 * max_r
    
    return pos, radii

def _polish(pos, radii, w, h, start_time, deadline):
    """High-precision polish with progressive eps reduction."""
    current_pos = pos.copy()
    current_radii = radii.copy()
    
    # Progressive eps values for smooth-max
    eps_schedule = [0.001, 0.0001, 1e-5, 1e-6, 1e-7]
    penalty_schedule = [1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
    
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
    
    for eps, pen in zip(eps_schedule, penalty_schedule):
        if time.time() > deadline - 2:
            break
        
        x0 = np.concatenate([current_pos.flatten(), current_radii])
        
        try:
            result = minimize(_objective, x0, args=(w, h, pen, eps), 
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-15, 'maxls': 50})
            
            x_opt = result.x
            new_pos = x_opt[:N*2].reshape(N, 2)
            new_radii = x_opt[N*2:]
            
            viol, score, valid = _compute_violations_and_score(new_pos, new_radii, w, h)
            
            if valid or viol < 1e-6:
                # Accept the improvement
                current_pos = new_pos
                current_radii = new_radii
            else:
                # Try to scale down radii to make valid
                scale = 0.95
                while viol > 1e-8 and scale > 0.1:
                    rad_new = new_radii * scale
                    viol, s, valid = _compute_violations_and_score(new_pos, rad_new, w, h)
                    if valid:
                        current_pos = new_pos
                        current_radii = rad_new
                        break
                    scale *= 0.9
                if not valid:
                    break
        except Exception:
            break
    
    return current_pos, current_radii

def _pack_for_aspect(w, h, start_time, deadline):
    best_score = 0
    best_pos = None
    best_radii = None
    
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
        
    initializations = []
    
    # 1. Hexagonal lattice start
    if time.time() < deadline - 15:
        pos_hex, rad_hex = _hexagonal_start(w, h, N, seed=42)
        x_hex = np.concatenate([pos_hex.flatten(), rad_hex])
        initializations.append(('hex', x_hex))
        
        viol, score, valid = _compute_violations_and_score(pos_hex, rad_hex, w, h)
        if valid and score > best_score:
            best_score = score
            best_pos = pos_hex.copy()
            best_radii = rad_hex.copy()
            
    # 2. Random restarts
    if time.time() < deadline - 10:
        for seed in [123, 456, 789, 1011, 2023, 3456, 7890, 1234]:
            if time.time() > deadline - 5:
                break
            pos_rand, rad_rand = _random_start(w, h, N, seed=seed)
            x_rand = np.concatenate([pos_rand.flatten(), rad_rand])
            initializations.append(('rand', x_rand))
            
    for idx, (name, x0) in enumerate(initializations):
        if time.time() > deadline - 3:
            break
            
        try:
            # First pass: coarse optimization
            result = minimize(_objective, x0, args=(w, h, 1000.0, 0.001), 
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 800, 'ftol': 1e-15, 'maxls': 100})
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            
            if not valid:
                if viol > 1e-4:
                    scale = 0.95
                    while viol > 1e-8 and scale > 0.1:
                        rad_new = rad_opt * scale
                        viol, s, valid = _compute_violations_and_score(pos_opt, rad_new, w, h)
                        if valid:
                            rad_opt = rad_new
                            score = s
                            break
                        scale *= 0.9
                    if not valid:
                        continue
            else:
                score = np.sum(rad_opt)
                
            # Polish phase
            if time.time() < deadline - 2:
                pos_opt, rad_opt = _polish(pos_opt, rad_opt, w, h, start_time, deadline)
                viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
                if not valid:
                    scale = 0.95
                    while viol > 1e-8 and scale > 0.1:
                        rad_new = rad_opt * scale
                        viol, s, valid = _compute_violations_and_score(pos_opt, rad_new, w, h)
                        if valid:
                            rad_opt = rad_new
                            score = s
                            break
                        scale *= 0.9
                    if not valid:
                        continue
                else:
                    score = np.sum(rad_opt)
                    
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = rad_opt.copy()
                
        except Exception:
            continue
            
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_global_score = 0
    best_global_result = None
    
    np.random.seed(42)
    
    # Try a range of aspect ratios
    aspect_ratios = np.linspace(0.5, 2.0, 25)
    
    # Ensure square is included
    if not np.any(np.isclose(aspect_ratios, 1.0)):
        aspect_ratios = np.insert(aspect_ratios, len(aspect_ratios)//2, 1.0)
        
    for ar in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline)
        
        if pos is not None and score > best_global_score:
            best_global_score = score
            viol, s, valid = _compute_violations_and_score(pos, radii, w, h)
            if valid:
                best_global_result = np.column_stack([pos, radii])
            elif viol < 1e-3:
                scale = 0.99
                while viol > 1e-8 and scale > 0.1:
                    radii_new = radii * scale
                    viol, s, valid = _compute_violations_and_score(pos, radii_new, w, h)
                    if valid:
                        radii = radii_new
                        break
                    scale *= 0.95
                if valid:
                    best_global_result = np.column_stack([pos, radii])
    
    # Fallback
    if best_global_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val * 0.9
        best_global_result = np.column_stack([pos, rad])
        
    return best_global_result
