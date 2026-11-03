import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _check_validity(pos, radii, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(radii) != N:
        return False
    for i in range(N):
        x, y, r = pos[i, 0], pos[i, 1], radii[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _compute_violations_and_score(pos, radii, w, h):
    """
    Compute constraint violations and the sum of radii.
    Returns (violation_sum, sum_radii, valid_flag)
    """
    n = len(radii)
    # Vectorized wall violations
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    # Circle-Circle violations
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
    valid = total_viol < 1e-6
    return total_viol, sum_rad, valid

def _separate_circles(pos, radii, W, H, max_iter=2000, tol=1e-10):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (pos, radii, is_valid).
    """
    n = len(radii)
    x = pos[:, 0].copy()
    y = pos[:, 1].copy()
    r = radii.copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by shrinking radii if needed
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            # Clamp position to keep circle inside rectangle
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - tol:
                    # Push apart
                    if dist < 1e-12:
                        # Random nudge
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    # Distribute push based on relative sizes
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation, shrinking radius if circle doesn't fit
        for i in range(n):
            # If circle can't fit at current position, shrink it
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result_pos = np.column_stack([x, y])
    valid = _check_validity(result_pos, r, W, H)
    return result_pos, r, valid

def _smooth_max_grad(v, eps=1e-4):
    """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
    smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
    grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
    return smax, grad_smax

def _objective_and_grad(params, W, H, boundary_weight=200.0, overlap_weight=200.0):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Returns (obj, grad) with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum(r))/dr

    eps = 1e-4

    # Boundary penalties (smooth)
    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W  => violation = x + r - W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties: for each pair, violation = r_i + r_j - dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
                dx = 1e-12
                dy = 0.0
            v = r[i] + r[j] - dist
            smax, gsmax = _smooth_max_grad(v, eps)
            obj += overlap_weight * smax ** 2

            factor = 2.0 * overlap_weight * smax * gsmax
            d_obj_dx_i = factor * (-dx / dist)
            d_obj_dy_i = factor * (-dy / dist)
            d_obj_dx_j = factor * (dx / dist)
            d_obj_dy_j = factor * (dy / dist)
            d_obj_dr_i = factor * 1.0
            d_obj_dr_j = factor * 1.0

            grad[3 * i] += d_obj_dx_i
            grad[3 * i + 1] += d_obj_dy_i
            grad[3 * i + 2] += d_obj_dr_i
            grad[3 * j] += d_obj_dx_j
            grad[3 * j + 1] += d_obj_dy_j
            grad[3 * j + 2] += d_obj_dr_j

    return obj, grad

def _local_optimize(circles, W, H, max_iter=1000, xatol=1e-10, fatol=1e-12):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))  # x
        bounds.append((1e-6, H - 1e-6))  # y
        bounds.append((1e-6, r_max))     # r

    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': fatol, 'gtol': 1e-10, 'maxls': 100}
    )
    return result

def _greedy_place(w, h, num_circles, grid_res=50):
    """
    Place circles sequentially to maximize the minimum radius at each step.
    Returns pos (N, 2) and radii (N,)
    """
    pos = np.zeros((num_circles, 2))
    radii = np.zeros(num_circles)
    
    # Grid of potential centers
    xs = np.linspace(0.05*w, 0.95*w, grid_res)
    ys = np.linspace(0.05*h, 0.95*h, grid_res)
    
    for k in range(num_circles):
        best_r = 0
        best_x = w / 2.0
        best_y = h / 2.0
        
        # Vectorized grid search for best position/radius for circle k
        XX, YY = np.meshgrid(xs, ys)
        
        # Wall radius
        r_wall = np.minimum(np.minimum(XX, w - XX), np.minimum(YY, h - YY))
        
        # Circle radius constraint
        r_circle = np.full_like(r_wall, np.inf)
        for i in range(k):
            dx = XX - pos[i, 0]
            dy = YY - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r_max_i = dist - radii[i]
            r_circle = np.minimum(r_circle, r_max_i)
            
        # Max allowed radius at each grid point
        r_allowed = np.minimum(r_wall, r_circle)
        
        # Find max
        max_idx = np.unravel_index(np.argmax(r_allowed), r_allowed.shape)
        best_r = r_allowed[max_idx]
        best_y = ys[max_idx[0]]
        best_x = xs[max_idx[1]]
        
        pos[k, 0] = best_x
        pos[k, 1] = best_y
        radii[k] = best_r
        
    return pos, radii

def _pack_for_aspect(w, h, start_time, deadline):
    """
    Pack circles in w x h rectangle.
    Returns best (pos, radii) and score found for this aspect.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    max_r = min(w, h) / 2.0
    
    # 1. Greedy Initialization
    pos_grid, rad_grid = _greedy_place(w, h, N, grid_res=60)
    x0 = np.concatenate([pos_grid.flatten(), rad_grid])
    
    # Check initial validity and score
    viol, score, valid = _compute_violations_and_score(pos_grid, rad_grid, w, h)
    if valid and score > best_score:
        best_score = score
        best_pos = pos_grid.copy()
        best_radii = rad_grid.copy()
        
    # 2. Optimization
    # We will run L-BFGS-B from the greedy start.
    # Add a few random restarts if time permits.
    
    initializations = [x0]
    
    # Add random restarts
    if time.time() < deadline - 5:
        np.random.seed(42) 
        for _ in range(2):
            pos_rand = np.random.rand(N, 2) * np.array([w, h])
            rad_rand = np.random.rand(N) * 0.1 * min(w, h)
            x_rand = np.concatenate([pos_rand.flatten(), rad_rand])
            initializations.append(x_rand)

    for idx, x0 in enumerate(initializations):
        if time.time() > deadline - 2:
            break
            
        try:
            result = _local_optimize(x0.reshape(N, 3), w, h)
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            
            # Verify and clean up
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            
            if not valid:
                # Try to fix by separating and shrinking
                pos_fixed, rad_fixed, valid_fixed = _separate_circles(pos_opt, rad_opt, w, h)
                if valid_fixed:
                    pos_opt = pos_fixed
                    rad_opt = rad_fixed
                    viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
                    
                if not valid:
                    # Try to fix by scaling down radii
                    if viol > 1e-4:
                        scale = 0.95
                        while viol > 1e-6 and scale > 0.1:
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
    # 150s limit. Use 140s for safety.
    deadline = start_time + 140
    
    best_global_score = 0
    best_global_result = None
    
    # Aspect ratios to try
    # Perimeter 4 => 2(w+h)=4 => w+h=2 => h = 2-w
    # w in [0.5, 1.5]
    # Square is w=1, h=1.
    # We will try a range of aspect ratios.
    # The optimal aspect ratio for packing is often slightly elongated.
    
    np.random.seed(42)
    
    # Generate aspect ratios w/h
    # w = 2*ar/(1+ar), h = 2/(1+ar)
    # ar = w/h
    # Let's try ar from 0.7 to 1.4
    aspect_ratios = np.linspace(0.7, 1.4, 15)
    
    # Ensure we check the square (ar=1.0)
    if not np.any(np.isclose(aspect_ratios, 1.0)):
        aspect_ratios = np.insert(aspect_ratios, len(aspect_ratios)//2, 1.0)
        
    for ar in aspect_ratios:
        if time.time() > deadline - 5: # Leave 5s for cleanup
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline)
        
        if pos is not None and score > best_global_score:
            best_global_score = score
            # Ensure strict validity
            viol, s, valid = _compute_violations_and_score(pos, radii, w, h)
            if valid:
                best_global_result = np.column_stack([pos, radii])
            elif viol < 1e-3:
                # Fix by separating and shrinking
                pos_fixed, radii_fixed, valid_fixed = _separate_circles(pos, radii, w, h)
                if valid_fixed:
                    pos = pos_fixed
                    radii = radii_fixed
                    viol, s, valid = _compute_violations_and_score(pos, radii, w, h)
                    
                if not valid:
                    # Fix by shrinking
                    scale = 0.99
                    while viol > 1e-6 and scale > 0.1:
                        radii_new = radii * scale
                        viol, s, valid = _compute_violations_and_score(pos, radii_new, w, h)
                        if valid:
                            radii = radii_new
                            break
                        scale *= 0.95
                if valid:
                    best_global_result = np.column_stack([pos, radii])
    
    # Fallback if no valid result found
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
