"""Circle packing in perimeter-4 rectangle: L-BFGS-B with log-sum-exp smooth penalties."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _force_relax(x, y, r, w, h, max_iter=500, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries."""
    n = N
    dt = dt_init
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-9:
            break

        f_pairs = overlap * 12.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = max(0.0001, dt * 0.7)
        elif max_force < 0.05:
            dt = min(0.005, dt * 1.1)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y

def _objective_lse(z, alpha=100.0):
    """
    Objective with log-sum-exp smooth penalty.
    z: [x1..xN, y1..yN, r1..rN, w]
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlaps = r_sum[iu] - d_pairs
    
    # Log-sum-exp of overlaps (smooth max)
    # To avoid overflow, shift by max
    if np.max(overlaps) > 0:
        max_ov = np.max(overlaps)
        lse_ov = max_ov + np.log(np.sum(np.exp(alpha * (overlaps - max_ov)))) / alpha
        obj += 100.0 * lse_ov
    else:
        obj += 0.0

    # Boundary penalties
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h
    
    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    if np.max(all_viol) > 0:
        max_v = np.max(all_viol)
        lse_v = max_v + np.log(np.sum(np.exp(alpha * (all_viol - max_v)))) / alpha
        obj += 100.0 * lse_v
    else:
        obj += 0.0

    # Aspect ratio penalty (keep w in [0.4, 1.6])
    if w < 0.4:
        obj += 50.0 * (0.4 - w) ** 2
    elif w > 1.6:
        obj += 50.0 * (w - 1.6) ** 2

    return obj

def _polish_positions(x, y, r, w, h, time_budget):
    """Polish positions via force-directed method, then grow radii."""
    n = N
    x = x.copy()
    y = y.copy()

    max_iter = int(min(2000, time_budget * 100))
    dt = 0.001
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None

def _expand_individual(x, y, r, w, h, time_budget):
    """Try to expand each circle individually by moving it to free space."""
    n = N
    improved = False
    for i in range(n):
        if time.time() > time_budget:
            break
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == i:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            r_max = min(r_max, r_allowed)
        r_max = max(0.0, r_max)

        if r_max > r[i] + 1e-5:
            r[i] = r_max
            improved = True

    if _is_valid(x, y, r, w, h):
        return x, y, r, improved
    return None

def _grid_initial(w0, h0, rows, cols):
    n = N
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x[i] = (col + 0.5) * (w0 / cols)
        y[i] = (row + 0.5) * (h0 / rows)
        r[i] = min(w0 / cols, h0 / rows) * 0.45
        
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    # Phase 1: Multi-start L-BFGS-B with log-sum-exp objective
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]
    
    # We will try a few aspect ratios and for each, do a multi-start optimization
    # To save time, we limit the number of starts and iterations per start
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
            
        # Initial guess: Grid
        # Choose grid dimensions based on aspect ratio
        if w0 > h0:
            rows = 4
            cols = 6
        else:
            rows = 6
            cols = 4
            
        x0, y0, r0 = _grid_initial(w0, h0, rows, cols)
        
        # Relax initial guess
        x_rel, y_rel = _force_relax(x0, y0, r0, w0, h0, max_iter=100)
        
        z0 = np.zeros(3 * N + 1)
        z0[0::3][:N] = x_rel
        z0[1::3][:N] = y_rel
        z0[2::3][:N] = r0
        z0[-1] = w0
        
        # Multi-start: Perturb initial guess
        starts = 3
        for s in range(starts):
            if time.time() > deadline - 30:
                break
                
            if s == 0:
                z_start = z0.copy()
            else:
                # Add noise
                noise_scale = 0.05 * w0
                z_start = z0.copy()
                z_start[0::3][:N] += np.random.normal(0, noise_scale, N)
                z_start[1::3][:N] += np.random.normal(0, noise_scale, N)
                # Keep r same initially, let optimizer adjust
                # Clip to bounds
                z_start[0::3][:N] = np.clip(z_start[0::3][:N], 0.01, w0 - 0.01)
                z_start[1::3][:N] = np.clip(z_start[1::3][:N], 0.01, h0 - 0.01)

            bounds = [(0.01, 1.99)] * (2 * N) + [(0.01, 1.0)] * N + [(0.3, 1.7)]
            
            try:
                res = minimize(_objective_lse, z_start, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                z_opt = res.x
                x_opt = z_opt[0::3][:N]
                y_opt = z_opt[1::3][:N]
                r_opt = z_opt[2::3][:N]
                w_opt = z_opt[-1]
                h_opt = 2.0 - w_opt
                
                # Check validity and repair if necessary
                if not _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                    # Try to repair by scaling radii down
                    scale = 0.95
                    while scale > 0.5:
                        if _is_valid(x_opt, y_opt, r_opt * scale, w_opt, h_opt):
                            r_opt = r_opt * scale
                            break
                        scale *= 0.95
                    else:
                        continue # Skip if invalid even after repair

                val = float(np.sum(r_opt))
                if val > best_value:
                    best_value = val
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)
                    
            except Exception:
                pass

    # Phase 2: Polish the best result found
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 5
        polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_best, h_best)

    # Phase 3: Final expansion
    if best_result is not None and time.time() < deadline - 3:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 1
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, remaining)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_best, h_best)

    if best_result is None:
        # Fallback: Simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
