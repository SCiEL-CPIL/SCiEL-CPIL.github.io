"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # seconds wall-clock limit (evaluator kills at 150s, use 145s safety)
SAFETY_MARGIN = 5.0


def _is_valid(arr, w, h, tol=1e-9):
    """Check if circles are valid: disjoint interiors, inside bounding box [0,w]x[0,h]."""
    n = arr.shape[0]
    if n != N:
        return False
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    # Check boundaries
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    # Disjoint interiors - vectorized
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = rs[i_idx] + rs[j_idx]
    if np.any(dist < min_dist - tol):
        return False
    
    return True


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    return obj


def _polish_result(arr, w, h, time_budget=10.0):
    """Polish the result by expanding radii and adjusting positions."""
    start = time.time()
    best = arr.copy()
    best_val = np.sum(best[:, 2])
    
    # Iteratively try to expand circles
    while time.time() - start < time_budget:
        improved = False
        for i in range(N):
            # Try to expand radius i
            dr = 0.001
            arr_try = best.copy()
            arr_try[i, 2] += dr
            if _is_valid(arr_try, w, h):
                best = arr_try
                best_val = np.sum(best[:, 2])
                improved = True
            else:
                # Try to move circle i to make room for expansion
                for _ in range(5):
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    arr_try2 = best.copy()
                    arr_try2[i, 0] += dx
                    arr_try2[i, 1] += dy
                    arr_try2[i, 2] = best[i, 2] + dr
                    if _is_valid(arr_try2, w, h):
                        best = arr_try2
                        best_val = np.sum(best[:, 2])
                        improved = True
                        break
        if not improved:
            break
            
    return best, best_val


def construct():
    start_time = time.time()
    
    best_arr = None
    best_w = 1.0
    best_h = 1.0
    best_sum = 0.0
    
    # Try different initial aspect ratios (widths)
    # w + h = 2, so w in [0.2, 1.8]
    initial_widths = [1.0, 1.2, 0.8, 1.4, 0.6, 1.1, 0.9, 1.3, 0.7, 1.5, 0.5]
    
    for w0 in initial_widths:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        h0 = 2.0 - w0
        
        # Generate initial configuration
        z0 = np.zeros(3 * N + 1)
        side = int(np.ceil(np.sqrt(N)))
        
        # Grid initialization
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.45
            z0[3 * i] = cx
            z0[3 * i + 1] = cy
            z0[3 * i + 2] = r0
        z0[-1] = w0
        
        # Relax initial overlaps using simple force-directed steps
        for _ in range(50):
            x = z0[0::3][:N]
            y = z0[1::3][:N]
            r = z0[2::3][:N]
            w = z0[-1]
            h = 2.0 - w
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Boundary forces
            fx -= np.maximum(r - x, 0) * 10
            fx += np.maximum(x + r - w, 0) * 10
            fy -= np.maximum(r - y, 0) * 10
            fy += np.maximum(y + r - h, 0) * 10
            
            # Overlap forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    r_sum = r[i] + r[j]
                    if d < r_sum and d > 1e-6:
                        f = (r_sum - d) * 10
                        fx[i] += f * dx / d
                        fx[j] -= f * dx / d
                        fy[i] += f * dy / d
                        fy[j] -= f * dy / d
            
            # Update
            z0[0::3][:N] += fx * 0.01
            z0[1::3][:N] += fy * 0.01
            z0[2::3][:N] = np.maximum(z0[2::3][:N] - 0.001, 0.01) # Shrink slightly to help separation
            
        # SLSQP optimization
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
            bounds.append((0.0, 2.0)) # y
            bounds.append((0.01, 1.0)) # r
        bounds.append((0.2, 1.8)) # w
        
        try:
            res = minimize(_vectorized_objective, z0, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10})
            
            z_opt = res.x
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt
            
            arr_opt = np.zeros((N, 3))
            arr_opt[:, 0] = z_opt[0::3][:N]
            arr_opt[:, 1] = z_opt[1::3][:N]
            arr_opt[:, 2] = z_opt[2::3][:N]
            
            if _is_valid(arr_opt, w_opt, h_opt):
                val = np.sum(arr_opt[:, 2])
                if val > best_sum:
                    best_sum = val
                    best_arr = arr_opt.copy()
                    best_w = w_opt
                    best_h = h_opt
        except Exception:
            pass
            
    # Final polish
    if best_arr is not None and time.time() - start_time < DEADLINE - 5.0:
        remaining = DEADLINE - 5.0 - (time.time() - start_time)
        polished, val = _polish_result(best_arr, best_w, best_h, time_budget=remaining)
        if val > best_sum:
            best_sum = val
            best_arr = polished
    
    if best_arr is None:
        # Fallback
        side = 5
        best_arr = np.zeros((N, 3))
        for idx in range(N):
            i = idx // side
            j = idx % side
            best_arr[idx, 0] = (j + 0.5) / side
            best_arr[idx, 1] = (i + 0.5) / side
            best_arr[idx, 2] = 0.5 / side - 0.01
        best_w = 1.0
        best_h = 1.0

    return best_arr
