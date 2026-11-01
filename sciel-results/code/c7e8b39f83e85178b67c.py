"""Circle packing in perimeter-4 rectangle: Jiggling + Greedy Max-Radius Expansion."""
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

def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j: continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Phase 1: Jiggling + Expansion
    # We will try multiple aspect ratios and random seeds
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5, 1.1, 0.9]
    
    rng = np.random.default_rng(42)

    for w0 in aspect_ratios:
        if time.time() > deadline - 15:
            break
            
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
            
        # Initialize circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.45
            x[i] = cx
            y[i] = cy
            r[i] = r0

        # Add some random jitter to break symmetry
        for i in range(N):
            x[i] += rng.uniform(-0.01, 0.01)
            y[i] += rng.uniform(-0.01, 0.01)
            # Ensure initial validity roughly
            x[i] = np.clip(x[i], r[i], w0 - r[i])
            y[i] = np.clip(y[i], r[i], h0 - r[i])

        # Jiggling loop
        # We alternate between resolving overlaps and expanding radii
        max_jiggle_iter = 2000
        step_size = 0.05
        
        for it in range(max_jiggle_iter):
            if time.time() > deadline - 15:
                break
                
            # 1. Resolve overlaps via Jiggling
            # Check for overlaps
            overlap_found = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx*dx + dy*dy
                    r_sum = r[i] + r[j]
                    if d2 < r_sum * r_sum:
                        overlap_found = True
                        # Move i and j apart
                        d = np.sqrt(max(d2, 1e-12))
                        if d < 1e-9:
                            # Random direction
                            angle = rng.uniform(0, 2*np.pi)
                            dx_move = np.cos(angle)
                            dy_move = np.sin(angle)
                        else:
                            dx_move = dx / d
                            dy_move = dy / d
                        
                        # Move by a fraction of the overlap
                        # overlap = r_sum - d
                        # We want to increase d by overlap
                        # Move each by overlap/2 in opposite directions
                        # But we can't move more than step_size
                        move = min(step_size, (r_sum - d) * 0.5)
                        
                        new_xi = x[i] + dx_move * move
                        new_yi = y[i] + dy_move * move
                        new_xj = x[j] - dx_move * move
                        new_yj = y[j] - dy_move * move
                        
                        # Check bounds for i
                        if new_xi < r[i] or new_xi > w0 - r[i] or new_yi < r[i] or new_yi > h0 - r[i]:
                             # Try smaller move
                             move = move * 0.5
                             new_xi = x[i] + dx_move * move
                             new_yi = y[i] + dy_move * move
                             new_xj = x[j] - dx_move * move
                             new_yj = y[j] - dy_move * move
                             
                        if new_xi >= r[i] and new_xi <= w0 - r[i] and new_yi >= r[i] and new_yi <= h0 - r[i]:
                            x[i] = new_xi
                            y[i] = new_yi
                            
                        # Check bounds for j
                        if new_xj < r[j] or new_xj > w0 - r[j] or new_yj < r[j] or new_yj > h0 - r[j]:
                             move = move * 0.5
                             new_xj = x[j] - dx_move * move
                             new_yj = y[j] - dy_move * move
                             
                        if new_xj >= r[j] and new_xj <= w0 - r[j] and new_yj >= r[j] and new_yj <= h0 - r[j]:
                            x[j] = new_xj
                            y[j] = new_yj

            # If no overlaps, try to expand
            if not overlap_found:
                # 2. Expand radii greedily
                improved = False
                for i in range(N):
                    r_max = _max_radius_for_circle(x, y, r, w0, h0, i)
                    if r_max > r[i] + 1e-6:
                        r[i] = r_max
                        improved = True
                
                if not improved:
                    # Local optimum reached for this configuration
                    break
                
                # After expanding, we might have created new overlaps? 
                # No, because we set r to the max allowed by current positions.
                # However, expanding one circle might block others.
                # So we loop back to check overlaps.
                
                # Occasionally, if stuck, add a small random jitter to escape local minima
                if it % 100 == 50:
                    for i in range(N):
                        angle = rng.uniform(0, 2*np.pi)
                        dist = rng.uniform(0, step_size * 0.5)
                        new_xi = x[i] + dist * np.cos(angle)
                        new_yi = y[i] + dist * np.sin(angle)
                        if new_xi >= r[i] and new_xi <= w0 - r[i] and new_yi >= r[i] and new_yi <= h0 - r[i]:
                            x[i] = new_xi
                            y[i] = new_yi
                            # Check if this move caused overlap? 
                            # The next iteration's overlap check will handle it.
                            # But if it causes overlap, we just jiggle back.
            
        # Final validity check
        if _is_valid(x, y, r, w0, h0):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

    # Phase 2: SLSQP Refinement on best result
    if best_result is not None and time.time() < deadline - 10:
        x0, y0, r0, w0, h0 = best_result
        
        def objective(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            h = 2.0 - w
            
            if w < 0.2 or h < 0.2 or np.any(r < 0):
                return 1e6
                
            obj = -np.sum(r)
            
            # Pairwise overlap penalty
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            
            iu = np.triu_indices(N, k=1)
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            
            diff = r_sum_pairs - np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            smooth_overlap = np.sqrt(np.maximum(diff, 0.0) ** 2 + 1e-8) - np.sqrt(1e-8)
            obj += 1000.0 * np.sum(smooth_overlap)
            
            # Boundary penalty
            left_viol = r - x
            right_viol = x + r - w
            bottom_viol = r - y
            top_viol = y + r - h
            
            smooth_left = np.sqrt(np.maximum(left_viol, 0.0) ** 2 + 1e-8) - np.sqrt(1e-8)
            smooth_right = np.sqrt(np.maximum(right_viol, 0.0) ** 2 + 1e-8) - np.sqrt(1e-8)
            smooth_bottom = np.sqrt(np.maximum(bottom_viol, 0.0) ** 2 + 1e-8) - np.sqrt(1e-8)
            smooth_top = np.sqrt(np.maximum(top_viol, 0.0) ** 2 + 1e-8) - np.sqrt(1e-8)
            
            obj += 1000.0 * (np.sum(smooth_left) + np.sum(smooth_right) + np.sum(smooth_bottom) + np.sum(smooth_top))
            
            return obj

        z0 = np.zeros(3 * N + 1)
        z0[0::3][:N] = x0
        z0[1::3][:N] = y0
        z0[2::3][:N] = r0
        z0[-1] = w0

        try:
            res = minimize(objective, z0, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': 500, 'ftol': 1e-12, 'maxls': 80})
            
            z_opt = res.x
            x_opt = z_opt[0::3][:N]
            y_opt = z_opt[1::3][:N]
            r_opt = z_opt[2::3][:N]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt
            
            if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                val = float(np.sum(r_opt))
                if val > best_value:
                    best_value = val
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)
        except Exception:
            pass

    # Phase 3: Final Polish (Greedy expansion again on refined result)
    if best_result is not None and time.time() < deadline - 2:
        x, y, r, w, h = best_result
        # Just one more pass of greedy expansion
        for _ in range(5):
            improved = False
            for i in range(N):
                r_max = _max_radius_for_circle(x, y, r, w, h, i)
                if r_max > r[i] + 1e-8:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        if _is_valid(x, y, r, w, h):
             val = float(np.sum(r))
             if val > best_value:
                 best_value = val
                 best_result = (x, y, r, w, h)

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    
    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
