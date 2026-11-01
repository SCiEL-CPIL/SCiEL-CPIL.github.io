"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def check_validity(positions, radii, w, h, tol=1e-7):
        """Check if packing is valid and return sum of radii if so."""
        # Wall checks
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            if x < r - tol or x > w - r + tol:
                return -1.0
            if y < r - tol or y > h - r + tol:
                return -1.0

        # Pairwise checks
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx * dx + dy * dy
                r_sum = radii[i] + radii[j]
                if dist_sq < (r_sum - tol) ** 2:
                    return -1.0

        return np.sum(radii)

    def objective(z):
        """
        Objective function for minimization.
        z: [x0, y0, r0, ..., x20, y20, r20, w]
        Minimize: -sum(r) + penalty
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        # Hard bounds check to avoid nan/inf
        if w < 0.2 or h < 0.2:
            return 1e6

        # Clip radii to physically possible range to keep objective smooth
        r_clip = np.clip(r, 0.001, min(w, h) / 2.0 - 0.001)
        
        base_obj = -np.sum(r_clip)

        # Wall penalties
        # Distance to walls must be >= r
        # Penalty if x < r or x > w-r
        x_low_viol = np.maximum(r_clip - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clip), 0.0)
        y_low_viol = np.maximum(r_clip - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clip), 0.0)
        
        wall_pen = 100.0 * (np.sum(x_low_viol**2) + np.sum(x_high_viol**2) + 
                            np.sum(y_low_viol**2) + np.sum(y_high_viol**2))

        # Overlap penalties
        # Vectorized pairwise distance
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx*dx + dy*dy
        r_sum = r_clip[:, None] + r_clip[None, :]
        
        # Only upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        
        # Smooth approximation of max(r_sum - dist, 0)
        # dist = sqrt(d_sq)
        # We want penalty if dist < r_sum
        # penalty = sum(max(r_sum - dist, 0)^2)
        
        # To avoid sqrt(0) issues and keep it differentiable, we can use:
        # max(r_sum - sqrt(d_sq), 0)
        # Let's compute dist
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_pen = 100.0 * np.sum(overlap**2)

        return base_obj + wall_pen + overlap_pen

    def get_initial_guesses():
        """Generate diverse initial guesses."""
        guesses = []
        
        # Aspect ratios: w + h = 2.0
        # Try a range of aspect ratios
        w_vals = np.linspace(0.5, 1.5, 11)
        
        for w in w_vals:
            h = 2.0 - w
            if w < 0.3 or h < 0.3:
                continue
            
            # Grid initialization
            # 21 circles.
            # Try 3x7, 4x6 (with some merging), 5x5 (with merging)
            
            # 3x7 Grid
            rows, cols = 3, 7
            positions = np.zeros((N, 2))
            radii = np.zeros(N)
            cell_w = w / cols
            cell_h = h / rows
            
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= N: break
                    cx = (c + 0.5) * cell_w
                    cy = (r + 0.5) * cell_h
                    # Max radius in cell is min(cell_w, cell_h)/2
                    # We use 0.95 to leave a small gap for initial validity
                    max_r = min(cell_w, cell_h) / 2.0 * 0.95
                    positions[idx] = [cx, cy]
                    radii[idx] = max_r
                    idx += 1
            
            # 4x6 Grid (24 cells, pick 21)
            # We can just use 3x7 as it's exact. 
            # Let's add some random perturbations to the 3x7 grid
            for _ in range(5):
                pos_copy = positions.copy()
                rad_copy = radii.copy()
                # Jitter positions
                pos_copy += np.random.normal(0, 0.02, pos_copy.shape)
                # Ensure they stay in bounds roughly
                pos_copy[:, 0] = np.clip(pos_copy[:, 0], rad_copy, w - rad_copy)
                pos_copy[:, 1] = np.clip(pos_copy[:, 1], rad_copy, h - rad_copy)
                
                # Check validity
                if check_validity(pos_copy, rad_copy, w, h) > 0:
                    guesses.append((w, h, pos_copy, rad_copy))
            
            # Add the base grid if valid
            if check_validity(positions, radii, w, h) > 0:
                guesses.append((w, h, positions, radii))
                
        # Also try a "hexagonal" like start
        # 21 circles in a hex pattern?
        # Just stick to grids for simplicity and reliability.
        
        return guesses

    def polish(w, h, positions, radii):
        """
        Optimize the packing.
        Returns best (positions, radii, w, h, score) found.
        """
        z0 = np.concatenate([positions.flatten(), [w]])
        
        # Bounds
        bounds = []
        for i in range(N):
            # x, y bounds
            bounds.append((0.0, w))
            bounds.append((0.0, h))
            # r bounds
            bounds.append((0.01, min(w, h)/2.0))
        # w bounds
        bounds.append((0.3, 1.7))
        
        best_result = None
        best_score = check_validity(positions, radii, w, h)
        
        # Run SLSQP
        try:
            res = minimize(
                objective, 
                z0, 
                method='SLSQP', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.fun < 0: # If it found something negative (meaning sum(r) > penalties)
                z_opt = res.x
                x_opt = z_opt[0::3][:N]
                y_opt = z_opt[1::3][:N]
                r_opt = z_opt[2::3][:N]
                w_opt = z_opt[-1]
                h_opt = 2.0 - w_opt
                pos_opt = np.column_stack([x_opt, y_opt])
                
                score = check_validity(pos_opt, r_opt, w_opt, h_opt)
                if score > best_score:
                    best_score = score
                    best_result = (pos_opt, r_opt, w_opt, h_opt, score)
        except Exception:
            pass
            
        # Local search / Expansion phase
        if best_result is not None:
            pos_best, r_best, w_best, h_best, score_best = best_result
            
            # Iterative expansion
            for _ in range(20):
                if elapsed() > deadline - 5:
                    break
                
                improved = False
                # Try to expand each circle
                for i in range(N):
                    old_r = r_best[i]
                    # Try small increments
                    for dr in [0.005, 0.002, 0.001]:
                        new_r = old_r + dr
                        if new_r > min(w_best, h_best)/2.0 - 0.001:
                            continue
                        
                        # Check if we can place circle i with new_r
                        # We try to keep position same, or move slightly
                        # Simple check: is it valid with same pos?
                        valid = True
                        x, y = pos_best[i]
                        
                        # Walls
                        if x < new_r - 1e-7 or x > w_best - new_r + 1e-7:
                            valid = False
                        if y < new_r - 1e-7 or y > h_best - new_r + 1e-7:
                            valid = False
                        
                        if valid:
                            # Overlaps
                            for j in range(N):
                                if i == j: continue
                                dx = x - pos_best[j, 0]
                                dy = y - pos_best[j, 1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < new_r + r_best[j] - 1e-7:
                                    valid = False
                                    break
                        
                        if valid:
                            r_best[i] = new_r
                            improved = True
                            break
                            
                if not improved:
                    break
                
                # Update score
                score_new = check_validity(pos_best, r_best, w_best, h_best)
                if score_new > score_best:
                    score_best = score_new
                    best_result = (pos_best, r_best, w_best, h_best, score_best)
                else:
                    # If it became invalid, revert? 
                    # The loop above only updates if valid, so it shouldn't become invalid.
                    pass
                    
        return best_result

    # Main logic
    best_overall = None
    best_overall_score = 0.0
    
    guesses = get_initial_guesses()
    
    # Sort guesses by initial sum of radii
    guesses.sort(key=lambda g: -np.sum(g[3]))
    
    # Limit number of guesses to try to fit in time
    # Each polish takes ~0.1-0.5s. 140s / 0.5s = ~280 iterations.
    # We have ~20-30 good guesses.
    
    for i, (w, h, pos, rad) in enumerate(guesses):
        if elapsed() > deadline - 10:
            break
            
        result = polish(w, h, pos, rad)
        
        if result is not None:
            pos_r, rad_r, w_r, h_r, score_r = result
            if score_r > best_overall_score:
                best_overall_score = score_r
                best_overall = np.column_stack([pos_r, rad_r])
                
                # If we found a very good solution, we can stop early or refine more
                # Let's keep going to try other aspect ratios

    # Fallback
    if best_overall is None:
        # Return a trivial valid solution
        w, h = 1.0, 1.0
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        # 5x5 grid in 1x1 box? No, 21 circles.
        # 3x7 grid
        cell_w, cell_h = 1.0/7, 1.0/3
        idx = 0
        for r in range(3):
            for c in range(7):
                pos[idx] = [(c+0.5)*cell_w, (r+0.5)*cell_h]
                rad[idx] = min(cell_w, cell_h)/2 * 0.9
                idx += 1
        best_overall = np.column_stack([pos, rad])

    return best_overall
# EVOLVE-BLOCK-END
