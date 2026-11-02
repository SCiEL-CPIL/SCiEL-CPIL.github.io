"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21

def _objective(x):
    """
    Objective function for differential_evolution and minimize.
    x = [w, x1, y1, r1, ..., xN, yN, rN]
    Minimize: -sum(r_i) + penalty
    """
    w = x[0]
    h = 2.0 - w
    
    # Extract positions and radii
    # x[1:] is flat array of [x1, y1, r1, x2, y2, r2, ...]
    params = x[1:].reshape(N, 3)
    positions = params[:, :2]
    radii = params[:, 2]
    
    # 1. Boundary Penalties
    # x_i >= r_i, w - x_i >= r_i, y_i >= r_i, h - y_i >= r_i
    # Violation = max(0, r_i - x_i), etc.
    v_left = np.maximum(0, radii - positions[:, 0])
    v_right = np.maximum(0, radii - (w - positions[:, 0]))
    v_bottom = np.maximum(0, radii - positions[:, 1])
    v_top = np.maximum(0, radii - (h - positions[:, 1]))
    boundary_violations = v_left + v_right + v_bottom + v_top
    
    # 2. Overlap Penalties
    # dist_ij >= r_i + r_j
    # Vectorized distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    overlap_violations = np.maximum(0, r_sum - dists)
    overlap_violations = overlap_violations * mask
    
    # Sum of squared violations for smoothness
    penalty = np.sum(boundary_violations**2) + np.sum(overlap_violations**2)
    
    # Objective: Maximize sum(radii) -> Minimize -sum(radii) + large_penalty
    score = np.sum(radii)
    
    # Penalty weight needs to be high enough to dominate score changes if invalid
    # Max possible score is roughly N * 0.1 = 2.1. 
    # Violations can be up to ~1.0.
    # Weight 1000 should be safe.
    weight = 1000.0
    
    return -score + weight * penalty

def _refine_solution(x0):
    """
    Local refinement using L-BFGS-B.
    """
    # Bounds for local refinement
    # w in [0.4, 1.6]
    # For each circle i:
    # x_i in [0, w] -> but w is variable. 
    # L-BFGS-B requires fixed bounds. 
    # Since w is the first variable, and bounds for x_i depend on w, 
    # we cannot easily use L-BFGS-B with dynamic bounds unless we re-parameterize.
    # Alternative: Use Nelder-Mead or Powell which don't require bounds.
    # Or, we can just use differential_evolution with high polish.
    
    # Let's stick to differential_evolution's polish or use a simple gradient step if we had gradients.
    # Since we are using penalty method, we can use L-BFGS-B if we fix w for a moment?
    # No, w is part of the optimization.
    
    # Let's use `minimize` with method='Nelder-Mead' or 'Powell' which handle unconstrained problems well.
    # We will run a short Nelder-Mead on the penalized objective.
    
    res = minimize(_objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    return res.x

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_score = -np.inf
    best_solution = None
    
    # Bounds for differential_evolution
    # w: [0.4, 1.6]
    # For each circle (x, y, r):
    # x, y: [0, 1.6] (since w+h=2, max dim is 1.6 if min is 0.4)
    # r: [0, 0.8] (max radius is half of min dimension)
    
    bounds = [(0.4, 1.6)]
    for _ in range(N):
        bounds.append((0.0, 1.6))
        bounds.append((0.0, 1.6))
        bounds.append((0.0, 0.8))
        
    # Strategy: Multiple restarts of Differential Evolution
    # DE is global, so it might find a good solution in one shot, but restarts help.
    
    num_restarts = 8
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Randomize seed for DE
        seed = i * 100 + 42
        
        try:
            # Run Differential Evolution
            # maxiter and popsize control the search. 
            # For 64 vars, popsize=15 is reasonable.
            # tol and xatol control convergence.
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=seed, 
                maxiter=100, 
                popsize=15, 
                tol=1e-4, 
                xatol=1e-6,
                polish=True  # DE's built-in polish uses L-BFGS-B, which is good
            )
            
            x_best = res.x
            val = _objective(x_best)
            
            # The objective is -score + penalty. 
            # We want to check if the penalty is low enough.
            # If penalty is 0, score = -val.
            # If penalty > 0, it's likely invalid or suboptimal.
            
            # Extract score and penalty for logging/decision
            w = x_best[0]
            h = 2.0 - w
            params = x_best[1:].reshape(N, 3)
            positions = params[:, :2]
            radii = params[:, 2]
            
            # Recalculate true score and penalty to be sure
            v_left = np.maximum(0, radii - positions[:, 0])
            v_right = np.maximum(0, radii - (w - positions[:, 0]))
            v_bottom = np.maximum(0, radii - positions[:, 1])
            v_top = np.maximum(0, radii - (h - positions[:, 1]))
            boundary_violations = v_left + v_right + v_bottom + v_top
            
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            mask = ~np.eye(N, dtype=bool)
            overlap_violations = np.maximum(0, r_sum - dists)
            overlap_violations = overlap_violations * mask
            
            penalty = np.sum(boundary_violations**2) + np.sum(overlap_violations**2)
            score = np.sum(radii)
            
            # Only accept if penalty is very small
            if penalty < 1e-8:
                if score > best_score:
                    best_score = score
                    best_solution = x_best
            else:
                # If penalty is not zero, try to refine locally to fix it?
                # DE's polish might have left it slightly off.
                # Let's try a few Nelder-Mead steps on the penalized objective
                # hoping it drives penalty to 0 while maintaining high score.
                # This is a heuristic "repair" step.
                if penalty < 1.0: # Only try if close
                    res_refine = minimize(_objective, x_best, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                    x_refined = res_refine.x
                    val_refined = _objective(x_refined)
                    
                    # Check refined penalty
                    w_r = x_refined[0]
                    h_r = 2.0 - w_r
                    params_r = x_refined[1:].reshape(N, 3)
                    pos_r = params_r[:, :2]
                    rad_r = params_r[:, 2]
                    
                    v_l = np.maximum(0, rad_r - pos_r[:, 0])
                    v_r = np.maximum(0, rad_r - (w_r - pos_r[:, 0]))
                    v_b = np.maximum(0, rad_r - pos_r[:, 1])
                    v_t = np.maximum(0, rad_r - (h_r - pos_r[:, 1]))
                    bv = v_l + v_r + v_b + v_t
                    
                    diff_r = pos_r[:, np.newaxis, :] - pos_r[np.newaxis, :, :]
                    dists_r = np.linalg.norm(diff_r, axis=2)
                    r_sum_r = rad_r[:, np.newaxis] + rad_r[np.newaxis, :]
                    mask_r = ~np.eye(N, dtype=bool)
                    ov = np.maximum(0, r_sum_r - dists_r)
                    ov = ov * mask_r
                    
                    pen_r = np.sum(bv**2) + np.sum(ov**2)
                    score_r = np.sum(rad_r)
                    
                    if pen_r < 1e-8 and score_r > best_score:
                        best_score = score_r
                        best_solution = x_refined
                        
        except Exception as e:
            pass # Skip failed restarts

    if best_solution is None:
        # Fallback: Simple grid packing
        # 5x5 grid is 25 cells, we need 21.
        # Square 1x1.
        # 5 cols, 5 rows.
        # r = 0.1
        # Positions at (0.1, 0.1), (0.3, 0.1)...
        positions = []
        radii = []
        for i in range(N):
            row = i // 5
            col = i % 5
            x = (col + 0.5) * 0.2
            y = (row + 0.5) * 0.2
            positions.append([x, y])
            radii.append(0.1)
            
        w = 1.0
        h = 1.0
        # Construct result array
        result = np.zeros((N, 3))
        for i in range(N):
            result[i] = [positions[i][0], positions[i][1], radii[i]]
        return result

    # Parse best_solution
    w = best_solution[0]
    h = 2.0 - w
    params = best_solution[1:].reshape(N, 3)
    positions = params[:, :2]
    radii = params[:, 2]
    
    # Final Cleanup: Ensure strict validity by shrinking radii if needed
    # This is a safety net for floating point errors or tiny violations.
    # We iterate a few times to resolve overlaps.
    
    for _ in range(20):
        changed = False
        # Check bounds
        for i in range(N):
            max_r_bound = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
            if radii[i] > max_r_bound - 1e-9:
                radii[i] = max(0, max_r_bound - 1e-9)
                changed = True
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                required = radii[i] + radii[j]
                if dist < required - 1e-9:
                    # Shrink both
                    overlap = required - dist
                    shrink = overlap / 2.0 + 1e-9
                    radii[i] = max(0, radii[i] - shrink)
                    radii[j] = max(0, radii[j] - shrink)
                    changed = True
                    
        if not changed:
            break
            
    # Final Validation
    valid = True
    for i in range(N):
        if positions[i, 0] < -1e-9 or positions[i, 0] > w + 1e-9 or positions[i, 1] < -1e-9 or positions[i, 1] > h + 1e-9:
            valid = False
            break
        if radii[i] < -1e-9:
            valid = False
            break
        max_r_bound = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
        if radii[i] > max_r_bound + 1e-7:
            valid = False
            break
            
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
            if dist < radii[i] + radii[j] - 1e-7:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If still invalid, return the fallback grid
        positions = []
        radii = []
        for i in range(N):
            row = i // 5
            col = i % 5
            x = (col + 0.5) * 0.2
            y = (row + 0.5) * 0.2
            positions.append([x, y])
            radii.append(0.1)
        result = np.zeros((N, 3))
        for i in range(N):
            result[i] = [positions[i][0], positions[i][1], radii[i]]
        return result

    result = np.column_stack([positions, radii])
    return result

# EVOLVE-BLOCK-END
