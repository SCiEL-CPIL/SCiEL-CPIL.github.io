"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n):
    """Random initialization with small radii."""
    xy = np.random.rand(n, 2)
    r = np.full(n, 0.01)
    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=60):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = np.sum(best_r)
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _sa_packing(xy_init, r_init, deadline, n_restarts=5):
    """
    Simulated Annealing for circle packing.
    State: positions (N,2) and radii (N,).
    Move: pick a random circle, perturb its position and radius.
    Objective: maximize sum(r) subject to constraints (penalized).
    """
    n = len(xy_init)
    best_total = 0.0
    best_xy = xy_init.copy()
    best_r = r_init.copy()
    
    # Initial check
    init_total, init_valid = _eval_packing(xy_init, r_init)
    if init_valid:
        best_total = init_total
        best_xy = xy_init.copy()
        best_r = r_init.copy()
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Start from best found so far or a perturbed version
        if restart == 0:
            current_xy = xy_init.copy()
            current_r = r_init.copy()
        else:
            current_xy = best_xy.copy()
            current_r = best_r.copy()
            # Perturb
            current_xy += rng.normal(0, 0.05, size=(n, 2))
            current_r = np.maximum(current_r * (1 + rng.normal(0, 0.2, size=n)), 0.001)
            # Clip to square
            for i in range(n):
                current_xy[i, 0] = np.clip(current_xy[i, 0], current_r[i], 1.0 - current_r[i])
                current_xy[i, 1] = np.clip(current_xy[i, 1], current_r[i], 1.0 - current_r[i])
        
        # Annealing schedule
        T_start = 0.1
        T_end = 0.0001
        n_steps = 5000
        cooling_rate = (T_end / T_start) ** (1.0 / n_steps)
        
        current_total, current_valid = _eval_packing(current_xy, current_r)
        if not current_valid:
            current_total = -1.0  # Penalize invalid states heavily
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
                
            T = T_start * (cooling_rate ** step)
            
            # Pick a random circle
            i = rng.integers(0, n)
            
            # Perturb position
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            new_x = np.clip(current_xy[i, 0] + dx, 0.0, 1.0)
            new_y = np.clip(current_xy[i, 1] + dy, 0.0, 1.0)
            
            # Perturb radius (grow or shrink)
            if rng.random() < 0.7:
                # Try to grow
                new_r = current_r[i] * (1.0 + rng.uniform(0, 0.1))
            else:
                # Shrink
                new_r = current_r[i] * (1.0 - rng.uniform(0, 0.1))
            new_r = max(new_r, 0.001)
            new_r = min(new_r, 0.5)
            
            # Ensure circle is inside square
            new_r = min(new_r, new_x, 1.0 - new_x, new_y, 1.0 - new_y)
            if new_r < 0.001:
                continue
                
            # Save old state
            old_x, old_y, old_r = current_xy[i, 0], current_xy[i, 1], current_r[i]
            
            # Apply new state
            current_xy[i, 0] = new_x
            current_xy[i, 1] = new_y
            current_r[i] = new_r
            
            # Evaluate new state
            new_total, new_valid = _eval_packing(current_xy, current_r)
            if not new_valid:
                new_total = -1.0
            
            # Acceptance probability
            delta = new_total - current_total
            if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
                current_total = new_total
                if new_valid and new_total > best_total:
                    best_total = new_total
                    best_xy = current_xy.copy()
                    best_r = current_r.copy()
            else:
                # Reject, revert
                current_xy[i, 0] = old_x
                current_xy[i, 1] = old_y
                current_r[i] = old_r
                
    return best_xy, best_r, best_total


def _gradient_optimize(xy, r, deadline, max_iter=200, penalty_weight=500.0):
    """
    Use L-BFGS-B with penalty terms to optimize the packing.
    """
    n = len(xy)
    
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.4))
    
    best_params = params.copy()
    best_total = 0.0
    best_valid = False
    
    init_total, init_valid = _eval_packing(xy, r)
    if init_valid:
        best_total = init_total
        best_valid = True
    
    def _penalized_objective(params, n, penalty_weight):
        params = np.asarray(params)
        xy = params[0::3].reshape(-1, 2)
        r = params[1::3].reshape(-1)
        r = np.maximum(r, 0.001)
        
        wall_violation = 0.0
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            wall_violation += max(0.0, ri - x)**2
            wall_violation += max(0.0, ri - (1.0 - x))**2
            wall_violation += max(0.0, ri - y)**2
            wall_violation += max(0.0, ri - (1.0 - y))**2
        
        overlap_violation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_violation += (min_dist - dist)**2
        
        penalty = penalty_weight * (wall_violation + overlap_violation)
        total_radius = np.sum(r)
        return -total_radius + penalty

    try:
        result = minimize(
            _penalized_objective,
            params,
            args=(n, penalty_weight),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    if time.time() < deadline - 5.0 and best_valid:
        try:
            result2 = minimize(
                _penalized_objective,
                best_params,
                args=(n, penalty_weight * 5),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-10}
            )
            opt_params2 = result2.x
            opt_xy2 = opt_params2[0::3].reshape(-1, 2)
            opt_r2 = np.maximum(opt_params2[1::3], 0.001)
            
            total2, valid2 = _eval_packing(opt_xy2, opt_r2)
            if valid2 and total2 > best_total:
                best_total = total2
                best_valid = True
                best_params = opt_params2.copy()
                
        except Exception:
            pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    
    return final_xy, final_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Try multiple grid sizes for initialization
    for side in [5, 6]:
        if time.time() > deadline - 40.0:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        
        # Step 1: Grid-based local search
        if time.time() < deadline - 30.0:
            out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 30.0, grid_size=50)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r
        
        # Step 2: Simulated Annealing (NEW)
        if time.time() < deadline - 15.0 and best_valid:
            sa_xy, sa_r, sa_total = _sa_packing(
                best_xy if best_xy is not None else xy,
                best_r if best_r is not None else r,
                deadline - 15.0,
                n_restarts=3
            )
            if sa_total > best_total:
                best_total = sa_total
                best_valid = True
                best_xy = sa_xy
                best_r = sa_r
        
        # Step 3: Gradient-based optimization
        if time.time() < deadline - 5.0 and best_valid:
            g_xy, g_r, g_total, g_valid = _gradient_optimize(
                best_xy if best_xy is not None else xy,
                best_r if best_r is not None else r,
                deadline - 2.0,
                max_iter=100,
                penalty_weight=500.0
            )
            if g_valid and g_total > best_total:
                best_total = g_total
                best_valid = True
                best_xy = g_xy
                best_r = g_r
    
    # If no valid result found, fall back to a simple grid
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            side = int(np.ceil(np.sqrt(N)))
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
