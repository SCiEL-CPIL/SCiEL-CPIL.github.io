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
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
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
                # Initial radius: half the cell size minus a small margin
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    # Pad if needed (shouldn't happen for side=ceil(sqrt(N)))
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init():
    """Generate a random valid packing with small circles."""
    positions = []
    r = 0.02
    for i in range(N):
        x = random.uniform(r, 1 - r)
        y = random.uniform(r, 1 - r)
        positions.append([x, y, r])
    return np.array(positions[:N])


def _best_fit_local_search(xy, r, deadline, grid_size=120):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = np.sum(best_r)
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    # Flatten to (M, 2)
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 80
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)  # (M,)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _sa_packing(xy, r, deadline, initial_temp=0.05, cooling=0.995, max_iter=50000):
    """
    Simulated Annealing to optimize the packing.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(xy, r)
    
    cur_xy = xy.copy()
    cur_r = r.copy()
    cur_total, cur_valid = _eval_packing(cur_xy, cur_r)
    if not cur_valid:
        cur_total = 0.0
    else:
        cur_total = np.sum(cur_r)
        
    temp = initial_temp
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Perturb a random circle
        i = rng.integers(0, n)
        dx = rng.normal(0, temp * 0.5)
        dy = rng.normal(0, temp * 0.5)
        dr = rng.normal(0, temp * 0.2)
        
        new_x = cur_xy[i, 0] + dx
        new_y = cur_xy[i, 1] + dy
        new_r = cur_r[i] + dr
        
        # Bounds
        new_x = np.clip(new_x, 0.001, 0.999)
        new_y = np.clip(new_y, 0.001, 0.999)
        new_r = np.clip(new_r, 0.001, 0.4)
        
        old_x, old_y, old_r = cur_xy[i, 0], cur_xy[i, 1], cur_r[i]
        cur_xy[i, 0] = new_x
        cur_xy[i, 1] = new_y
        cur_r[i] = new_r
        
        new_total, new_valid = _eval_packing(cur_xy, cur_r)
        if not new_valid:
            new_total = 0.0
        else:
            new_total = np.sum(cur_r)
            
        delta = new_total - cur_total
        
        if delta > 0:
            cur_total = new_total
            if new_valid and new_total > best_total:
                best_total = new_total
                best_xy = cur_xy.copy()
                best_r = cur_r.copy()
        else:
            # Accept with probability exp(delta / temp)
            if rng.random() < np.exp(delta / temp):
                cur_total = new_total
            else:
                # Reject
                cur_xy[i, 0] = old_x
                cur_xy[i, 1] = old_y
                cur_r[i] = old_r
                
        temp *= cooling
        
    return best_xy, best_r, best_total, best_valid


def _epigraph_polish(xy, r, deadline):
    """
    High-precision polish using epigraph formulation.
    Variables: x_i, y_i, r_i, t_i
    Minimize: -sum(r_i) + sum(t_i^2)
    Subject to:
      t_i >= r_i - x_i, t_i >= r_i - (1-x_i), t_i >= r_i - y_i, t_i >= r_i - (1-y_i)
      t_i >= r_i + r_j - dist(i,j)
      t_i >= 0
    """
    n = len(xy)
    
    # Params: [x0, y0, r0, t0, x1, y1, r1, t1, ...]
    params = np.zeros(4 * n)
    for i in range(n):
        params[4*i] = xy[i, 0]
        params[4*i+1] = xy[i, 1]
        params[4*i+2] = r[i]
        params[4*i+3] = 0.0 # t_i init
        
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999)) # x
        bounds.append((0.001, 0.999)) # y
        bounds.append((0.001, 0.4))   # r
        bounds.append((0.0, 1.0))     # t
        
    def objective(p):
        x = p[0::4]
        y = p[1::4]
        r = p[2::4]
        t = p[3::4]
        
        # Penalty for t < 0 is handled by bounds, but let's ensure t is used correctly
        # We want t to be the max violation.
        # Objective: -sum(r) + sum(t^2)
        # Constraints (soft): t >= violation
        
        # Wall violations
        v_walls = np.zeros(n)
        for i in range(n):
            vi = np.max([r[i] - x[i], r[i] - (1.0 - x[i]), r[i] - y[i], r[i] - (1.0 - y[i]), 0.0])
            v_walls[i] = vi
            
        # Overlap violations
        v_overlaps = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                v = r[i] + r[j] - dist
                if v > 0:
                    v_overlaps[i] = max(v_overlaps[i], v)
                    v_overlaps[j] = max(v_overlaps[j], v)
                    
        # Soft constraint: t >= v
        # Penalty: sum(max(0, v - t)^2)
        pen = 0.0
        for i in range(n):
            v = max(v_walls[i], v_overlaps[i])
            if v > t[i]:
                pen += (v - t[i])**2
                
        # Also penalize t itself to keep it small? 
        # Actually, minimizing sum(t^2) encourages t -> 0.
        # The constraint penalty encourages t >= v.
        # So t will settle at max(v, 0).
        
        return -np.sum(r) + 100.0 * pen + np.sum(t**2)
        
    try:
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        opt_params = result.x
        opt_xy = np.column_stack([opt_params[0::4], opt_params[1::4]])
        opt_r = opt_params[2::4]
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > 0:
            return opt_xy, opt_r, total, valid
    except Exception:
        pass
        
    return xy, r, 0.0, False


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # 1. Grid Init + Local Search
    for side in [5, 6, 7]:
        if time.time() > deadline - 30.0:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        
        # Step 1: Grid-based local search
        if time.time() < deadline - 20.0:
            out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 20.0, grid_size=120)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r

    # If no valid result found, try random init
    if best_xy is None or not best_valid:
        xy = _random_init()
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        if time.time() < deadline - 20.0:
            out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 20.0, grid_size=120)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r

    # If still no valid result, fall back to a simple grid
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
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # 2. Simulated Annealing
    if time.time() < deadline - 15.0:
        sa_xy, sa_r, sa_total, sa_valid = _sa_packing(best_xy, best_r, deadline - 15.0, initial_temp=0.03, cooling=0.998, max_iter=20000)
        if sa_valid and sa_total > best_total:
            best_total = sa_total
            best_valid = True
            best_xy = sa_xy
            best_r = sa_r

    # 3. Local Search again to clean up SA result
    if time.time() < deadline - 5.0:
        ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(best_xy, best_r, deadline - 5.0, grid_size=120)
        if ls_valid and ls_total > best_total:
            best_total = ls_total
            best_valid = True
            best_xy = ls_xy
            best_r = ls_r

    # 4. Epigraph Polish
    if time.time() < deadline - 2.0:
        ep_xy, ep_r, ep_total, ep_valid = _epigraph_polish(best_xy, best_r, deadline - 2.0)
        if ep_valid and ep_total > best_total:
            best_total = ep_total
            best_valid = True
            best_xy = ep_xy
            best_r = ep_r

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        # Ensure circle is inside unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to simple grid
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
