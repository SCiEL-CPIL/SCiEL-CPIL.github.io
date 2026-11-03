"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Successive Linearization (Convex Relaxation)
"""
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 26

# EVOLVE-BLOCK-START
def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
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
    return float(np.sum(r)), True

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r

def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Fast grid-based local search to fix initial overlaps and find a good starting point.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
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

def _solve_lp(xy, r):
    """
    Solve the LP relaxation of the circle packing problem.
    Variables: x_i, y_i, r_i for i in 0..N-1
    Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    Constraints:
    1. Wall: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    2. Non-overlap (linearized): r_i + r_j <= d_ij + (x_i - x_i0)*ux_ij + (y_i - y_i0)*uy_ij - (x_j - x_j0)*ux_ij - (y_j - y_j0)*uy_ij
       where (x_i0, y_i0) are current positions, and (ux, uy) is unit vector from i to j.
    """
    n = len(xy)
    c = np.zeros(3 * n)
    c[2 * n: 3 * n] = -1.0  # Minimize -sum(r)

    A_ub = []
    b_ub = []

    # 1. Wall constraints
    # r_i - x_i <= 0
    # -r_i - x_i <= 0  (x_i + r_i <= 1)
    # r_i - y_i <= 0
    # -r_i - y_i <= 0  (y_i + r_i <= 1)
    
    # r_i - x_i <= 0
    for i in range(n):
        row = np.zeros(3 * n)
        row[2 * n + i] = 1.0
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
    # x_i + r_i <= 1  =>  x_i + r_i <= 1
    for i in range(n):
        row = np.zeros(3 * n)
        row[i] = 1.0
        row[2 * n + i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
        
    # r_i - y_i <= 0
    for i in range(n):
        row = np.zeros(3 * n)
        row[2 * n + i] = 1.0
        row[n + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
    # y_i + r_i <= 1
    for i in range(n):
        row = np.zeros(3 * n)
        row[n + i] = 1.0
        row[2 * n + i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

    # 2. Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[j, 0] - xy[i, 0]
            dy = xy[j, 1] - xy[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            
            # If circles are exactly on top of each other, pick a random direction
            if dist < 1e-8:
                angle = np.random.uniform(0, 2 * np.pi)
                ux = np.cos(angle)
                uy = np.sin(angle)
            else:
                ux = dx / dist
                uy = dy / dist

            # Constraint: r_i + r_j <= dist + (x_i - x_i0)*ux + (y_i - y_i0)*uy - (x_j - x_j0)*ux - (y_j - y_j0)*uy
            # Rearranged for A_ub * v <= b_ub:
            # -r_i - r_j + ux*(x_i - x_j) + uy*(y_i - y_j) <= dist - ux*(x_i0 - x_j0) - uy*(y_i0 - y_j0)
            # Note: (x_i - x_i0) + (x_j0 - x_j) ... wait.
            # Linearization: dist(x,y) >= dist(x0,y0) + grad_dist . (x - x0)
            # grad_dist wrt x_i is (x_i - x_j)/dist = -ux (if ux is x_j-x_i)/dist? 
            # Let's define u = (x_j - x_i) / dist.
            # dist = |x_j - x_i|.
            # Linear approx: |x_j - x_i| >= |x_j0 - x_i0| + u . ((x_j - x_j0) - (x_i - x_i0))
            # |x_j - x_i| >= d0 + u . (x_j - x_i) - u . (x_j0 - x_i0)
            # We want r_i + r_j <= |x_j - x_i|
            # So r_i + r_j <= d0 + u . (x_j - x_i) - u . (x_j0 - x_i0)
            # -r_i - r_j + u . (x_j - x_i) <= d0 - u . (x_j0 - x_i0)
            
            # u = (ux, uy) where ux = (x_j0 - x_i0)/d0, uy = (y_j0 - y_i0)/d0
            # LHS: -r_i - r_j + ux*(x_j - x_i) + uy*(y_j - y_i)
            # RHS: d0 - ux*(x_j0 - x_i0) - uy*(y_j0 - y_i0)
            
            row = np.zeros(3 * n)
            row[2 * n + i] = -1.0
            row[2 * n + j] = -1.0
            row[i] = -ux
            row[j] = ux
            row[n + i] = -uy
            row[n + j] = uy
            
            A_ub.append(row)
            b_ub.append(dist - ux * (xy[j, 0] - xy[i, 0]) - uy * (xy[j, 1] - xy[i, 1]))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status == 0:
        return res.x
    else:
        return None

def _successive_linearization(xy, r, deadline, max_iter=100):
    """
    Iteratively solve the LP relaxation, updating the linearization point.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, _ = _eval_packing(best_xy, best_r)
    
    cur_xy = xy.copy()
    cur_r = r.copy()
    
    for i in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        v_new = _solve_lp(cur_xy, cur_r)
        if v_new is None:
            break
        
        new_xy = np.column_stack([v_new[0:n], v_new[n:2*n]])
        new_r = v_new[2*n:3*n]
        
        # Check if valid (LP solution might violate non-convex constraints slightly due to linearization gap)
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            if total > best_total:
                best_total = total
                best_xy = new_xy.copy()
                best_r = new_r.copy()
            
            # Update linearization point to the new solution
            cur_xy = new_xy
            cur_r = new_r
            
            # Check for convergence
            if np.linalg.norm(cur_xy - best_xy) < 1e-6 and np.linalg.norm(cur_r - best_r) < 1e-6:
                break
        else:
            # If LP solution is invalid, we might be stuck or need to step back
            # Try a small step towards the LP solution
            alpha = 0.5
            trial_xy = cur_xy + alpha * (new_xy - cur_xy)
            trial_r = cur_r + alpha * (new_r - cur_r)
            
            total, valid = _eval_packing(trial_xy, trial_r)
            if valid:
                if total > best_total:
                    best_total = total
                    best_xy = trial_xy.copy()
                    best_r = trial_r.copy()
                cur_xy = trial_xy
                cur_r = trial_r
            else:
                # If even half step is invalid, break
                break

    return best_xy, best_r

def _smooth_objective(v, n, beta=50.0, penalty=100.0):
    """
    Smooth objective for final polish.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    # Wall violations
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    # Pairwise violations
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = r_sum - dist
    iu = np.triu_indices(n, k=1)
    viol_upper = viol[iu]
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol_upper) / beta)

    return -np.sum(r) + penalty * (wall_pen + pair_pen)

def _smooth_grad(v, n, beta=50.0, penalty=100.0):
    """
    Gradient of the smooth objective.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = -np.ones(n)

    # Wall terms
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    dists_x = np.stack([x, 1.0 - x], axis=0)
    dists_y = np.stack([y, 1.0 - y], axis=0)
    all_dists = np.stack([dists_x, dists_y], axis=1)
    min_idx = np.argmin(all_dists, axis=1)

    for i in range(n):
        s = sig_wall[i] * penalty
        if s < 1e-12:
            continue
        idx = min_idx[i]
        grad_r[i] += s
        if idx == 0:
            grad_x[i] -= s
        elif idx == 1:
            grad_x[i] += s
        elif idx == 2:
            grad_y[i] -= s
        else:
            grad_y[i] += s

    # Pairwise terms
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq + 1e-30)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = r_sum - dist
    sig_pair = 1.0 / (1.0 + np.exp(-beta * viol))

    iu = np.triu_indices(n, k=1)
    sig_upper = sig_pair[iu]
    dx_upper = dx[iu]
    dy_upper = dy[iu]
    dist_upper = dist[iu]

    factor = penalty * sig_upper

    ii = iu[0]
    jj = iu[1]
    np.add.at(grad_x, ii, factor * dx_upper / dist_upper)
    np.add.at(grad_x, jj, -factor * dx_upper / dist_upper)
    np.add.at(grad_y, ii, factor * dy_upper / dist_upper)
    np.add.at(grad_y, jj, -factor * dy_upper / dist_upper)
    np.add.at(grad_r, ii, factor)
    np.add.at(grad_r, jj, factor)

    return np.concatenate([grad_x, grad_y, grad_r])

def _final_polish(xy, r, deadline):
    """
    Final gradient-based polish to squeeze out more value.
    """
    n = len(xy)
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))

    best_xy = xy.copy()
    best_r = r.copy()
    best_total, _ = _eval_packing(best_xy, best_r)

    stages = [
        (20.0, 80.0, 200),
        (50.0, 150.0, 300),
        (100.0, 300.0, 500),
    ]

    current_xy = xy.copy()
    current_r = r.copy()

    for beta, penalty, iters in stages:
        if time.time() > deadline - 5:
            break

        def obj(v, b=beta, p=penalty):
            return _smooth_objective(v, n, b, p)

        def grad(v, b=beta, p=penalty):
            return _smooth_grad(v, n, b, p)

        v0 = np.concatenate([current_xy[:, 0], current_xy[:, 1], current_r])
        try:
            result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': iters, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            continue

        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]

        total, valid = _eval_packing(new_xy, new_r)
        if valid and total > best_total:
            best_total = total
            current_xy = new_xy
            current_r = new_r
            best_xy = new_xy.copy()
            best_r = new_r.copy()

    return best_xy, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + local search to get a valid starting point
    for side in [5, 6]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Successive Linearization (Main Strategy)
    if best_valid and best_xy is not None:
        if time.time() < deadline - 30:
            sl_xy, sl_r = _successive_linearization(best_xy, best_r, deadline)
            total, valid = _eval_packing(sl_xy, sl_r)
            if valid and total > best_total:
                best_total = total
                best_xy = sl_xy
                best_r = sl_r

    # Phase 3: Final Polish
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        fp_xy, fp_r = _final_polish(best_xy, best_r, deadline)
        total, valid = _eval_packing(fp_xy, fp_r)
        if valid and total > best_total:
            best_total = total
            best_xy = fp_xy
            best_r = fp_r

    # Fallback
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

    # Final check
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
