"""Discrete Grid Constraint Satisfaction + Continuous Refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _hex_seed():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        if r <= best_r:
            continue
        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
        centers = np.array(centers[:N])
        r_actual = _score(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
    return best_cents

def _grid_csp_solve(grid_res, r_target, deadline):
    """
    Discrete CSP: place N circles on grid points with minimum distance 2*r_target.
    Uses backtracking with arc-consistency pruning.
    Returns list of grid indices (or None if infeasible).
    """
    elapsed = time.time() - deadline[0]
    if elapsed > deadline[1]:
        return None
    
    # Grid points
    g = grid_res
    xs = np.linspace(-0.5 + r_target, 0.5 - r_target, g)
    ys = np.linspace(-0.5 + r_target, 0.5 - r_target, g)
    
    # Precompute valid grid indices (within bounds)
    valid_mask = (xs >= -0.5 + r_target) & (xs <= 0.5 - r_target) & \
                 (ys >= -0.5 + r_target) & (ys <= 0.5 - r_target)
    
    # Build grid of valid points
    grid_points = []
    for yi in range(g):
        for xi in range(g):
            if valid_mask[yi] and valid_mask[xi]:
                grid_points.append((yi, xi))
    
    n_points = len(grid_points)
    if n_points < N:
        return None
    
    # Precompute distance matrix between grid points
    pts = np.array([(ys[yi], xs[xi]) for yi, xi in grid_points])
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Minimum distance requirement
    min_dist = 2.0 * r_target
    
    # Precompute conflict graph: which points conflict with which
    # conflict[i] = set of j that are too close to i
    conflict = [set() for _ in range(n_points)]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dists[i, j] < min_dist - 1e-10:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Sort grid points by degree (most constrained first) for better pruning
    degree_order = sorted(range(n_points), key=lambda i: -len(conflict[i]))
    
    # Backtracking with MRV heuristic
    assignment = [-1] * N  # assignment[k] = grid point index for circle k
    used_points = [False] * n_points
    
    def backtrack(k):
        if time.time() - deadline[0] > deadline[1]:
            return False
        
        if k == N:
            return True
        
        # MRV: pick unassigned circle with fewest available options
        best_k = -1
        best_options = None
        min_count = n_points + 1
        
        for ck in range(k, N):
            if assignment[ck] != -1:
                continue
            # Count available points
            count = 0
            for pi in range(n_points):
                if not used_points[pi]:
                    count += 1
            if count < min_count:
                min_count = count
                best_k = ck
                if count == 0:
                    break
                if count == 1:
                    break
        
        if best_k == -1 or min_count == 0:
            return False
        
        # Try all available points for best_k
        for pi in degree_order:
            if used_points[pi]:
                continue
            
            # Check if this point conflicts with any already-assigned points
            valid = True
            for ck in range(N):
                if assignment[ck] != -1:
                    if pi in conflict[assignment[ck]]:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Assign
            assignment[best_k] = pi
            used_points[pi] = True
            
            if backtrack(k + 1):
                return True
            
            # Unassign
            assignment[best_k] = -1
            used_points[pi] = False
        
        return False
    
    # Start with circle 0 at center-ish point for symmetry breaking
    # Find point closest to center
    center_idx = 0
    min_dist_center = float('inf')
    for i in range(n_points):
        d = pts[i, 0]**2 + pts[i, 1]**2
        if d < min_dist_center:
            min_dist_center = d
            center_idx = i
    
    assignment[0] = center_idx
    used_points[center_idx] = True
    
    if backtrack(1):
        return assignment
    else:
        return None

def _discrete_to_continuous(assignment, grid_res, r_target):
    """Convert discrete grid assignment to continuous positions."""
    g = grid_res
    xs = np.linspace(-0.5 + r_target, 0.5 - r_target, g)
    ys = np.linspace(-0.5 + r_target, 0.5 - r_target, g)
    
    centers = np.zeros((N, 2))
    for k in range(N):
        pi = assignment[k]
        yi, xi = pi // g, pi % g
        centers[k] = [xs[xi], ys[yi]]
    
    return centers

def _refine_continuous(cents, r_target, time_limit):
    """
    Refine continuous positions to maximize radius starting from discrete solution.
    Uses L-BFGS-B on violation loss.
    """
    r_est = _score(cents)
    r_target_eff = max(r_est, r_target)
    
    x0 = cents.reshape(-1).copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def _violation_loss(x, r):
        c = x.reshape(N, 2)
        wall_pen_x = np.maximum(0.0, np.abs(c[:, 0]) - (0.5 - r))
        wall_pen_y = np.maximum(0.0, np.abs(c[:, 1]) - (0.5 - r))
        wall_loss = np.sum(wall_pen_x**2) + np.sum(wall_pen_y**2)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlaps = np.maximum(0.0, 2.0 * r - dists)
        pair_loss = np.sum(overlaps**2)
        
        return wall_loss + pair_loss
    
    start = time.time()
    try:
        result = minimize(
            _violation_loss, x0, args=(r_target_eff,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50, 'maxfun': 800}
        )
        
        x_best = result.x
        loss_best = _violation_loss(x_best, r_target_eff)
        
        r_new = _score(x_best.reshape(N, 2))
        if r_new >= r_est:
            return x_best.reshape(N, 2)
        else:
            return cents
    except Exception:
        return cents

def _binary_search_radius(cents_init, time_budget):
    """
    Binary search on radius with discrete CSP + continuous refinement.
    """
    start_time = time.time()
    deadline = [start_time, time_budget]
    
    r_low = _score(cents_init)
    r_high = 0.072  # Slightly above known best
    
    # First, try to establish a feasible r_high
    best_cents = cents_init.copy()
    best_r = r_low
    
    # Try several radii in binary search
    for _ in range(15):
        if time.time() - start_time > time_budget - 5:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try discrete CSP at this radius
        grid_res = max(20, int(20 * (0.072 / max(r_mid, 1e-3))))
        grid_res = min(grid_res, 40)  # Cap grid resolution
        
        assignment = _grid_csp_solve(grid_res, r_mid, deadline)
        
        if assignment is not None:
            # Discrete solution found, refine continuously
            cents_disc = _discrete_to_continuous(assignment, grid_res, r_mid)
            cents_ref = _refine_continuous(cents_disc, r_mid, min(5.0, time_budget - (time.time() - start_time)))
            
            r_new = _score(cents_ref)
            if r_new >= r_mid - 1e-6:
                # Feasible! Update bounds
                best_cents = cents_ref.copy()
                best_r = max(best_r, r_new)
                r_low = r_mid
            else:
                r_high = r_mid
        else:
            # Infeasible at this radius
            r_high = r_mid
        
        # Also try direct continuous optimization from current best
        if time.time() - start_time < time_budget - 3:
            cents_ref2 = _refine_continuous(best_cents, r_low * 1.001, min(3.0, time_budget - (time.time() - start_time)))
            r_new2 = _score(cents_ref2)
            if r_new2 > best_r:
                best_cents = cents_ref2.copy()
                best_r = r_new2
                r_low = min(r_low, r_new2)
    
    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Generate initial hexagonal seed
    cents = _hex_seed()
    best_r = _score(cents)
    best_cents = cents.copy()
    
    # 2. Binary search with discrete CSP + continuous refinement
    time_for_search = deadline - (time.time() - start_time) - 10  # Leave 10s for final polish
    time_for_search = max(time_for_search, 10)
    
    best_cents, best_r = _binary_search_radius(best_cents, time_for_search)
    
    # 3. Final continuous polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        final_polish = _refine_continuous(best_cents, best_r, min(remaining - 2, 10))
        final_r = _score(final_polish)
        if final_r > best_r:
            best_cents = final_polish
    
    best_cents = np.clip(best_cents, -0.5, 0.5)
    return best_cents
# EVOLVE-BLOCK-END
