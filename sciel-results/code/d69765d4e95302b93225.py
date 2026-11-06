"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DEADLINE_SECONDS = 115.0  # Safety margin before 150s limit

def compute_radius(pts):
    """
    Computes the maximum radius for the given centers.
    pts: (N, 2) array of centers.
    Returns r such that circles of radius r are disjoint and inside [-0.5, 0.5]^2.
    """
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = np.min(np.minimum(d_x, d_y))
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = np.sqrt(np.min(dist_sq))
    
    r_inter = min_dist_inter / 2.0
    r = np.minimum(r_inter, min_bound)
    
    return r

def lattice_points(row_lengths, h_spacing, v_spacing, x_offsets, y_offset=0.0):
    """
    Generate points from a parametric hexagonal lattice.
    row_lengths: list of ints, number of circles per row
    h_spacing: horizontal spacing between centers in a row
    v_spacing: vertical spacing between rows
    x_offsets: list of x-offsets for each row (relative to center)
    y_offset: global vertical shift
    Returns (N, 2) array.
    """
    points = []
    n_rows = len(row_lengths)
    total_height = (n_rows - 1) * v_spacing
    y_start = -total_height / 2.0 + y_offset
    
    for i, length in enumerate(row_lengths):
        y = y_start + i * v_spacing
        total_width = (length - 1) * h_spacing
        x_start = -total_width / 2.0 + x_offsets[i]
        
        for j in range(length):
            x = x_start + j * h_spacing
            points.append([x, y])
    
    pts = np.array(points)
    return pts

def lattice_objective(params, row_lengths):
    """
    Objective for lattice parameter optimization.
    params: [h_spacing, v_spacing, x_off_0, x_off_1, ..., x_off_{n-1}, y_offset]
    Returns negative radius (to minimize).
    """
    n_rows = len(row_lengths)
    h_spacing = params[0]
    v_spacing = params[1]
    x_offsets = params[2:2+n_rows]
    y_offset = params[2+n_rows]
    
    if h_spacing <= 0 or v_spacing <= 0:
        return 100.0
    
    pts = lattice_points(row_lengths, h_spacing, v_spacing, x_offsets, y_offset)
    
    # Check bounds
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = np.min(np.minimum(d_x, d_y))
    
    if min_bound < -0.01:
        return 100.0 + abs(min_bound) * 10
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = np.sqrt(np.min(dist_sq))
    
    r_inter = min_dist_inter / 2.0
    r = np.minimum(r_inter, min_bound)
    
    return -r

def optimize_lattice(row_lengths, start_time, deadline):
    """
    Optimize lattice parameters for a given row-length pattern.
    Uses differential evolution for global search, then Nelder-Mead for refinement.
    """
    n_rows = len(row_lengths)
    n_params = 2 + n_rows + 1  # h_spacing, v_spacing, x_offsets..., y_offset
    
    # Bounds: h_spacing in [0.08, 0.2], v_spacing in [0.08, 0.2], offsets in [-0.05, 0.05], y_offset in [-0.05, 0.05]
    bounds = [(0.08, 0.20), (0.08, 0.20)] + [(-0.05, 0.05)] * n_rows + [(-0.05, 0.05)]
    
    # Initial guess: hexagonal packing
    h0 = 1.0 / (max(row_lengths) + 0.5)
    v0 = h0 * np.sqrt(3) / 2.0
    x_offs0 = [0.0] * n_rows
    # Offset alternating rows by half spacing
    for i in range(1, n_rows, 2):
        x_offs0[i] = h0 / 2.0
    y0 = 0.0
    
    x0 = np.array([h0, v0] + x_offs0 + [y0])
    
    # Phase 1: Differential evolution (global)
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                lattice_objective,
                bounds,
                args=(row_lengths,),
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            x_best = de_result.x
        except Exception:
            x_best = x0
    else:
        x_best = x0
    
    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 5:
        try:
            nm_result = minimize(
                lattice_objective,
                x_best,
                args=(row_lengths,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            x_best = nm_result.x
        except Exception:
            pass
    
    # Extract final parameters
    h_spacing = x_best[0]
    v_spacing = x_best[1]
    x_offsets = x_best[2:2+n_rows]
    y_offset = x_best[2+n_rows]
    
    pts = lattice_points(row_lengths, h_spacing, v_spacing, x_offsets, y_offset)
    pts = np.clip(pts, -0.4999, 0.4999)
    r = compute_radius(pts)
    
    return pts, r

def soft_min(vals, beta=50.0):
    """
    Smooth approximation of min using log-sum-exp.
    As beta -> inf, soft_min -> exact min.
    """
    vals = np.asarray(vals, dtype=np.float64)
    m = np.min(vals)
    if m < -1e6:
        return m
    shifted = vals - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta

def smooth_neg_radius(x, beta=50.0):
    """
    Smooth surrogate objective: negative soft-min radius.
    x: flattened (N, 2) array of centers.
    """
    p = x.reshape(-1, 2)
    d_x = 0.5 - np.abs(p[:, 0])
    d_y = 0.5 - np.abs(p[:, 1])
    bound_vals = np.minimum(d_x, d_y)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dists = np.sqrt(np.maximum(dist_sq, 0.0))
    inter_vals = dists / 2.0
    
    all_vals = np.concatenate([bound_vals, inter_vals])
    r_soft = soft_min(all_vals, beta)
    return -r_soft

def free_point_optimize(pts, start_time, deadline):
    """
    Refine free points using multiple optimization passes:
    1. Powell for robustness
    2. L-BFGS-B with smooth soft-min objective for high-precision polish
    """
    def neg_r(x):
        p = x.reshape(-1, 2)
        d_x = 0.5 - np.abs(p[:, 0])
        d_y = 0.5 - np.abs(p[:, 1])
        min_bound = np.min(np.minimum(d_x, d_y))
        if min_bound < 0:
            return -min_bound * 10 + 100.0
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_inter = np.sqrt(np.min(dist_sq))
        r = np.minimum(min_dist_inter / 2.0, min_bound)
        return -r
    
    best_pts = pts.copy()
    best_r = compute_radius(best_pts)
    
    # Phase 1: Powell method
    if time.time() < deadline - 10:
        try:
            result = minimize(
                neg_r,
                pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            candidate_pts = result.x.reshape(-1, 2)
            candidate_pts = np.clip(candidate_pts, -0.49999, 0.49999)
            candidate_r = compute_radius(candidate_pts)
            if candidate_r > best_r:
                best_r = candidate_r
                best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B with smooth soft-min objective (higher beta for precision)
    if time.time() < deadline - 5:
        try:
            bounds = [(-0.5, 0.5)] * (2 * N)
            result2 = minimize(
                smooth_neg_radius,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
            )
            candidate_pts = result2.x.reshape(-1, 2)
            candidate_pts = np.clip(candidate_pts, -0.49999, 0.49999)
            candidate_r = compute_radius(candidate_pts)
            if candidate_r > best_r:
                best_r = candidate_r
                best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Final Powell polish with tighter tolerance
    if time.time() < deadline - 2:
        try:
            result3 = minimize(
                neg_r,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            candidate_pts = result3.x.reshape(-1, 2)
            candidate_pts = np.clip(candidate_pts, -0.49999, 0.49999)
            candidate_r = compute_radius(candidate_pts)
            if candidate_r > best_r:
                best_r = candidate_r
                best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_r

def local_push_apart(pts, start_time, deadline):
    """
    Direct local search: iteratively push apart the closest pair.
    This helps escape local minima where the min-distance pair is stuck.
    """
    best_pts = pts.copy()
    best_r = compute_radius(best_pts)
    
    for iteration in range(200):
        if time.time() > deadline - 2:
            break
        
        p = best_pts
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_idx = np.argmin(dist_sq)
        i, j = np.unravel_index(min_idx, (N, N))
        
        d_vec = p[i] - p[j]
        d_mag = np.sqrt(dist_sq[i, j])
        if d_mag < 1e-12:
            break
        
        # Try small perturbations that increase the min distance
        improved = False
        for step in [0.001, 0.0005, 0.0002, 0.0001]:
            for direction in ['push', 'pull']:
                if direction == 'push':
                    new_i = p[i] + d_vec / d_mag * step
                    new_j = p[j] - d_vec / d_mag * step
                else:
                    # Perpendicular push
                    perp = np.array([-d_vec[1], d_vec[0]]) / d_mag
                    new_i = p[i] + perp * step
                    new_j = p[j] - perp * step
                
                candidate = p.copy()
                candidate[i] = new_i
                candidate[j] = new_j
                
                # Check bounds
                if np.any(np.abs(candidate) > 0.49999):
                    continue
                
                candidate_r = compute_radius(candidate)
                if candidate_r > best_r + 1e-12:
                    best_r = candidate_r
                    best_pts = candidate.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try random small perturbations
            for _ in range(10):
                if time.time() > deadline - 2:
                    break
                perturbation = np.random.normal(0, 0.001, size=(N, 2))
                candidate = best_pts + perturbation
                if np.any(np.abs(candidate) > 0.49999):
                    continue
                candidate_r = compute_radius(candidate)
                if candidate_r > best_r + 1e-12:
                    best_r = candidate_r
                    best_pts = candidate.copy()
                    improved = True
                    break
        
        if not improved and iteration > 50:
            break
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_pts = None
    best_r = -1.0
    
    # Multiple row-length patterns that sum to 50
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4],
        [7, 7, 7, 7, 7, 7, 7, 1],
        [8, 7, 7, 7, 7, 7, 7],
        [6, 7, 7, 7, 7, 7, 6, 3],
        [7, 6, 7, 7, 7, 7, 6, 3],
        [8, 8, 7, 7, 7, 7, 6],
        [7, 7, 7, 7, 7, 7, 7, 1],
        [6, 6, 7, 7, 7, 7, 7, 3],
        [7, 7, 7, 7, 7, 7, 6, 2],
        [5, 7, 7, 7, 7, 7, 7, 3],
    ]
    
    # Filter patterns that sum to N
    valid_patterns = [p for p in patterns if sum(p) == N]
    
    # If no valid patterns, create some
    if not valid_patterns:
        valid_patterns = [[7, 6, 7, 6, 7, 6, 7, 4]]
    
    # Deduplicate patterns
    seen = set()
    unique_patterns = []
    for p in valid_patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    for pattern in unique_patterns:
        if time.time() > deadline - 25:
            break
        
        pts, r = optimize_lattice(pattern, start_time, deadline)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Additional refinement with free-point optimization
        if time.time() < deadline - 15:
            pts_ref, r_ref = free_point_optimize(pts, start_time, deadline)
            if r_ref > best_r:
                best_r = r_ref
                best_pts = pts_ref.copy()
        
        # Local push-apart refinement
        if time.time() < deadline - 8 and best_pts is not None:
            pts_push, r_push = local_push_apart(best_pts, start_time, deadline)
            if r_push > best_r:
                best_r = r_push
                best_pts = pts_push.copy()
    
    # Also try a few random initializations with free-point optimization
    if time.time() < deadline - 20:
        for trial in range(3):
            if time.time() > deadline - 12:
                break
            rand_pts = np.random.uniform(-0.4, 0.4, size=(N, 2))
            pts_ref, r_ref = free_point_optimize(rand_pts, start_time, deadline)
            if r_ref > best_r:
                best_r = r_ref
                best_pts = pts_ref.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < deadline - 5:
        pts_final, r_final = free_point_optimize(best_pts, start_time, deadline)
        if r_final > best_r:
            best_r = r_final
            best_pts = pts_final.copy()
        
        # One more push-apart pass
        if time.time() < deadline - 2:
            pts_push, r_push = local_push_apart(best_pts, start_time, deadline)
            if r_push > best_r:
                best_r = r_push
                best_pts = pts_push.copy()
    
    # Final safety
    if best_pts is None or best_r < 0:
        # Fallback: simple grid
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_r = compute_radius(best_pts)
    
    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts

# EVOLVE-BLOCK-END
