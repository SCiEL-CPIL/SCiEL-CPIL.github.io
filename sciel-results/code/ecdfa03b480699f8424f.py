"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# Global deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 135.0  # 15s safety margin before 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _compute_radius(pos):
    """Vectorized packing radius: min(min_pair/2, min_boundary)."""
    n = len(pos)
    if n == 0:
        return 0.0
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _soft_min_objective_and_grad(flat, beta):
    """
    Objective: minimize -soft_min(r) where r = min(min_pair/2, min_boundary).
    Uses log-sum-exp soft min: soft_min(c) = -beta * log(sum(exp(-c_i/beta)))
    We minimize -soft_min = beta * log(sum(exp(-c_i/beta)))
    
    flat: shape (2N,) flattened positions
    beta: temperature parameter (smaller = sharper)
    Returns (objective, gradient)
    """
    p = flat.reshape(N, 2)
    n = N
    
    # Boundary constraints: for each point i, two constraints
    # bx_i = 0.5 - |x_i|, by_i = 0.5 - |y_i|
    bx = 0.5 - np.abs(p[:, 0])
    by = 0.5 - np.abs(p[:, 1])
    
    # Pair constraints: for i < j, d_ij/2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]  # shape (n*(n-1)/2,)
    pair_constraints = pair_dists / 2.0
    
    # All constraints in one array
    # Boundary: 2n values
    # Pair: n*(n-1)/2 values
    all_constraints = np.concatenate([bx, by, pair_constraints])
    
    # Soft min: -beta * log(sum(exp(-c/beta)))
    # Objective to minimize: beta * log(sum(exp(-c/beta)))
    # For numerical stability, use log-sum-exp trick
    neg_c_over_beta = -all_constraints / beta
    max_val = np.max(neg_c_over_beta)
    if max_val > 700:
        # All constraints are very negative, objective is huge
        return 1e6, np.zeros_like(flat)
    
    exp_vals = np.exp(neg_c_over_beta - max_val)
    log_sum = max_val + np.log(np.sum(exp_vals))
    obj = beta * log_sum
    
    # Gradient: d(obj)/dp = sum_i [exp(-c_i/beta) / sum_j exp(-c_j/beta)] * (-1/beta) * dc_i/dp
    # = -sum_i [w_i * dc_i/dp] where w_i = exp(-c_i/beta) / sum_j exp(-c_j/beta)
    
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros((n, 2))
    
    # Boundary x gradients
    # c = 0.5 - |x|, dc/dx = -sign(x)
    w_bx = weights[:n]
    w_by = weights[n:2*n]
    
    # For bx_i: dc/dx_i = -sign(x_i)
    sign_x = np.sign(p[:, 0])
    sign_x[np.abs(p[:, 0]) < 1e-15] = 1.0  # handle zero
    grad[:, 0] += w_bx * (-sign_x) * (-1.0 / beta)
    
    # For by_i: dc/dy_i = -sign(y_i)
    sign_y = np.sign(p[:, 1])
    sign_y[np.abs(p[:, 1]) < 1e-15] = 1.0
    grad[:, 1] += w_by * (-sign_y) * (-1.0 / beta)
    
    # Pair gradients
    # c_ij = |p_i - p_j|/2
    # dc_ij/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
    # dc_ij/dp_j = (p_j - p_i) / (2 * |p_i - p_j|)
    
    w_pair = weights[2*n:]  # shape (n*(n-1)/2,)
    
    # Build full weight matrix for pairs
    w_mat = np.zeros((n, n))
    w_mat[iu] = w_pair
    w_mat = w_mat + w_mat.T  # symmetric
    
    # For each i: grad[i] += sum_j w_mat[i,j] * (p_i - p_j) / (2 * dist_ij) * (-1/beta)
    # Note: dist_ij = 2 * c_ij, so (p_i - p_j)/(2*dist_ij) = (p_i-p_j)/(4*c_ij)
    # But let's use dist directly: (p_i - p_j) / (2 * dist_ij)
    
    # Vectorized computation
    # factor[i,j] = w_mat[i,j] / (2 * dist[i,j]) * (-1/beta)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = w_mat / (2.0 * dist)
    factor = np.where(dist > 1e-12, factor, 0.0)
    factor = factor * (-1.0 / beta)
    
    # grad[i] = sum_j factor[i,j] * (p_i - p_j)
    # = sum_j factor[i,j] * diff[i,j]
    grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    grad += grad_pair
    
    return obj, grad.flatten()


def _generate_hex_lattice(spacing, jitter=0.0, seed=0):
    """Generate hexagonal lattice points inside [-0.5, 0.5]^2 with optional jitter."""
    dy = spacing * math.sqrt(3.0) / 2.0
    max_coord = 0.5
    points = []
    row = 0
    y = -max_coord
    while y <= max_coord:
        x_offset = 0.0 if (row % 2 == 0) else spacing / 2.0
        x = -max_coord + x_offset
        while x <= max_coord:
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append((x, y))
            x += spacing
        y += dy
        row += 1
    
    if not points:
        return np.zeros((0, 2))
    
    pts = np.array(points, dtype=float)
    
    if jitter > 0 and len(pts) >= N:
        rng = np.random.RandomState(seed)
        # Select N points with jitter
        # First, sort by some criterion and take first N, or random sample
        if len(pts) > N:
            idx = rng.choice(len(pts), N, replace=False)
            pts = pts[idx]
        pts = pts + rng.randn(N, 2) * jitter
        pts = np.clip(pts, -0.4999, 0.4999)
    
    return pts


def _force_directed_polish(pos, iterations=200, k=0.001):
    """Force-directed relaxation to resolve overlaps."""
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    if n < 2:
        return pos
    
    target_r = _compute_radius(pos)
    if target_r <= 0:
        target_r = 0.035
    
    rep_strength = target_r * 2.0
    
    for it in range(iterations):
        if _time_left() < 10.0:
            break
        
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=target_r * 2.5)
        
        forces = np.zeros_like(pos)
        
        for i, j in pairs:
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-8
                dist = np.linalg.norm(d)
            d_norm = d / dist
            f = rep_strength / (dist * dist)
            forces[i] += d_norm * f
            forces[j] -= d_norm * f
        
        center_pull = -pos * 0.0005
        pos += (forces + center_pull) * k
        pos = np.clip(pos, -0.4999, 0.4999)
    
    return pos


def _optimize_from_start(pos0, beta_start=50.0, beta_end=20.0, max_iter=300):
    """Run L-BFGS-B optimization from a starting position."""
    if _time_left() < 15.0:
        return pos0
    
    flat0 = pos0.flatten()
    
    # Adaptive beta: start coarse, end fine
    best_flat = flat0.copy()
    best_r = _compute_radius(pos0)
    
    # Multi-stage: start with larger beta, decrease
    betas = [beta_start, beta_start * 0.5, beta_end]
    
    for beta in betas:
        if _time_left() < 12.0:
            break
        
        def obj_and_grad(flat):
            return _soft_min_objective_and_grad(flat, beta)
        
        try:
            res = minimize(
                obj_and_grad,
                best_flat,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            best_flat = res.x
            best_flat = np.clip(best_flat, -0.4999, 0.4999)
        except Exception:
            pass
        
        current_r = _compute_radius(best_flat.reshape(N, 2))
        if current_r > best_r:
            best_r = current_r
    
    return best_flat.reshape(N, 2)


def construct():
    best_pos = None
    best_r = 0.0
    
    # Multiple starting configurations from hexagonal lattices with different spacings and jitters
    spacings = [0.036, 0.034, 0.032, 0.030, 0.028]
    jitters = [0.001, 0.003, 0.005]
    seeds = [0, 1, 2, 3, 4]
    
    attempts = 0
    max_attempts = 8
    
    for sp in spacings:
        if _time_left() < 20.0 or attempts >= max_attempts:
            break
        
        for jitter in jitters[:2]:
            if _time_left() < 18.0 or attempts >= max_attempts:
                break
            
            for seed in seeds[:2]:
                if _time_left() < 16.0 or attempts >= max_attempts:
                    break
                
                attempts += 1
                try:
                    pos0 = _generate_hex_lattice(sp, jitter=jitter, seed=seed)
                    if len(pos0) < N:
                        continue
                    
                    # Quick force-directed polish before gradient optimization
                    if _time_left() > 20.0:
                        pos0 = _force_directed_polish(pos0, iterations=100, k=0.0005)
                    
                    # Gradient optimization
                    pos_opt = _optimize_from_start(pos0, beta_start=50.0, beta_end=20.0, max_iter=200)
                    
                    # Final force-directed polish
                    if _time_left() > 15.0:
                        pos_opt = _force_directed_polish(pos_opt, iterations=100, k=0.0003)
                    
                    r = _compute_radius(pos_opt)
                    if r > best_r:
                        best_r = r
                        best_pos = pos_opt.copy()
                except Exception:
                    continue
    
    # Fallback if nothing worked
    if best_pos is None or best_r < 0.02:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best_pos = grid.copy()
    
    if best_pos is None:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_r = _compute_radius(best_pos)
    
    # Final high-precision polish if time allows
    if _time_left() > 15.0:
        try:
            final_pos = _optimize_from_start(best_pos, beta_start=30.0, beta_end=15.0, max_iter=300)
            final_pos = _force_directed_polish(final_pos, iterations=150, k=0.0002)
            r_final = _compute_radius(final_pos)
            if r_final > best_r:
                best_r = r_final
                best_pos = final_pos
        except Exception:
            pass
    
    # Ensure validity
    best_pos = np.asarray(best_pos, dtype=float)
    if best_pos.shape != (N, 2):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_pos = grid
    best_pos = np.clip(best_pos, -0.49999, 0.49999)
    if not np.all(np.isfinite(best_pos)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
