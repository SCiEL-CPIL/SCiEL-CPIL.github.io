"""
Circle packing: 200 equal circles in unit square [-0.5, 0.5]^2.
Combines soft-min L-BFGS-B refinement with force-based local search (steepest descent
on the exact min-distance objective) for structural diversity.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _compute_radius(pts):
    """Compute the maximum common radius for the given centres."""
    boundary_min = np.min(0.5 - np.abs(pts))
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(min_pair, 2.0 * boundary_min) / 2.0
    return r


def _objective_and_grad(x, t=50.0):
    """
    Soft-min objective for L-BFGS-B.
    x: flattened array of shape (2*N,)
    Returns (f, g).
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    bound_dists = np.stack([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ], axis=1)
    bound_flat = bound_dists.flatten()
    
    pair_flat = d.flatten()
    pair_flat = pair_flat[pair_flat < np.inf]
    
    all_dists = np.concatenate([pair_flat, bound_flat])
    
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    exp_vals = np.exp(-t * shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = min_val - log_sum / t
    
    f = -soft_min_val
    
    weights = exp_vals / np.sum(exp_vals)
    
    grad_x = np.zeros_like(x)
    
    n_pairs = len(pair_flat)
    pair_weights = weights[:n_pairs]
    bound_weights = weights[n_pairs:]
    
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = pair_weights[idx]
            dist = d[i, j]
            if dist > 1e-12:
                diff = pts[i] - pts[j]
                dir_vec = diff / dist
                grad_x[2*i:2*i+2] += -w * dir_vec
                grad_x[2*j:2*j+2] += w * dir_vec
            idx += 1
    
    for i in range(N):
        w_left = bound_weights[4*i]
        w_right = bound_weights[4*i+1]
        w_bottom = bound_weights[4*i+2]
        w_top = bound_weights[4*i+3]
        
        grad_x[2*i] += -w_left + w_right
        grad_x[2*i+1] += -w_bottom + w_top
    
    return f, grad_x


def _force_step(pts, step_size):
    """
    One step of force-based local search.
    Each point is pushed away from its nearest neighbors and toward the center
    if close to boundary. This is a steepest-ascent step on the min-distance.
    """
    pts = pts.copy()
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    forces = np.zeros_like(pts)
    
    for i in range(N):
        # Find nearest neighbor
        nearest_j = np.argmin(d[i])
        dist = d[i, nearest_j]
        if dist < 1e-10:
            continue
        # Repulsive force from nearest neighbor
        diff = pts[i] - pts[nearest_j]
        force = diff / dist
        forces[i] += force
        
        # Boundary forces
        for dim in range(2):
            if pts[i, dim] < -0.4:
                forces[i, dim] += (0.5 - pts[i, dim]) * 0.1
            elif pts[i, dim] > 0.4:
                forces[i, dim] -= (pts[i, dim] + 0.5) * 0.1
    
    pts += step_size * forces
    pts = np.clip(pts, -0.499, 0.499)
    return pts


def _local_search(pts, max_iters=200, time_limit=10.0):
    """
    Local search using alternating force steps and small perturbations.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    step_size = 0.01
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Force-based step
        new_pts = _force_step(pts, step_size)
        new_r = _compute_radius(new_pts)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            pts = new_pts
            step_size = max(0.001, step_size * 1.05)  # grow step slightly
        else:
            pts = new_pts
            step_size = max(0.001, step_size * 0.95)  # shrink step
        
        # Occasionally add small random perturbation to escape local minima
        if it % 50 == 0:
            perturbation = np.random.default_rng(it).normal(0, 0.005, size=pts.shape)
            perturbed = np.clip(pts + perturbation, -0.499, 0.499)
            pert_r = _compute_radius(perturbed)
            if pert_r > best_r:
                best_r = pert_r
                best_pts = perturbed.copy()
                pts = perturbed
    
    return best_pts, best_r


def _hex_init(n, rng):
    """Initialize n points in a hexagonal lattice pattern."""
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    rows = int(np.ceil(1.0 / dy)) + 2
    cols = int(np.ceil(1.0 / dx)) + 2
    pts = []
    for i in range(rows):
        y = -0.5 + i * dy
        offset = 0.0 if i % 2 == 0 else dx / 2.0
        for j in range(cols):
            x = -0.5 + j * dx + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) >= n:
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]
    else:
        result = np.empty((n, 2))
        result[:len(pts)] = pts
        result[len(pts):] = rng.uniform(-0.45, 0.45, size=(n - len(pts), 2))
        return result


def _random_init(n, rng):
    """Random initialization inside the unit square."""
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _grid_init(n, rng):
    """Grid initialization with slight random jitter."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= n:
        pts = grid[:n]
    else:
        pts = grid
        extra = rng.uniform(-0.4, 0.4, size=(n - len(pts), 2))
        pts = np.vstack([pts, extra])
    pts = pts + rng.normal(0, 0.01, size=pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    return pts


def construct():
    """Construct 200 circle centres in the unit square maximizing common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5 second safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = -np.inf
    
    # Multi-start with different initializations
    inits = [
        _hex_init(N, rng),
        _random_init(N, rng),
        _grid_init(N, rng),
        _hex_init(N, rng),
    ]
    
    for start_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Phase 1: L-BFGS-B refinement
        x0 = init_pts.flatten()
        try:
            res = minimize(_objective_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 3000,
                                    'ftol': 1e-8, 'maxls': 50})
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                pts_opt = res.x.reshape(-1, 2)
                pts_opt = np.clip(pts_opt, -0.4999, 0.4999)
                r = _compute_radius(pts_opt)
                
                if r > best_r:
                    best_r = r
                    best_pts = pts_opt.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break
        
        # Phase 2: Force-based local search from the L-BFGS-B result
        if best_pts is not None:
            try:
                ls_pts, ls_r = _local_search(best_pts, max_iters=300, time_limit=8.0)
                if ls_r > best_r:
                    best_r = ls_r
                    best_pts = ls_pts
            except Exception:
                pass
        
        if time.time() > deadline - 10:
            break
    
    # Final polish: one more local search from the best result
    if best_pts is not None and time.time() < deadline - 5:
        try:
            ls_pts, ls_r = _local_search(best_pts, max_iters=200, time_limit=5.0)
            if ls_r > best_r:
                best_r = ls_r
                best_pts = ls_pts
        except Exception:
            pass
    
    # If no valid result, return a simple grid
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
