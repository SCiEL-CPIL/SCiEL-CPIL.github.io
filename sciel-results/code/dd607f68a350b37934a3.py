import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 148.0  # Safety margin before 150s limit

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _violation_fast(pts, r):
    """Compute max violation: max(0, 2r - d_ij, r - w_i). Returns (violation, min_dist_pair, min_wall)."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    min_pair = np.min(dist)
    
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    min_wall = np.min(wall_dist)
    
    viol_pair = max(0.0, 2.0 * r - min_pair)
    viol_wall = max(0.0, r - min_wall)
    return max(viol_pair, viol_wall), min_pair, min_wall

def _softplus(x, beta):
    """Numerically stable softplus: log(1 + exp(beta*x))/beta"""
    x_clipped = np.clip(x, -50/beta, 50/beta)
    return np.log1p(np.exp(beta * x_clipped)) / beta

def _sigmoid(x, beta):
    """Numerically stable sigmoid: 1/(1+exp(-beta*x))"""
    x_clipped = np.clip(x, -50/beta, 50/beta)
    return 1.0 / (1.0 + np.exp(-beta * x_clipped))

def _soft_min_obj(pts, r, beta):
    """
    Smooth surrogate for max constraint violation using log-sum-exp.
    obj = (1/beta) * log(sum(exp(beta * violation_k)))
    where violations are 2r - d_ij for pairs and r - w_i for walls.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_viol = 2.0 * r - dist[iu]
    
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r - wall_dist
    
    all_viol = np.concatenate([pair_viol, wall_viol])
    
    # Log-sum-exp for numerical stability
    max_v = np.max(all_viol)
    if max_v > 50.0 / beta:
        return max_v + np.log(np.sum(np.exp(beta * (all_viol - max_v)))) / beta
    else:
        return np.log(np.sum(np.exp(beta * all_viol))) / beta

def _soft_min_grad(pts, r, beta):
    """
    Gradient of _soft_min_obj w.r.t. pts and r.
    Returns (grad_pts, grad_r).
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_viol = 2.0 * r - dist[iu]
    
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r - wall_dist
    
    all_viol = np.concatenate([pair_viol, wall_viol])
    
    # Softmax weights
    max_v = np.max(all_viol)
    exp_v = np.exp(beta * (all_viol - max_v))
    weights = exp_v / np.sum(exp_v)
    
    n_pairs = len(pair_viol)
    w_pair = weights[:n_pairs]
    w_wall = weights[n_pairs:]
    
    # Gradient w.r.t. r
    grad_r = np.sum(w_pair * 2.0) + np.sum(w_wall * 1.0)
    
    # Gradient w.r.t. pts from pairs
    grad_pts = np.zeros_like(pts)
    sig_full = np.zeros((N, N))
    sig_full[iu] = w_pair
    sig_full = sig_full + sig_full.T
    
    unit_vec = diff / dist[:, :, np.newaxis]
    # d obj/d pts_i = - sum_j w_ij * (pts_i - pts_j)/dist_ij
    grad_pts -= np.sum(sig_full[:, :, np.newaxis] * unit_vec, axis=1)
    
    # Gradient w.r.t. pts from walls (hard min gradient)
    for i in range(N):
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        s = w_wall[i]
        
        if abs(dl - wmin) < 1e-10:
            grad_pts[i, 0] -= s * 1.0
        if abs(dr - wmin) < 1e-10:
            grad_pts[i, 0] += s * 1.0
        if abs(db - wmin) < 1e-10:
            grad_pts[i, 1] -= s * 1.0
        if abs(dt - wmin) < 1e-10:
            grad_pts[i, 1] += s * 1.0
    
    return grad_pts, grad_r

def _packing_objective_and_grad(x, beta, lambda_r):
    """
    Objective: soft_min(violations) - lambda_r * r
    x: flattened array [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, r]
    Uses log-sum-exp based soft-min for smoother optimization.
    """
    c = x[:-1].reshape(N, 2)
    r = x[-1]
    
    obj_val = _soft_min_obj(c, r, beta) - lambda_r * r
    grad_c, grad_r = _soft_min_grad(c, r, beta)
    
    grad = np.zeros_like(x)
    grad[:-1] = grad_c.ravel()
    grad[-1] = grad_r - lambda_r
    
    return float(obj_val), grad

def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def _simulated_annealing(pts, rng, max_iter=5000, t_start=0.01, t_end=0.0001, deadline=None):
    """
    Simulated annealing to improve packing.
    Moves one circle at a time, accepts worse solutions with probability exp(-delta/T).
    """
    current = pts.copy()
    current_r = _compute_r(current)
    best = current.copy()
    best_r = current_r
    
    n_iter = 0
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Schedule temperature
        alpha = (i + 1) / max_iter
        T = t_start * (t_end / t_start) ** alpha
        
        # Choose a random circle to move
        idx = rng.integers(N)
        old_pos = current[idx].copy()
        
        # Propose a new position (small random step)
        step = rng.normal(0, T, size=2)
        new_pos = old_pos + step
        new_pos = np.clip(new_pos, -HALF, HALF)
        
        current[idx] = new_pos
        new_r = _compute_r(current)
        
        # Accept if better, or with probability exp(-(old_r - new_r)/T) if worse
        delta = current_r - new_r  # positive means worse
        if delta <= 0 or rng.random() < np.exp(-delta / (T + 1e-15)):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            current[idx] = old_pos
        
        n_iter += 1
    
    return best, best_r

def _lbfgs_polish(pts, r, beta, lambda_r, deadline, max_iter=300):
    """L-BFGS polishing with given beta and lambda_r."""
    x0 = np.concatenate([pts.ravel(), [r]])
    
    def obj_func(x):
        return _packing_objective_and_grad(x, beta, lambda_r)[0]
    
    def jac_func(x):
        return _packing_objective_and_grad(x, beta, lambda_r)[1]
    
    try:
        res = minimize(
            obj_func,
            x0,
            jac=jac_func,
            method='L-BFGS-B',
            bounds=[(-HALF, HALF)] * (2 * N) + [(0.0, HALF)],
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
        )
        
        c_opt = res.x[:-1].reshape(N, 2)
        c_opt = np.clip(c_opt, -HALF, HALF)
        r_achieved = _compute_r(c_opt)
        return c_opt, r_achieved
    except Exception:
        return pts, r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    # Initializations - more diverse
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.005),
        _hex_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.005),
        _grid_init(rng, jitter=0.02),
        _random_init(rng),
        _random_init(rng),
    ]
    
    # Optimization parameters
    lambda_r = 10.0
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
            
        current_pts = init
        current_r = _compute_r(current_pts)
        
        if current_r > best_r:
            best_r = current_r
            best_pts = current_pts.copy()
        
        # Phase 1: Simulated annealing for exploration
        if time.time() < deadline - 20.0:
            sa_pts, sa_r = _simulated_annealing(
                current_pts, rng,
                max_iter=3000,
                t_start=0.008,
                t_end=0.0005,
                deadline=deadline - 15.0
            )
            if sa_r > best_r:
                best_r = sa_r
                best_pts = sa_pts.copy()
            current_pts = sa_pts
            current_r = sa_r
        
        # Phase 2: L-BFGS polishing with increasing beta
        beta_schedule = [100, 500, 2000, 5000]
        for beta in beta_schedule:
            if time.time() > deadline - 5.0:
                break
            
            polished_pts, polished_r = _lbfgs_polish(
                current_pts, current_r, beta, lambda_r,
                deadline, max_iter=200
            )
            
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts.copy()
            
            current_pts = polished_pts
            current_r = polished_r
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 1.0:
        for beta in [10000, 20000]:
            if time.time() > deadline - 0.5:
                break
            polished_pts, polished_r = _lbfgs_polish(
                best_pts, best_r, beta, lambda_r,
                deadline, max_iter=100
            )
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts
    
    return best_pts
