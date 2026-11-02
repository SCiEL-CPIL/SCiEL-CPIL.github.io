"""Pack 50 equal circles in the unit square centred at the origin."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 50
HALF = 0.5

_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def _compute_radius(centers):
    """Compute the maximum common radius for a valid configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    wall = HALF - np.abs(centers)
    r_wall = np.min(wall)
    if r_wall < 0:
        return 0.0
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0
    return min(r_pair, r_wall)


def _soft_min_obj(x, beta):
    """
    Objective: -r + (1/beta) * log(sum(exp(-beta * slack_i)))
    where slack_i are the constraint margins.
    This is a smooth approximation of max(-r, -min_slack) but we want to
    maximize r subject to slacks >= 0.
    
    Actually, let's use: minimize -r + (1/beta)*log(sum(exp(-beta*s_i)))
    where s_i = min(d_ij - 2r, wall_i - r) for the tightest constraints.
    
    Better formulation: We want to maximize r such that all d_ij >= 2r and wall_i >= r.
    Let slacks = [d_ij - 2r for all i<j, wall_i - r for all i,k]
    We want min(slacks) >= 0.
    Objective: f(x) = -r + (1/beta) * log(sum(exp(-beta * s_i)))
    As beta -> inf, if min(s_i) < 0, f -> -r - min(s_i) (penalty for violation)
    If min(s_i) >= 0, f -> -r + (1/beta)*log(n) ~ -r
    So minimizing f pushes r up while penalizing violations.
    """
    c = x[:2 * N].reshape(N, 2)
    r = x[2 * N]
    
    # Compute pairwise distances
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d_pair = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Wall distances
    wall = HALF - np.abs(c)  # shape (N, 2)
    
    # Slacks
    slacks_pair = d_pair - 2 * r
    slacks_wall = wall - r  # shape (N, 2)
    
    all_slacks = np.concatenate([slacks_pair, slacks_wall.flatten()])
    
    # Soft min: (1/beta) * log(sum(exp(-beta * s)))
    # Numerically stable
    neg_beta_s = -beta * all_slacks
    max_val = np.max(neg_beta_s)
    if max_val > 700:  # avoid overflow
        log_sum = max_val + np.log(np.sum(np.exp(neg_beta_s - max_val)))
    else:
        log_sum = np.log(np.sum(np.exp(neg_beta_s)))
    
    soft_min_term = log_sum / beta
    
    obj = -r + soft_min_term
    return obj


def _soft_min_grad(x, beta):
    """Gradient of the soft min objective."""
    c = x[:2 * N].reshape(N, 2)
    r = x[2 * N]
    
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d_pair = np.sqrt(np.sum(diff ** 2, axis=1))
    
    wall = HALF - np.abs(c)
    
    slacks_pair = d_pair - 2 * r
    slacks_wall = wall - r
    
    all_slacks = np.concatenate([slacks_pair, slacks_wall.flatten()])
    
    neg_beta_s = -beta * all_slacks
    max_val = np.max(neg_beta_s)
    exp_vals = np.exp(neg_beta_s - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # Gradient of soft_min_term w.r.t. slacks: -weights
    # d(soft_min)/d(s_i) = -weights[i]
    
    # d obj / d r = -1 + d(soft_min)/dr
    # soft_min depends on r through slacks:
    # For pair: ds/dr = -2
    # For wall: ds/dr = -1
    
    grad_r = -1.0
    n_pair = len(iu[0])
    weights_pair = weights[:n_pair]
    weights_wall = weights[n_pair:]
    
    # d(soft_min)/dr = sum_i weights_i * (-ds_i/dr) ... wait
    # soft_min = (1/beta)*log(sum(exp(-beta*s_i)))
    # d(soft_min)/d s_i = -exp(-beta*s_i)/sum(exp(-beta*s_j)) = -weights[i]
    # d(soft_min)/dr = sum_i (d soft_min/d s_i) * (d s_i / dr)
    # = sum_i (-weights[i]) * (d s_i / dr)
    
    # For pairs: d s_i / dr = -2
    # For walls: d s_i / dr = -1
    grad_r += np.sum(weights_pair * (-2.0))
    grad_r += np.sum(weights_wall * (-1.0))
    
    # Gradient w.r.t. centers
    grad_c = np.zeros_like(c)
    
    # Pairs: d s_ij / d c_i = (c_i - c_j) / d_ij
    # d s_ij / d c_j = (c_j - c_i) / d_ij = -(c_i - c_j) / d_ij
    for idx in range(n_pair):
        if weights_pair[idx] > 1e-15:
            i, j = iu[0][idx], iu[1][idx]
            dij = d_pair[idx]
            if dij > 1e-12:
                unit = (c[i] - c[j]) / dij
                # d(soft_min)/d c_i = -weights[idx] * d s_ij / d c_i = -weights[idx] * unit
                grad_c[i] += -weights_pair[idx] * unit
                grad_c[j] += -weights_pair[idx] * (-unit)
    
    # Walls: d s_ik / d c_ik = -sign(c_ik)
    for i in range(N):
        for k in range(2):
            if weights_wall[i * 2 + k] > 1e-15:
                grad_c[i, k] += -weights_wall[i * 2 + k] * (-np.sign(c[i, k]))
    
    grad = np.concatenate([grad_c.flatten(), [grad_r]])
    return grad


def _initial_hex_grid():
    """Create a hexagonal grid initial guess for 50 circles."""
    # Hexagonal packing: rows with alternating offsets
    # For 50 circles, use a pattern like 7-8-7-8-7-7-6 = 50 or similar
    # Let's use a 7-column hex grid
    points = []
    row_height = np.sqrt(3) / 2  # for unit spacing
    spacing = 0.13  # approximate spacing for r~0.07
    
    rows = [7, 8, 7, 8, 7, 8, 5]  # total = 50
    y = -0.4
    for row_idx, count in enumerate(rows):
        x_start = -0.4 + (count % 2) * spacing / 2
        for i in range(count):
            x = x_start + i * spacing
            points.append([x, y])
        y += spacing * row_height
    
    points = np.array(points[:N])
    # Center and scale
    points -= np.mean(points, axis=0)
    # Scale to fit in square
    max_abs = np.max(np.abs(points))
    if max_abs > 0.45:
        points *= 0.45 / max_abs
    return points


def _initial_square_grid():
    """Create a square grid initial guess."""
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _simulated_annealing(c, r, t_max=30.0):
    """Simple SA to improve the packing."""
    c = c.copy()
    best_c = c.copy()
    best_r = r
    best_score = _compute_radius(c)
    
    t = t_max
    t_min = 0.001
    cooling = 0.995
    step_size = 0.05
    
    start_time = time.time()
    while t > t_min and (time.time() - start_time) < t_max:
        # Propose move
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        c_new = c.copy()
        c_new[idx] += delta
        
        # Keep inside square
        c_new = np.clip(c_new, -HALF + 0.001, HALF - 0.001)
        
        score_new = _compute_radius(c_new)
        
        if score_new > best_score:
            best_score = score_new
            best_c = c_new.copy()
        
        # Accept/reject
        if score_new >= _compute_radius(c) or np.random.random() < np.exp((score_new - _compute_radius(c)) / t):
            c = c_new
        
        t *= cooling
        step_size = max(0.001, 0.05 * t / t_max)
    
    return best_c, best_score


def _lbfgs_optimize(c, r, beta, max_iter=500, time_limit=20.0):
    """Optimize using L-BFGS with soft min objective."""
    x0 = np.concatenate([c.flatten(), [r]])
    
    def obj(x):
        return _soft_min_obj(x, beta)
    
    def grad(x):
        return _soft_min_grad(x, beta)
    
    start_time = time.time()
    try:
        res = minimize(
            obj,
            x0,
            jac=grad,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-15,
                'gtol': 1e-10,
                'maxls': 50,
            }
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_opt = res.x[:2 * N].reshape(N, 2)
            r_opt = res.x[2 * N]
            score = _compute_radius(c_opt)
            return c_opt, score
    except Exception:
        pass
    return c, _compute_radius(c)


def construct():
    """Return centres of N equal circles in [-1/2,1/2]^2."""
    np.random.seed(42)
    
    best_c = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        _initial_hex_grid(),
        _initial_square_grid(),
        _initial_square_grid() + np.random.normal(0, 0.02, (N, 2)),
    ]
    
    for c_init in initializations:
        c_init = np.clip(c_init, -HALF + 0.01, HALF - 0.01)
        r_init = _compute_radius(c_init)
        
        if best_c is None or r_init > best_score:
            best_c = c_init.copy()
            best_score = r_init
        
        # SA phase
        if _remaining() > 20.0:
            c_sa, score_sa = _simulated_annealing(c_init, r_init, t_max=min(15.0, _remaining() * 0.3))
            if score_sa > best_score:
                best_score = score_sa
                best_c = c_sa.copy()
        
        # L-BFGS polishing with increasing beta
        if _remaining() > 10.0:
            c_curr = best_c.copy()
            r_curr = best_score
            for beta in [100, 500, 2000, 10000]:
                if _remaining() < 5.0:
                    break
                time_limit = min(15.0, _remaining() * 0.5)
                c_opt, score_opt = _lbfgs_optimize(c_curr, r_curr, beta, max_iter=300, time_limit=time_limit)
                if score_opt > best_score:
                    best_score = score_opt
                    best_c = c_opt.copy()
                    c_curr = c_opt.copy()
                else:
                    break
        
        if _remaining() < 15.0:
            break
    
    # Final high-precision polish
    if best_c is not None and _remaining() > 5.0:
        for beta in [50000, 100000]:
            if _remaining() < 3.0:
                break
            c_opt, score_opt = _lbfgs_optimize(best_c, best_score, beta, max_iter=200, time_limit=min(10.0, _remaining() * 0.5))
            if score_opt > best_score:
                best_score = score_opt
                best_c = c_opt.copy()
    
    if best_c is None:
        best_c = _initial_square_grid()
    
    # Ensure validity
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    return best_c
