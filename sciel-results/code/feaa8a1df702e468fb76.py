import numpy as np
import time
from scipy.optimize import minimize
import scipy.spatial.distance as sd

N = 32

def _smooth_penalty(xy, r, beta=50.0):
    """
    Smooth approximation of overlap penalty using log-sum-exp.
    For each pair (i,j), violation = r_i + r_j - dist(i,j).
    We want to penalize max(0, violation) smoothly.
    """
    n = len(r)
    # Compute all pairwise distances
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    # Sum of radii for each pair
    sum_r = r[:, None] + r[None, :]
    # Violation: positive when overlapping
    viol = sum_r - dist
    # Only consider upper triangle (i < j) to avoid double counting
    iu = np.triu_indices(n, k=1)
    viol_pairs = viol[iu]
    # Smooth max(0, viol) via softplus-like: log(1 + exp(beta * viol)) / beta
    # For large positive viol, this approximates viol; for negative, approximates 0
    smooth_viol = np.log1p(np.exp(beta * viol_pairs)) / beta
    return np.sum(smooth_viol)

def _boundary_penalty(xy, r):
    """
    Penalty for circles extending outside [0,1]^2.
    """
    n = len(r)
    # For each circle, distance to each boundary
    d_left = xy[:, 0] - r
    d_right = 1.0 - xy[:, 0] - r
    d_bottom = xy[:, 1] - r
    d_top = 1.0 - xy[:, 1] - r
    # Penalty for negative distances (outside)
    viol = np.concatenate([d_left, d_right, d_bottom, d_top])
    smooth_viol = np.log1p(np.exp(50.0 * viol)) / 50.0
    return np.sum(smooth_viol)

def _objective(params, beta=50.0):
    """
    Objective to minimize: -sum(r) + penalty.
    params: (3n,) vector = [x0,y0,r0, x1,y1,r1, ...]
    """
    n = len(params) // 3
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    # Ensure radii are positive
    r = np.maximum(r, 0.001)
    overlap_pen = _smooth_penalty(xy, r, beta)
    bound_pen = _boundary_penalty(xy, r)
    # We want to maximize sum(r), so minimize -sum(r) + penalties
    return -np.sum(r) + 10.0 * overlap_pen + 10.0 * bound_pen

def _objective_grad(params, beta=50.0):
    """
    Gradient of the objective.
    """
    n = len(params) // 3
    xy = params[:2*n].reshape(n, 2)
    r = np.maximum(params[2*n:], 0.001)
    
    # Gradient of -sum(r)
    grad = np.zeros_like(params)
    grad[2*n:] = -1.0
    
    # Gradient of smooth overlap penalty
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)
    sum_r = r[:, None] + r[None, :]
    viol = sum_r - dist
    
    # softplus derivative: sigmoid(beta * viol)
    sig = 1.0 / (1.0 + np.exp(-beta * viol))
    
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    sig_pairs = sig[iu]
    
    # Gradient w.r.t. xy: for pair (i,j), d/dxy_i = -sig * (xy_i - xy_j) / dist
    # d/dxy_j = +sig * (xy_i - xy_j) / dist
    # d/dr_i = sig, d/dr_j = sig
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        sig_val = sig_pairs[idx]
        d = diff[i, j]  # xy_i - xy_j
        dist_val = dist[i, j]
        # Gradient w.r.t. xy_i: -sig * d / dist
        grad[2*i] += -10.0 * sig_val * d[0] / dist_val
        grad[2*i+1] += -10.0 * sig_val * d[1] / dist_val
        # Gradient w.r.t. xy_j: +sig * d / dist
        grad[2*j] += 10.0 * sig_val * d[0] / dist_val
        grad[2*j+1] += 10.0 * sig_val * d[1] / dist_val
        # Gradient w.r.t. r_i and r_j
        grad[2*n + i] += 10.0 * sig_val
        grad[2*n + j] += 10.0 * sig_val
    
    # Gradient of boundary penalty
    d_left = xy[:, 0] - r
    d_right = 1.0 - xy[:, 0] - r
    d_bottom = xy[:, 1] - r
    d_top = 1.0 - xy[:, 1] - r
    
    # softplus derivative
    sig_l = 1.0 / (1.0 + np.exp(-50.0 * d_left))
    sig_r = 1.0 / (1.0 + np.exp(-50.0 * d_right))
    sig_b = 1.0 / (1.0 + np.exp(-50.0 * d_bottom))
    sig_t = 1.0 / (1.0 + np.exp(-50.0 * d_top))
    
    for i in range(n):
        grad[2*i] += 10.0 * (-sig_l[i] + sig_r[i])
        grad[2*i+1] += 10.0 * (-sig_b[i] + sig_t[i])
        grad[2*n+i] += 10.0 * (sig_l[i] + sig_r[i] + sig_b[i] + sig_t[i])
    
    return grad

def _is_valid(xy, r, tol=1e-9):
    n = len(r)
    if np.any(r < -tol):
        return False
    if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
        return False
    if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
        return False
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
    return True

def _refine(params, beta=50.0, maxiter=200):
    """
    Use L-BFGS-B to refine the configuration.
    """
    n = len(params) // 3
    bounds = []
    for i in range(n):
        # x, y in [0, 1], r in [0.001, 0.5]
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.001, 0.5))
    
    res = minimize(
        _objective, params, jac=_objective_grad,
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    return res.x

def _powell_refine(params, beta=50.0, maxiter=100):
    """
    Use Powell's method as a derivative-free backup.
    """
    n = len(params) // 3
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.001, 0.5))
    
    res = minimize(
        _objective, params,
        method='Powell', bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-10}
    )
    return res.x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.RandomState(42)
    
    best_params = None
    best_score = -np.inf
    best_xy = None
    best_r = None
    
    # Initial configuration: random positions and small radii
    def _init_config():
        xy = rng.uniform(0.1, 0.9, size=(N, 2))
        r = rng.uniform(0.02, 0.1, size=N)
        return np.concatenate([xy.flatten(), r])
    
    # Simulated Annealing
    T_start = 1.0
    T_end = 0.001
    n_iter = 2000
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    params = _init_config()
    current_obj = _objective(params)
    
    # Pre-allocate for speed
    for it in range(n_iter):
        if time.time() > deadline - 30:
            break
        
        T = T_start * (cooling ** it)
        
        # Propose a move: perturb a random subset of parameters
        n_perturb = rng.randint(1, 5)
        indices = rng.choice(len(params), size=n_perturb, replace=False)
        new_params = params.copy()
        for idx in indices:
            # Determine if this is x, y, or r
            pos = idx % 3
            if pos < 2:  # x or y
                new_params[idx] += rng.normal(0, 0.05)
            else:  # radius
                new_params[idx] += rng.normal(0, 0.02)
        
        # Clip to bounds
        for i in range(N):
            new_params[2*i] = np.clip(new_params[2*i], 0.0, 1.0)
            new_params[2*i+1] = np.clip(new_params[2*i+1], 0.0, 1.0)
            new_params[2*N+i] = np.clip(new_params[2*N+i], 0.001, 0.5)
        
        new_obj = _objective(new_params)
        
        # Metropolis criterion
        if new_obj < current_obj:
            params = new_params
            current_obj = new_obj
        else:
            delta = new_obj - current_obj
            if delta < 0 or rng.random() < np.exp(-delta / T):
                params = new_params
                current_obj = new_obj
        
        # Periodic refinement with L-BFGS-B
        if it % 50 == 49:
            if time.time() > deadline - 60:
                break
            refined = _refine(params, beta=50.0, maxiter=100)
            refined_obj = _objective(refined)
            if refined_obj < current_obj:
                params = refined
                current_obj = refined_obj
        
        # Track best
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = np.maximum(params[2*n:], 0.001)
        if _is_valid(xy, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_xy = xy.copy()
                best_r = r.copy()
    
    # Final refinement with Powell
    if time.time() < deadline - 10:
        if best_params is not None:
            final = _powell_refine(best_params, beta=50.0, maxiter=50)
            n = N
            xy = final[:2*n].reshape(n, 2)
            r = np.maximum(final[2*n:], 0.001)
            if _is_valid(xy, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_xy = xy.copy()
                    best_r = r.copy()
    
    if best_xy is not None:
        return np.column_stack([best_xy, best_r])
    
    # Fallback: simple grid
    side = 6
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
        r[i] = 0.5 / side - 0.005
    return np.column_stack([xy, r])
