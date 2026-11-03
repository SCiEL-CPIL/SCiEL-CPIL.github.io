import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 145

def _get_radius(pts):
    if len(pts) == 0:
        return 0.0
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)
    
    # Use cdist for speed in evaluation
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    return min(wall_dist, min_pair)

def _thomson_energy_grad(x, wall_k=500.0):
    """
    Minimize sum of 1/d^2 (Thomson potential) + wall penalties.
    """
    pts = x.reshape((N, 2))
    
    # Wall penalties
    wall_pen = np.zeros(N)
    grad_wall = np.zeros((N, 2))
    
    over_x = np.maximum(0, np.abs(pts[:, 0]) - 0.5)
    wall_pen += wall_k * over_x**2
    grad_wall[:, 0] += 2 * wall_k * over_x * np.sign(pts[:, 0])
    
    over_y = np.maximum(0, np.abs(pts[:, 1]) - 0.5)
    wall_pen += wall_k * over_y**2
    grad_wall[:, 1] += 2 * wall_k * over_y * np.sign(pts[:, 1])
    
    # Pair potential: 1/d^2
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    d2 = dx**2 + dy**2
    
    d2_safe = np.maximum(d2, 1e-12)
    
    # E = sum 1/d2
    pair_energy = np.sum(1.0 / d2_safe)
    
    # Grad E wrt x_i: -1/d2^2 * 2*dx = -2*dx/d2^2
    coeff = -2.0 / (d2_safe**2)
    
    grad_pair = np.zeros((N, 2))
    np.add.at(grad_pair[:, 0], i_idx, coeff * dx)
    np.add.at(grad_pair[:, 1], i_idx, coeff * dy)
    
    # Grad E wrt x_j: -1/d2^2 * 2*(-dx) = 2*dx/d2^2 = -coeff*dx
    np.add.at(grad_pair[:, 0], j_idx, -coeff * dx)
    np.add.at(grad_pair[:, 1], j_idx, -coeff * dy)
    
    total_energy = pair_energy + np.sum(wall_pen)
    total_grad = grad_pair + grad_wall
    
    return total_energy, total_grad.flatten()

def _soft_min(vals, beta):
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

def _objective_and_grad(x, beta):
    pts = x.reshape((N, 2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    C = np.empty(4*N + num_pairs)
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C[4*N:] = dists / 2.0
    
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
        
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    weights = exp_shifted / sum_exp
    
    grad_pts = np.zeros((N, 2))
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]
    
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
    
    pair_weights = weights[4*N:]
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = -grad_pts / beta
    return obj_val, grad_pts.flatten()

def _hex_init(spacing, jitter=0.01):
    points = []
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5
    row_idx = 0
    while y <= 0.5:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + shift
        while x <= 0.5:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    if len(pts) < N:
        return np.random.uniform(-0.45, 0.45, (N, 2))
        
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    # Add jitter
    pts += np.random.normal(0, jitter, (N, 2))
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    
    return pts

def _run_bfgs(pts, time_limit):
    t_start = time.time()
    x0 = pts.flatten()
    try:
        res = minimize(
            _thomson_energy_grad, 
            x0, 
            jac=True, 
            method='BFGS', 
            options={'maxiter': 1000, 'gtol': 1e-6}
        )
        x_opt = res.x
    except:
        x_opt = x0
        
    pts_opt = x_opt.reshape((N, 2))
    pts_opt[:, 0] = np.clip(pts_opt[:, 0], -0.5, 0.5)
    pts_opt[:, 1] = np.clip(pts_opt[:, 1], -0.5, 0.5)
    return pts_opt

def _polish_soft_min(pts, time_limit):
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    betas = [10, 50, 200]
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                args=(beta,),
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
        except:
            pass
    return best_r, best_x.reshape((N, 2))

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    spacings = [0.07, 0.075, 0.08]
    
    for spacing in spacings:
        elapsed = time.time() - t0
        if elapsed > _TIME_LIMIT - 30:
            break
            
        pts_init = _hex_init(spacing)
        
        # Phase 1: BFGS with Thomson Potential
        pts_bfgs = _run_bfgs(pts_init, _TIME_LIMIT - elapsed - 20)
        r_bfgs = _get_radius(pts_bfgs)
        
        if r_bfgs > best_r:
            best_r = r_bfgs
            best_pts = pts_bfgs
            
    # Phase 2: Polish with Soft-Min L-BFGS-B
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - elapsed - 5
        if remaining > 5:
            r_polish, pts_polish = _polish_soft_min(best_pts, remaining)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
                
    if best_pts is None:
        best_pts = np.random.uniform(-0.45, 0.45, (N, 2))
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
