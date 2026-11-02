"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice.
2. Global optimization using Log-Sum-Exp surrogate for max-min distance.
3. High-precision local polish using SLSQP active-set method and L-BFGS-B.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)

def _initialize_hexagonal():
    """Initialize with hexagonal pattern - optimized search over parameters."""
    best_c = None
    best_r = 0.0
    
    # Try different densities with finer granularity around optimal region
    for d in np.linspace(0.052, 0.075, 40):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _soft_min_energy(params, beta=200.0):
    """
    Minimize the negative log-sum-exp of distances.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    # Log-Sum-Exp trick for stability
    d_shifted = d_pairs - np.min(d_pairs)
    weights = np.exp(-beta * d_shifted)
    p = weights / np.sum(weights)
    
    E_pairs = np.log(np.sum(weights)) / beta
    
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    
    grad_x[i_idx] -= p * vec_x * inv_d
    grad_x[j_idx] += p * vec_x * inv_d
    
    grad_y[i_idx] -= p * vec_y * inv_d
    grad_y[j_idx] += p * vec_y * inv_d
    
    # Wall Penalty
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    eps = 1e-8
    d_left_safe = np.clip(d_left, eps, None)
    d_right_safe = np.clip(d_right, eps, None)
    d_down_safe = np.clip(d_down, eps, None)
    d_up_safe = np.clip(d_up, eps, None)
    
    E_wall = (1.0/d_left_safe**2 + 1.0/d_right_safe**2 + 1.0/d_down_safe**2 + 1.0/d_up_safe**2).sum()
    
    grad_x += -2.0 / d_left_safe**3 + 2.0 / d_right_safe**3
    grad_y += -2.0 / d_down_safe**3 + 2.0 / d_up_safe**3
    
    E_total = E_wall + E_pairs
    total_grad = np.concatenate([grad_x, grad_y])
    return E_total, total_grad

def _optimize_energy(c0, time_budget=20.0, beta=100.0):
    """Run L-BFGS-B with Log-Sum-Exp energy."""
    start = time.time()
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        def fun_and_grad(params):
            return _soft_min_energy(params, beta=beta)
            
        res = minimize(
            fun_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000, 'maxls': 50}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c0

def _get_active_constraints(c, r, tol=1e-5):
    """
    Identify active pairs and boundaries for a given radius r.
    """
    active_pairs = []
    active_bounds = []
    
    dists = pdist(c)
    iu = np.triu_indices(N, k=1)
    for k in range(len(dists)):
        if dists[k] <= 2*r + tol:
            active_pairs.append((iu[0][k], iu[1][k]))
            
    bdists = np.minimum(
        np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
        np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
    )
    for i in range(N):
        if bdists[i] <= r + tol:
            d_left = 0.5 - c[i, 0]
            d_right = 0.5 + c[i, 0]
            d_down = 0.5 - c[i, 1]
            d_up = 0.5 + c[i, 1]
            min_d = min(d_left, d_right, d_down, d_up)
            if d_left == min_d: active_bounds.append((i, 0))
            elif d_right == min_d: active_bounds.append((i, 1))
            elif d_down == min_d: active_bounds.append((i, 2))
            elif d_up == min_d: active_bounds.append((i, 3))
            
    return active_pairs, active_bounds

def _slsqp_polish(c, r_init, time_limit=30.0):
    """
    SLSQP polish using active constraints.
    """
    start_time = time.time()
    c = c.copy()
    r = r_init
    
    def objective_active(params):
        return -params[-1]
        
    def gradient_active(params):
        grad = np.zeros(2*N + 1)
        grad[-1] = -1.0
        return grad

    current_c = c
    current_r = r_init
    
    for iter_count in range(20):
        if time.time() - start_time > time_limit - 2:
            break
            
        tol = 1e-4
        active_pairs, active_bounds = _get_active_constraints(current_c, current_r, tol=tol)
        
        if not active_pairs and not active_bounds:
            current_r += 1e-4
            continue
            
        def make_pair_constraint(i, j):
            def fun(p):
                di = p[2*i] - p[2*j]
                dj = p[2*i+1] - p[2*j+1]
                d = np.sqrt(di*di + dj*dj)
                return d - p[-1]
            def jac(p):
                jv = np.zeros(2*N + 1)
                di = p[2*i] - p[2*j]
                dj = p[2*i+1] - p[2*j+1]
                d = np.sqrt(di*di + dj*dj)
                if d > 1e-12:
                    jv[2*i] = di/d
                    jv[2*i+1] = dj/d
                    jv[2*j] = -di/d
                    jv[2*j+1] = -dj/d
                jv[-1] = -1.0
                return jv
            return {'type': 'ineq', 'fun': fun, 'jac': jac}
            
        def make_bound_constraint(i, wall):
            def fun(p):
                if wall == 0: return 0.5 - p[2*i] - p[-1]
                if wall == 1: return 0.5 + p[2*i] - p[-1]
                if wall == 2: return 0.5 - p[2*i+1] - p[-1]
                if wall == 3: return 0.5 + p[2*i+1] - p[-1]
            def jac(p):
                jv = np.zeros(2*N + 1)
                if wall == 0: jv[2*i] = -1.0
                if wall == 1: jv[2*i] = 1.0
                if wall == 2: jv[2*i+1] = -1.0
                if wall == 3: jv[2*i+1] = 1.0
                jv[-1] = -1.0
                return jv
            return {'type': 'ineq', 'fun': fun, 'jac': jac}
            
        cons = []
        for i, j in active_pairs:
            cons.append(make_pair_constraint(i, j))
        for i, w in active_bounds:
            cons.append(make_bound_constraint(i, w))
            
        x0 = np.concatenate([current_c.flatten(), [current_r]])
        bounds = [(-0.5, 0.5)] * (2*N) + [(current_r, current_r + 0.01)]
        
        try:
            res = minimize(
                objective_active,
                x0,
                method='SLSQP',
                jac=gradient_active,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if res.success:
                new_c = res.x[:2*N].reshape(N, 2)
                new_r = res.x[-1]
                verified_r = _compute_radius(new_c)
                
                if verified_r > current_r + 1e-8:
                    current_c = new_c
                    current_r = verified_r
            else:
                break
                
        except Exception:
            break
            
    return current_c, current_r

def _local_polish(c, time_limit=30.0):
    """
    Local polish combining SLSQP active-set and L-BFGS-B.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    if time.time() - start_time < time_limit * 0.5:
        c_slsqp, r_slsqp = _slsqp_polish(c, current_r, time_limit=time_limit * 0.5)
        if r_slsqp > current_r:
            c = c_slsqp
            current_r = r_slsqp
            
    if time.time() - start_time < time_limit - 2:
        def high_prec_energy(params, beta=500.0):
            c_ = params.reshape(N, 2)
            x = c_[:, 0]
            y = c_[:, 1]
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1.0)
            
            iu = np.triu_indices(N, k=1)
            d_pairs = d[iu]
            
            bdists = np.minimum(
                np.minimum(0.5 - x, 0.5 + x),
                np.minimum(0.5 - y, 0.5 + y)
            )
            
            d_shifted = d_pairs - np.min(d_pairs)
            w_pairs = np.exp(-beta * d_shifted)
            E_pairs = np.log(np.sum(w_pairs)) / beta
            
            bd_shifted = bdists - np.min(bdists)
            w_bounds = np.exp(-beta * bd_shifted)
            E_bounds = np.log(np.sum(w_bounds)) / beta
            
            E = E_pairs + E_bounds
            
            p_pairs = w_pairs / np.sum(w_pairs)
            p_bounds = w_bounds / np.sum(w_bounds)
            
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)
            
            i_idx, j_idx = iu
            vec_x = x[i_idx] - x[j_idx]
            vec_y = y[i_idx] - y[j_idx]
            
            inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
            
            grad_x[i_idx] -= p_pairs * vec_x * inv_d
            grad_x[j_idx] += p_pairs * vec_x * inv_d
            grad_y[i_idx] -= p_pairs * vec_y * inv_d
            grad_y[j_idx] += p_pairs * vec_y * inv_d
            
            d_left = 0.5 - x
            d_right = 0.5 + x
            d_down = 0.5 - y
            d_up = 0.5 + y
            
            d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)
            d_min = np.min(d_all, axis=1, keepdims=True)
            wall_weights = np.exp(-beta * (d_all - d_min))
            wall_weights /= wall_weights.sum(axis=1, keepdims=True)
            
            grad_x += p_bounds * (-wall_weights[:, 0] + wall_weights[:, 1])
            grad_y += p_bounds * (-wall_weights[:, 2] + wall_weights[:, 3])
            
            total_grad = np.concatenate([grad_x, grad_y])
            return E, total_grad
            
        try:
            res = minimize(
                lambda p: high_prec_energy(p, beta=500.0),
                c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.49999, 0.49999)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 1000}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
            r_opt = _compute_radius(c_opt)
            if r_opt > current_r:
                c = c_opt
                current_r = r_opt
        except Exception:
            pass
            
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    for beta in [40.0, 80.0, 120.0, 180.0]:
        if time.time() > deadline - 90:
            break
        c_opt = _optimize_energy(best_c, time_budget=18.0, beta=beta)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    remaining = deadline - time.time() - 5
    if remaining > 15:
        c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 80.0))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    remaining = deadline - time.time() - 3
    if remaining > 15:
        c_polished2, r_polished2 = _local_polish(best_c, time_limit=min(remaining, 40.0))
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2.copy()
    
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
