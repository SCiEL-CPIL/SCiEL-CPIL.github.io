import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(c):
    """
    Compute the exact maximum radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    
    # Wall clearance
    wall = 0.5 - np.max(np.abs(c))
    
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """
    Compute the soft-min objective and its gradient.
    Returns (f, g) where f is the negative soft-min and g is the gradient.
    """
    c = x.reshape(N, 3)
    
    # Wall clearances: 0.5 - |c_ik| for each coordinate
    wall = 0.5 - np.abs(c)  # shape (N, 3)
    wall_flat = wall.ravel()  # shape (3N,)
    
    # Pairwise distances / 2
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    i, j = np.triu_indices(N, k=1)
    d_half_upper = d_half[i, j]  # shape (N*(N-1)/2,)
    
    # Combine all constraints
    vals = np.concatenate([wall_flat, d_half_upper])
    n_vals = len(vals)
    
    # Soft-min via log-sum-exp
    m = np.min(vals)
    shifted = vals - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = np.log(sum_exp)
    soft_min_val = m - log_sum_exp / beta
    
    # Weights for each constraint in the soft-min
    weights = exp_vals / sum_exp  # shape (n_vals,)
    
    # Gradient w.r.t. x
    grad_x = np.zeros_like(c)  # shape (N, 3)
    
    # Wall contribution: dv/dx = -sign(c) for each coordinate
    wall_weights = weights[:3*N]  # shape (3N,)
    wall_weights_reshaped = wall_weights.reshape(N, 3)
    sign_c = np.sign(c)
    grad_x -= wall_weights_reshaped * sign_c
    
    # Pair contribution
    pair_weights = weights[3*N:]  # shape (N*(N-1)/2,)
    
    ci = c[i]  # shape (n_pairs, 3)
    cj = c[j]  # shape (n_pairs, 3)
    diff = ci - cj  # shape (n_pairs, 3)
    norms = np.linalg.norm(diff, axis=1, keepdims=True)  # shape (n_pairs, 1)
    norms = np.maximum(norms, 1e-15)
    
    # dv/d(ci) = diff / (2 * norm)
    grad_ci = diff / (2.0 * norms)  # shape (n_pairs, 3)
    grad_cj = -grad_ci  # shape (n_pairs, 3)
    
    weighted_grad_ci = pair_weights[:, np.newaxis] * grad_ci
    weighted_grad_cj = pair_weights[:, np.newaxis] * grad_cj
    
    np.add.at(grad_x, i, weighted_grad_ci)
    np.add.at(grad_x, j, weighted_grad_cj)
    
    # Objective is -soft_min
    f = -soft_min_val
    g = -grad_x.ravel()
    
    return f, g

def _generate_bcc_lattice(n_points):
    """
    Generate points on a BCC lattice scaled to fit in [-0.5, 0.5]^3.
    """
    n_div = 5
    points = []
    
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])
    
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    
    points = np.array(points)
    points = points - 0.5
    
    rng = np.random.default_rng(123)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points_shuffled = points[indices]
    
    selected = [points_shuffled[0]]
    for i in range(1, len(points_shuffled)):
        if len(selected) >= n_points:
            break
        p = points_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.08:
            selected.append(p)
    
    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    
    return np.array(selected[:n_points])

def _generate_fcc_lattice(n_points):
    """
    Generate points on an FCC lattice scaled to fit in [-0.5, 0.5]^3.
    """
    n_div = 4
    points = []
    
    # Corner and face centers
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])
    
    # Edge centers
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div):
                points.append([i / n_div, j / n_div, (k + 0.5) / n_div])
    
    for i in range(n_div + 1):
        for j in range(n_div):
            for k in range(n_div + 1):
                points.append([i / n_div, (j + 0.5) / n_div, k / n_div])
    
    for i in range(n_div):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([(i + 0.5) / n_div, j / n_div, k / n_div])
    
    # Body center
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    
    points = np.array(points)
    points = points - 0.5
    
    rng = np.random.default_rng(456)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points_shuffled = points[indices]
    
    selected = [points_shuffled[0]]
    for i in range(1, len(points_shuffled)):
        if len(selected) >= n_points:
            break
        p = points_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.08:
            selected.append(p)
    
    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    
    return np.array(selected[:n_points])

def _optimize_from_initial(c_init, deadline):
    """
    Run continuation optimization from an initial configuration.
    Returns (best_c, best_val)
    """
    best_c = c_init.copy()
    best_val = _compute_radius(best_c)
    
    # Continuation schedule with more stages
    betas = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        def obj_grad(x):
            f, g = _soft_min_objective_and_grad(x, beta)
            return f, g
        
        try:
            res = minimize(obj_grad, best_c.ravel(), method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8,
                                    'maxls': 50})
            
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.4995, 0.4995)
            
            new_val = _compute_radius(new_c)
            
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
                
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        for beta_polish in [5000, 10000]:
            if time.time() > deadline - 5:
                break
            
            def obj_grad_polish(x):
                f, g = _soft_min_objective_and_grad(x, beta_polish)
                return f, g
            
            try:
                res = minimize(obj_grad_polish, best_c.ravel(), method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                                        'maxls': 30})
                
                new_c = res.x.reshape(N, 3)
                new_c = np.clip(new_c, -0.49999, 0.49999)
                
                new_val = _compute_radius(new_c)
                
                if new_val > best_val:
                    best_val = new_val
                    best_c = new_c.copy()
                    
            except Exception:
                pass
    
    return best_c, best_val

def _local_polish(c, deadline):
    """
    Additional local polish using a different approach:
    Directly optimize the minimum distance using a different parameterization.
    """
    best_c = c.copy()
    best_val = _compute_radius(best_c)
    
    if time.time() > deadline - 5:
        return best_c, best_val
    
    # Try a few more optimization passes with different betas
    for beta in [20000, 50000]:
        if time.time() > deadline - 3:
            break
        
        def obj_grad(x):
            f, g = _soft_min_objective_and_grad(x, beta)
            return f, g
        
        try:
            res = minimize(obj_grad, best_c.ravel(), method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-11,
                                    'maxls': 20})
            
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.499999, 0.499999)
            
            new_val = _compute_radius(new_c)
            
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
                
        except Exception:
            pass
    
    return best_c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    c_bcc = _generate_bcc_lattice(N)
    c_fcc = _generate_fcc_lattice(N)
    
    initial_configs = [c_bcc.copy(), c_fcc.copy()]
    
    # Create variants with different perturbations
    for seed in [1, 2, 3, 4]:
        c_variant = c_bcc.copy()
        noise = rng.normal(0, 0.015, size=(N, 3))
        c_variant = c_variant + noise
        c_variant = np.clip(c_variant, -0.45, 0.45)
        initial_configs.append(c_variant)
        
        c_variant2 = c_fcc.copy()
        noise2 = rng.normal(0, 0.015, size=(N, 3))
        c_variant2 = c_variant2 + noise2
        c_variant2 = np.clip(c_variant2, -0.45, 0.45)
        initial_configs.append(c_variant2)
    
    # Also add a random initialization
    c_random = rng.uniform(-0.4, 0.4, size=(N, 3))
    initial_configs.append(c_random)
    
    # Optimize each initial configuration
    best_c = initial_configs[0]
    best_val = _compute_radius(best_c)
    
    for i, c_init in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
        
        c_opt, val = _optimize_from_initial(c_init, deadline)
        
        if val > best_val:
            best_val = val
            best_c = c_opt.copy()
        
        # Additional local polish for the best configuration
        if time.time() < deadline - 5:
            c_polished, val_polished = _local_polish(c_opt, deadline)
            if val_polished > best_val:
                best_val = val_polished
                best_c = c_polished.copy()
    
    return best_c
# EVOLVE-BLOCK-END
