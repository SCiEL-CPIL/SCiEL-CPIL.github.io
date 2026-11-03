"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

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
    FCC has higher packing density than BCC.
    """
    n_div = 4
    points = []
    
    # FCC: corners + face centers
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])
    
    # Face centers
    for i in range(n_div):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([(i + 0.5) / n_div, j / n_div, k / n_div])
    for i in range(n_div + 1):
        for j in range(n_div):
            for k in range(n_div + 1):
                points.append([i / n_div, (j + 0.5) / n_div, k / n_div])
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div):
                points.append([i / n_div, j / n_div, (k + 0.5) / n_div])
    
    points = np.array(points)
    points = points - 0.5
    
    # Keep only points within the cube
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
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
        if np.min(dists) > 0.09:
            selected.append(p)
    
    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    
    return np.array(selected[:n_points])

def _local_polish(c, deadline):
    """
    Additional local polish using very high beta values and coordinate-wise refinement.
    """
    best_c = c.copy()
    best_val = _compute_radius(best_c)
    
    if time.time() > deadline - 5:
        return best_c, best_val
    
    # Try a few more optimization passes with different betas
    for beta in [20000, 50000, 100000]:
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
    
    # Coordinate-wise refinement: try small perturbations in each coordinate
    if time.time() < deadline - 1:
        step_sizes = [1e-5, 5e-6, 1e-6, 5e-7]
        for step in step_sizes:
            if time.time() > deadline - 0.5:
                break
            improved = True
            while improved and time.time() < deadline - 0.5:
                improved = False
                for i in range(N):
                    for k in range(3):
                        for sign in [-1, 1]:
                            c_trial = best_c.copy()
                            c_trial[i, k] += sign * step
                            c_trial[i, k] = np.clip(c_trial[i, k], -0.499999, 0.499999)
                            val_trial = _compute_radius(c_trial)
                            if val_trial > best_val + 1e-12:
                                best_val = val_trial
                                best_c = c_trial
                                improved = True
    
    return best_c, best_val

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
        if time.time() > deadline - 25:
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
    if time.time() < deadline - 15:
        for beta_polish in [5000, 10000]:
            if time.time() > deadline - 10:
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
    
    # Additional local polish
    if time.time() < deadline - 8:
        best_c, best_val = _local_polish(best_c, deadline)
    
    return best_c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
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
    
    # FCC variants
    for seed in [5, 6]:
        c_variant = c_fcc.copy()
        noise = rng.normal(0, 0.012, size=(N, 3))
        c_variant = c_variant + noise
        c_variant = np.clip(c_variant, -0.45, 0.45)
        initial_configs.append(c_variant)
    
    # Grid-based initialization
    grid_size = 10
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    indices = np.arange(len(grid))
    rng.shuffle(indices)
    grid_shuffled = grid[indices]
    
    selected = [grid_shuffled[0]]
    for i in range(1, len(grid_shuffled)):
        if len(selected) >= N:
            break
        p = grid_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.1:
            selected.append(p)
    
    while len(selected) < N:
        p = grid_shuffled[rng.integers(len(grid_shuffled))]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if len(dists) == 0 or np.min(dists) > 0.05:
            selected.append(p)
        else:
            p = rng.uniform(-0.45, 0.45, size=3)
            selected.append(p)
    
    initial_configs.append(np.array(selected[:N]))
    
    # Optimize from each initial configuration
    best_c = initial_configs[0].copy()
    best_val = _compute_radius(best_c)
    
    # Prioritize FCC and BCC (highest density packings)
    for idx, c_init in enumerate(initial_configs):
        if time.time() > deadline - 35:
            break
        
        try:
            c_opt, val = _optimize_from_initial(c_init, deadline)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Final polish on the best result
    if time.time() < deadline - 10:
        best_c, best_val = _local_polish(best_c, deadline)
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.5, 0.5)
    best_c = np.asarray(best_c, dtype=float)
    
    if best_c.shape != (N, 3) or not np.all(np.isfinite(best_c)):
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c

# EVOLVE-BLOCK-END
