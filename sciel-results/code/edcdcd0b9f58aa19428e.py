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

N = 50

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
    
    # Handle numerical stability if sum_exp is 0 or inf
    if sum_exp == 0 or not np.isfinite(sum_exp):
        # Fallback to hard min gradient if numerical issues arise
        # This is a rare edge case, but good for robustness
        min_idx = np.argmin(vals)
        weights = np.zeros(n_vals)
        weights[min_idx] = 1.0
        soft_min_val = m
    else:
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
    # Where c is 0, sign is 0, but derivative of |x| at 0 is subgradient. 
    # For optimization, 0 is a safe choice or we can use a small epsilon.
    # Using sign(c) is standard for smooth regions.
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

def _generate_layered_initial(n_points, seed=42):
    """
    Generate points in a layered structure, which is often close to optimal packings.
    """
    rng = np.random.default_rng(seed)
    points = []
    
    # Try different layer configurations
    # For 50 points, maybe 4 layers of 12-13 points? Or 5 layers?
    # Let's try to distribute them somewhat evenly.
    
    # A simple heuristic: divide the cube into layers along Z axis.
    # Layer thickness ~ 1/num_layers.
    # Points in each layer arranged in a grid or hexagonal pattern.
    
    num_layers = 4
    points_per_layer = [13, 13, 12, 12] # Total 50
    
    z_positions = []
    for i in range(num_layers):
        # Spread layers evenly in z, keeping some margin from walls
        z = -0.4 + i * (0.8 / (num_layers - 1))
        z_positions.append(z)
    
    for idx, count in enumerate(points_per_layer):
        z = z_positions[idx]
        # Arrange points in this layer
        # Use a grid in XY plane
        # Approximate grid size
        n_grid = int(np.ceil(np.sqrt(count)))
        # Adjust to get exactly count
        # Simple approach: place in a grid, then shuffle and take first 'count'
        # Or use a hexagonal packing approximation
        
        # Let's use a simple grid for initialization
        xs = np.linspace(-0.4, 0.4, n_grid)
        ys = np.linspace(-0.4, 0.4, n_grid)
        
        layer_points = []
        for x in xs:
            for y in ys:
                layer_points.append([x, y, z])
                if len(layer_points) >= count:
                    break
            if len(layer_points) >= count:
                break
        
        # If we didn't get enough, add random ones in the layer
        while len(layer_points) < count:
            x = rng.uniform(-0.4, 0.4)
            y = rng.uniform(-0.4, 0.4)
            layer_points.append([x, y, z])
            
        points.extend(layer_points[:count])
        
    return np.array(points[:n_points])

def _generate_grid_initial(n_points, seed=42):
    """
    Generate points on a dense grid and select the best n_points.
    """
    rng = np.random.default_rng(seed)
    n_div = 6
    points = []
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div - 0.5, j / n_div - 0.5, k / n_div - 0.5])
    
    points = np.array(points)
    
    # Shuffle and pick
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points = points[indices]
    
    selected = []
    for p in points:
        if len(selected) >= n_points:
            break
        # Check distance to already selected
        if not selected:
            selected.append(p)
            continue
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.1: # Minimum separation
            selected.append(p)
            
    # If not enough, add random
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
    
    # Continuation schedule
    # Start with low beta (smooth), increase to high beta (sharp)
    betas = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        def obj_grad(x):
            f, g = _soft_min_objective_and_grad(x, beta)
            return f, g
        
        try:
            res = minimize(obj_grad, best_c.ravel(), method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8,
                                    'maxls': 50})
            
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.4999, 0.4999)
            
            new_val = _compute_radius(new_c)
            
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
                
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        for beta_polish in [10000, 20000]:
            if time.time() > deadline - 2:
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

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Layered structure
    c_layered = _generate_layered_initial(N, seed=1)
    initial_configs.append(c_layered)
    
    # 2. Grid structure
    c_grid = _generate_grid_initial(N, seed=2)
    initial_configs.append(c_grid)
    
    # 3. Random with repulsion
    c_rand = rng.uniform(-0.4, 0.4, size=(N, 3))
    # Simple repulsion step
    for _ in range(50):
        if time.time() > deadline - 100: break
        for i in range(N):
            for j in range(i+1, N):
                diff = c_rand[i] - c_rand[j]
                dist = np.linalg.norm(diff)
                if dist < 0.15:
                    force = diff / dist * 0.01
                    c_rand[i] += force
                    c_rand[j] -= force
        c_rand = np.clip(c_rand, -0.45, 0.45)
    initial_configs.append(c_rand)
    
    # 4. Perturbed layered
    c_pert = c_layered + rng.normal(0, 0.02, size=(N, 3))
    c_pert = np.clip(c_pert, -0.45, 0.45)
    initial_configs.append(c_pert)
    
    # 5. Another random seed
    c_rand2 = rng.uniform(-0.4, 0.4, size=(N, 3))
    initial_configs.append(c_rand2)
    
    best_c = initial_configs[0]
    best_val = _compute_radius(best_c)
    
    for i, c_init in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        c_opt, val = _optimize_from_initial(c_init, deadline)
        
        if val > best_val:
            best_val = val
            best_c = c_opt.copy()
            
    return best_c
# EVOLVE-BLOCK-END
