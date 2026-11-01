"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 100
CUBE_HALF = 0.5

def _pairwise_distances(c):
    """Compute pairwise distances for N points."""
    # c has shape (N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist

def _compute_radius(c):
    """Compute the maximum common radius for configuration c."""
    # c has shape (N, 3)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    min_pair_dist = dist[iu].min()
    
    # Distance to walls
    wall_dists = np.minimum(np.abs(c), CUBE_HALF - np.abs(c)).min(axis=1)
    min_wall_dist = wall_dists.min()
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _objective_soft(c, beta):
    """
    Soft objective: maximize min constraint via log-sum-exp.
    We minimize -log(sum(exp(-beta * constraint_i)))
    Constraints:
      - pairwise: d_ij / 2
      - walls: CUBE_HALF - |c_ik|
    """
    c = c.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    pair_vals = dist[iu] / 2.0
    
    # Wall distances
    wall_vals = CUBE_HALF - np.abs(c).max(axis=1)
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Log-sum-exp for stability
    m = np.min(all_vals)
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (all_vals - m))))
    
    return -log_sum_exp

def _objective_gradient(c, beta):
    """
    Compute objective and gradient for soft min formulation.
    f(c) = -log(sum_j exp(-beta * g_j(c)))
    where g_j are the constraint functions (distances).
    """
    c = c.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-30)
    iu = np.triu_indices(N, k=1)
    pair_vals = dist[iu] / 2.0
    
    # Wall distances
    wall_vals = CUBE_HALF - np.abs(c).max(axis=1)
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Compute log-sum-exp
    m = np.min(all_vals)
    shifted = beta * (all_vals - m)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    f = -(m + np.log(sum_exp))
    
    # Gradient: df/dc = -beta * sum_j exp(beta*g_j) * grad_g_j / sum_j exp(beta*g_j)
    # = -beta * weighted_avg(grad_g_j)
    weights = exp_shifted / sum_exp
    
    grad = np.zeros_like(c)
    
    # Pairwise gradient
    # For pair (i,j), g_ij = d_ij / 2
    # d/d(c_i) d_ij = (c_i - c_j) / d_ij
    # d/d(c_j) d_ij = (c_j - c_i) / d_ij
    # So d/d(c_i) g_ij = (c_i - c_j) / (2 * d_ij)
    
    pair_weight_sum = weights[:len(iu)]
    
    # Accumulate gradients for pairs
    for idx in range(len(iu)):
        i, j = iu[0][idx], iu[1][idx]
        d = dist[i, j]
        if d < 1e-15:
            continue
        w = pair_weight_sum[idx]
        # d g_ij / d c_i = (c_i - c_j) / (2 * d)
        grad[i] += w * (c[i] - c[j]) / (2.0 * d)
        grad[j] += w * (c[j] - c[i]) / (2.0 * d)
    
    # Wall gradient
    # g_wall_i = CUBE_HALF - max_k |c_ik|
    # d/d(c_ik) g_wall_i = -sign(c_ik) if k is argmax
    wall_weight_sum = weights[len(iu):]
    for i in range(N):
        w = wall_weight_sum[i]
        abs_c = np.abs(c[i])
        k_max = np.argmax(abs_c)
        grad[i, k_max] -= w * np.sign(c[i, k_max])
    
    # Total gradient of f = -log(sum(exp(beta*g)))
    # df/dc = -beta * sum_j (exp(beta*g_j) / sum) * grad_g_j
    # = -beta * weighted_avg(grad_g_j)
    grad *= -beta
    
    return f, grad

def _initial_grid():
    """Create initial grid configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(float)

def _random_init(scale=0.45):
    """Random initialization inside the cube."""
    return np.random.uniform(-scale, scale, size=(N, 3))

def _sa_optimize(c0, deadline, max_iter=2000):
    """
    Simulated annealing with momentum to escape local minima.
    Uses the soft objective for smooth gradients.
    """
    c = c0.copy().reshape(N, 3)
    beta = 50.0  # Start with moderate beta
    
    # Track best solution
    best_c = c.copy()
    best_r = _compute_radius(c)
    best_obj = _objective_soft(c, beta)
    
    # SA parameters
    T = 0.01
    T_min = 1e-8
    alpha = 0.995  # Cooling rate
    step_scale = 0.05
    momentum = 0.5
    velocity = np.zeros_like(c)
    
    iter_count = 0
    while time.time() < deadline:
        iter_count += 1
        
        # Compute gradient
        obj, grad = _objective_gradient(c, beta)
        
        # Update with momentum
        velocity = momentum * velocity - step_scale * grad
        c_new = c + velocity
        
        # Clip to stay within cube (with small margin)
        margin = 1e-6
        c_new = np.clip(c_new, -CUBE_HALF + margin, CUBE_HALF - margin)
        
        # Compute new objective
        obj_new = _objective_soft(c_new, beta)
        
        # Accept or reject
        delta_obj = obj_new - obj
        if delta_obj < 0 or np.random.random() < np.exp(-delta_obj / T):
            c = c_new
            obj = obj_new
        else:
            # Revert velocity partially
            velocity *= 0.5
        
        # Update best
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Occasionally increase beta for sharper optimization
        if iter_count % 100 == 0:
            beta = min(beta * 1.2, 500.0)
        
        # Occasionally reset momentum
        if iter_count % 500 == 0:
            velocity *= 0.1
    
    return best_c, best_r

def _l_bfgs_polish(c0, deadline):
    """
    High-precision polish using L-BFGS-B with smooth objective.
    """
    c = c0.copy().reshape(N, 3)
    beta = 100.0
    
    def obj_func(x):
        return _objective_soft(x, beta)
    
    def obj_grad(x):
        _, g = _objective_gradient(x, beta)
        return g
    
    # Use L-BFGS-B with bounds
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    try:
        res = minimize(obj_func, c, method='L-BFGS-B', jac=obj_grad,
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
        c = res.x.reshape(N, 3)
    except Exception:
        pass
    
    r = _compute_radius(c)
    return c, r

def construct():
    """
    Main construct function: find optimal packing of 100 equal spheres in unit cube.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    best_c = None
    best_r = 0.0
    
    # Phase 1: Multiple random restarts with SA
    n_restarts = 0
    while time.time() < deadline - 20.0:
        n_restarts += 1
        
        if n_restarts == 1:
            c0 = _initial_grid()
        else:
            c0 = _random_init(scale=0.45)
        
        c, r = _sa_optimize(c0, deadline - 10.0)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # If we have good time left, do another restart
        if time.time() > deadline - 20.0:
            break
    
    # Phase 2: Polish the best solution with L-BFGS
    if best_c is not None and time.time() < deadline - 5.0:
        c, r = _l_bfgs_polish(best_c, deadline)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # Final validation: ensure all points are strictly inside
    if best_c is not None:
        # Clip to ensure validity
        best_c = np.clip(best_c, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
        # Final radius computation
        final_r = _compute_radius(best_c)
        if final_r > 0:
            return best_c
        else:
            # Fallback to grid
            return _initial_grid()
    else:
        return _initial_grid()
