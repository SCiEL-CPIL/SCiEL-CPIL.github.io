"""
Sphere packing: 100 equal spheres in a unit cube [-0.5, 0.5]^3.
Uses soft-min objective with L-BFGS-B, simulated annealing for local minima escape,
and final polishing.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
CUBE_LIMIT = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(CUBE_LIMIT - np.abs(centers)))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    min_wall_dist = np.min(CUBE_LIMIT - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _softmin_objective_and_grad(x, beta):
    """
    Objective: minimize -soft_min of distances (pairwise and wall).
    x: flattened centers (N*3,)
    beta: inverse temperature (larger beta = sharper min)
    Returns (obj, grad) for minimization of -soft_min.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances squared
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Wall distances
    wall_dists = CUBE_LIMIT - np.abs(centers)
    wall_sq_dists = wall_dists**2
    
    # Soft-min on squared pairwise distances
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        shifted = sq_dists - max_sq
        exp_terms = np.exp(-beta * shifted)
        exp_terms[~finite_mask] = 0.0
        sum_exp = np.sum(exp_terms)
        
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_min_sq = max_sq - (1.0 / beta) * log_sum
        else:
            soft_min_sq = max_sq
    else:
        soft_min_sq = 0.0
        
    # Wall part
    max_wall_sq = np.max(wall_sq_dists)
    shifted_wall = wall_sq_dists - max_wall_sq
    exp_wall = np.exp(-beta * shifted_wall)
    sum_wall = np.sum(exp_wall)
    
    if sum_wall > 0:
        log_sum_wall = np.log(sum_wall)
        soft_min_wall_sq = max_wall_sq - (1.0 / beta) * log_sum_wall
    else:
        soft_min_wall_sq = max_wall_sq
        
    # Combine pair and wall soft-mins using soft-min
    vals = np.array([soft_min_sq, soft_min_wall_sq])
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    sum_vals = np.sum(exp_vals)
    
    if sum_vals > 0:
        log_sum_vals = np.log(sum_vals)
        total_soft_min_sq = max_val - (1.0 / beta) * log_sum_vals
    else:
        total_soft_min_sq = max_val
        
    total_soft_min_sq = max(total_soft_min_sq, 0.0)
    obj_val = -np.sqrt(total_soft_min_sq)
    
    # Gradient: d(-sqrt(s))/dx = -s'/(2*sqrt(s))
    grad = np.zeros_like(centers)
    
    if total_soft_min_sq > 1e-12:
        # Gradient of soft_min_sq with respect to centers
        grad_sq = np.zeros_like(centers)
        
        if np.any(finite_mask) and sum_exp > 0:
            weights = exp_terms / sum_exp
            # d(soft_min_sq)/d(c_i) = sum_j weights[i,j] * 2*(c_i - c_j)
            # But we need to account for the soft-min formula:
            # soft_min_sq = max_sq - (1/beta)*log(sum(exp(-beta*(sq - max_sq))))
            # d(soft_min_sq)/d(c_i) = (1/beta) * sum_j exp(-beta*(sq_ij - max_sq)) / sum_exp * (-beta) * d(sq_ij)/d(c_i)
            # = -sum_j weights_ij * d(sq_ij)/d(c_i)
            # d(sq_ij)/d(c_i) = 2*(c_i - c_j)
            
            sum_w = np.sum(weights, axis=1)  # sum over j for each i
            weighted_sum_x = weights @ centers  # sum_j weights_ij * c_j
            
            # d(soft_min_sq)/d(c_i) = -sum_j weights_ij * 2*(c_i - c_j)
            # = -2*(sum_j weights_ij * c_i - sum_j weights_ij * c_j)
            # = -2*(sum_w_i * c_i - weighted_sum_x_i)
            grad_pair_sq = -2.0 * (sum_w[:, np.newaxis] * centers - weighted_sum_x)
            grad_sq += grad_pair_sq
        
        # Wall gradient
        if sum_wall > 0:
            weights_wall = exp_wall / sum_wall
            # d(soft_min_wall_sq)/d(c_i) = -sum_k weights_ik * d(wall_sq_ik)/d(c_ik)
            # d(wall_sq_ik)/d(c_ik) = -2*wall_dist_ik * sign(c_ik)
            signs = np.sign(centers)
            signs[centers == 0] = 1.0
            d_wall_sq_dc = -2.0 * wall_dists * signs
            
            # d(soft_min_wall_sq)/d(c_i) = -weights_wall_i * d_wall_sq_dc_i
            grad_wall_sq = -weights_wall[:, np.newaxis] * d_wall_sq_dc
            grad_sq += grad_wall_sq
        
        # Combine: d(total_soft_min_sq)/dx = w_A * grad_pair_sq + w_B * grad_wall_sq
        # But we already combined into grad_sq above via the soft-min combination
        # Actually, we need to be more careful. Let me recompute.
        
        # The total soft min is:
        # total = max_val - (1/beta)*log(exp(-beta*(sq_pair - max_val)) + exp(-beta*(sq_wall - max_val)))
        # = max_val - (1/beta)*log(exp_vals[0] + exp_vals[1])
        
        # d(total)/dx = (1/beta) * (exp_vals[0]*d(sq_pair)/dx + exp_vals[1]*d(sq_wall)/dx) / (exp_vals[0]+exp_vals[1])
        # = w_A * d(sq_pair)/dx + w_B * d(sq_wall)/dx
        
        w_A = exp_vals[0] / sum_vals
        w_B = exp_vals[1] / sum_vals
        
        # Recompute gradients properly
        grad_sq = np.zeros_like(centers)
        
        if np.any(finite_mask) and sum_exp > 0:
            weights = exp_terms / sum_exp
            sum_w = np.sum(weights, axis=1)
            weighted_sum_x = weights @ centers
            grad_pair_sq = -2.0 * (sum_w[:, np.newaxis] * centers - weighted_sum_x)
            grad_sq += w_A * grad_pair_sq
        
        if sum_wall > 0:
            weights_wall = exp_wall / sum_wall
            signs = np.sign(centers)
            signs[centers == 0] = 1.0
            d_wall_sq_dc = -2.0 * wall_dists * signs
            grad_wall_sq = -weights_wall[:, np.newaxis] * d_wall_sq_dc
            grad_sq += w_B * grad_wall_sq
        
        # d(obj)/dx = d(-sqrt(total))/dx = -grad_sq / (2*sqrt(total))
        grad = -grad_sq / (2.0 * np.sqrt(total_soft_min_sq))
    
    return obj_val, grad

def _optimize_packing(init_centers, time_limit=10.0):
    """
    Optimize packing from initial centers using soft-min L-BFGS-B.
    """
    start_time = time.time()
    x0 = init_centers.flatten()
    
    # Beta schedule: start small (smooth), end large (sharp)
    betas = [5, 20, 50, 100, 300, 1000, 3000]
    
    best_x = x0
    best_val = _compute_radius(init_centers)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        def obj_grad(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _softmin_objective_and_grad(x, beta)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
            
        try:
            result = minimize(
                obj_grad, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            x0 = result.x
            r = _compute_radius(x0.reshape(N, 3))
            if r > best_val:
                best_val = r
                best_x = x0
        except Exception:
            pass
            
    return best_x.reshape(N, 3)

def _simulated_annealing(centers, time_limit=15.0):
    """
    Simulated annealing to escape local minima.
    Uses multi-sphere moves for better exploration.
    """
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Temperature schedule
    T0 = 0.08
    Tend = 0.0005
    n_steps = 5000
    cooling_rate = (Tend / T0) ** (1.0 / n_steps)
    
    rng = np.random.RandomState(42)
    T = T0
    
    # Precompute current radius
    current_radius = best_radius
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Decide how many spheres to move
        if step % 5 == 0:
            n_move = 1
        elif step % 5 == 1:
            n_move = 2
        else:
            n_move = 3
            
        # Pick random spheres to move
        idx = rng.choice(N, size=n_move, replace=False)
        
        # Random displacements
        disp = rng.normal(0, 0.03, size=(n_move, 3))
        new_centers = centers.copy()
        for k, i in enumerate(idx):
            new_centers[i] += disp[k]
        
        # Clip to cube
        new_centers = np.clip(new_centers, -0.49999, 0.49999)
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        delta = new_radius - current_radius
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        
        # Cool down
        T *= cooling_rate
        
    return best_centers

def _epigraph_polish(centers, time_limit=10.0):
    """
    High-precision polish using epigraph formulation.
    Maximize t subject to ||c_i - c_j||^2 >= 4*t^2 and wall constraints.
    Uses a penalty-based approach with increasing penalty weights.
    """
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    def _epigraph_obj_and_grad(x, t, mu):
        """
        Minimize -t + mu * sum(max(0, 4*t^2 - ||c_i-c_j||^2)^2 + max(0, t - wall_dist)^2)
        This is a smooth approximation of the epigraph constraints.
        """
        c = x.reshape(N, 3)
        
        # Pairwise constraints: ||c_i - c_j||^2 >= 4*t^2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        # Violation: max(0, 4*t^2 - sq_dist)^2
        violations = np.maximum(0, 4*t*t - sq_dists)
        pair_penalty = np.sum(violations**2)
        
        # Wall constraints: wall_dist >= t
        wall_dists = CUBE_LIMIT - np.abs(c)
        wall_violations = np.maximum(0, t - wall_dists)
        wall_penalty = np.sum(wall_violations**2)
        
        obj = -t + mu * (pair_penalty + wall_penalty)
        
        # Gradient
        grad = np.zeros_like(c)
        
        # Pair gradient
        finite_mask = np.isfinite(sq_dists)
        if np.any(finite_mask):
            active = violations > 0
            if np.any(active):
                # d(pair_penalty)/d(c_i) = sum_j 2*viol_ij * (-2*(c_i - c_j))
                # = -4 * sum_j viol_ij * (c_i - c_j) for active pairs
                grad_pair = np.zeros_like(c)
                for i in range(N):
                    for j in range(N):
                        if i != j and active[i, j]:
                            grad_pair[i] += -4.0 * violations[i, j] * (c[i] - c[j])
                grad += mu * grad_pair
        
        # Wall gradient
        active_wall = wall_violations > 0
        if np.any(active_wall):
            signs = np.sign(c)
            signs[c == 0] = 1.0
            # d(wall_penalty)/d(c_ik) = 2*wall_viol_i * (-sign(c_ik)) for active
            grad_wall = np.zeros_like(c)
            for i in range(N):
                for k in range(3):
                    if active_wall[i, k]:
                        grad_wall[i, k] = 2.0 * wall_violations[i, k] * (-signs[i, k])
            grad += mu * grad_wall
        
        return obj, grad.flatten()
    
    # Penalty annealing
    mu_values = [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]
    
    t_init = _compute_radius(centers) * 0.9
    t = t_init
    
    for mu in mu_values:
        if time.time() - start_time > time_limit:
            break
        
        x0 = centers.flatten()
        t0 = t
        
        def obj_grad_combined(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _epigraph_obj_and_grad(x, t0, mu)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
        
        try:
            result = minimize(
                obj_grad_combined, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
            )
            centers = result.x.reshape(N, 3)
            r = _compute_radius(centers)
            if r > best_radius:
                best_radius = r
            t = min(r * 1.05, CUBE_LIMIT)
        except Exception:
            pass
    
    return centers

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _jitter_init(n, seed=None, jitter=0.01):
    """Generate jittered grid initial configuration."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def _random_init(n, seed=None):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializations
    inits = []
    
    # 1. Grid
    inits.append(_grid_init(N))
    
    # 2. Jittered grids
    for seed in range(8):
        inits.append(_jitter_init(N, seed=seed, jitter=0.015))
        
    # 3. Random inits
    for seed in range(8):
        inits.append(_random_init(N, seed=seed))
        
    # Optimize each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        try:
            time_remaining = deadline - time.time() - 15.0
            if time_remaining < 8.0:
                break
                
            # First optimize with gradient descent
            opt_time = min(8.0, time_remaining * 0.4)
            optimized = _optimize_packing(init, time_limit=opt_time)
            r_opt = _compute_radius(optimized)
            
            # Then apply simulated annealing
            if time.time() < deadline - 20.0:
                sa_time = min(8.0, deadline - time.time() - 12.0)
                sa_centers = _simulated_annealing(optimized, time_limit=sa_time)
                r_sa = _compute_radius(sa_centers)
                if r_sa > r_opt:
                    optimized = sa_centers
                    r_opt = r_sa
            
            # Epigraph polish
            if time.time() < deadline - 8.0:
                ep_time = min(6.0, deadline - time.time() - 5.0)
                ep_centers = _epigraph_polish(optimized, time_limit=ep_time)
                r_ep = _compute_radius(ep_centers)
                if r_ep > r_opt:
                    r_opt = r_ep
                    optimized = ep_centers
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
                
        except Exception:
            continue
            
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            ep_time = min(5.0, deadline - time.time() - 3.0)
            ep_centers = _epigraph_polish(best_centers, time_limit=ep_time)
            r_ep = _compute_radius(ep_centers)
            if r_ep > best_radius:
                best_radius = r_ep
                best_centers = ep_centers.copy()
        except Exception:
            pass
            
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
        
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
