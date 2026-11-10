"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the valid radius for a set of centers."""
    if centers is None or centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    min_dist_wall = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(min_dist_wall)
    
    # Distance between spheres
    if N < 2:
        return r_wall
        
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist_pair = np.min(dist)
    r_pair = min_dist_pair / 2.0
    
    return min(r_wall, r_pair)

def _soft_min_obj_and_grad(centers, beta):
    """
    Computes the negative soft-min of effective distances and its gradient.
    Effective distances:
    - Pairwise: d_ij
    - Wall: 2 * w_i (where w_i is distance to nearest wall)
    
    We want to maximize r = min(eff_dists) / 2.
    Equivalent to maximizing min(eff_dists).
    Equivalent to minimizing -min(eff_dists).
    
    Soft min: m(x) = -1/beta * log(sum(exp(-beta * x_i)))
    Grad: g_i = exp(-beta * x_i) / sum(exp(-beta * x_k))
    """
    centers = np.asarray(centers).reshape(-1, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances (distance to nearest wall is 0.5 - max(|x|, |y|, |z|))
    # Note: max is not differentiable at ties, but subgradient is fine. 
    # For soft-min, we treat each wall constraint separately? 
    # Actually, the constraint is c_k + r <= 0.5 and c_k - r >= -0.5.
    # So 2*r <= 0.5 - c_k AND 2*r <= 0.5 + c_k.
    # Let's define effective wall distances as:
    # w_i_plus = 0.5 - centers[i]
    # w_i_minus = 0.5 + centers[i]
    # The limiting wall distance for sphere i is min(w_i_plus, w_i_minus).
    # To be robust, we can include all 6*N wall constraints in the soft-min sum.
    
    # Vectorized wall constraints:
    # For each dimension k in {0,1,2}:
    # d_plus = 0.5 - centers[:, k]
    # d_minus = 0.5 + centers[:, k]
    # Effective dist = 2 * d (to match pair scale? No, pair dist is d_ij, wall is w.
    # r = min( d_ij/2, w_i ).
    # Let X be the set of constraints such that r <= X_k for all k.
    # For pair (i,j): r <= d_ij/2 => 2r <= d_ij.
    # For wall (i, k, sign): r <= 0.5 - sign*c_ik => 2r <= 1 - 2*sign*c_ik.
    
    # So we define a vector V of "2r upper bounds":
    # V_pairs = d_ij
    # V_walls = 1 - 2 * sign * c_ik
    
    # We want to maximize min(V).
    
    # Build V
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # Wall constraints:
    # c is (N, 3)
    # walls_plus = 0.5 - c  (N, 3) -> 2r <= 1 - 2c
    # walls_minus = 0.5 + c (N, 3) -> 2r <= 1 + 2c
    # Let's just concatenate all 6*N wall values.
    wall_vals = np.concatenate([
        (1.0 - 2.0 * centers).flatten(), 
        (1.0 + 2.0 * centers).flatten()
    ])
    
    V = np.concatenate([pair_dists, wall_vals])
    
    # Soft min calculation
    # To avoid overflow in exp, subtract max
    m = np.max(V)
    # If m is very large, exp(-beta*(V-m)) is stable.
    # If V has very small values, exp(-beta*V) might underflow, but we subtract max.
    # exp(-beta * (V - m)) = exp(-beta*V + beta*m)
    
    # We want log(sum(exp(-beta * V)))
    # = beta*m + log(sum(exp(-beta*(V-m))))
    
    shifted = V - m
    # beta * shifted is <= 0.
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        # Fallback, shouldn't happen if m is max
        return 0.0, np.zeros_like(centers)
        
    log_sum_exp = beta * m + np.log(sum_exp)
    
    # Objective: -min(V) approx -1/beta * log_sum_exp
    # We want to MINIMIZE this objective.
    # So obj = -1/beta * log_sum_exp
    # Wait, soft_min = -1/beta * log(sum(exp(-beta*V)))
    # We want to maximize soft_min.
    # So we minimize -soft_min = 1/beta * log(sum(exp(-beta*V)))
    
    obj = (1.0 / beta) * log_sum_exp
    
    # Gradient of obj w.r.t centers
    # d(obj)/dV_k = (1/beta) * (1/sum_exp) * (-beta * exp(-beta*V_k))
    #             = -exp(-beta*V_k) / sum_exp
    # Let p_k = exp(-beta*V_k) / sum_exp. Then d(obj)/dV_k = -p_k.
    
    p = exp_terms / sum_exp
    
    # Now chain rule: d(obj)/dc = sum_k (d(obj)/dV_k) * (dV_k/dc)
    # d(obj)/dV_k = -p_k
    
    grad = np.zeros_like(centers)
    
    # Pairs:
    # V_pair(i,j) = ||c_i - c_j||
    # dV/dc_i = (c_i - c_j) / ||c_i - c_j||
    # dV/dc_j = (c_j - c_i) / ||c_i - c_j||
    
    # We have p for all pairs in `p[:len(pair_dists)]`
    p_pairs = p[:len(pair_dists)]
    
    # We need to accumulate gradients.
    # Looping over pairs in Python is slow for 4950 pairs.
    # Let's try to vectorize or use sparse accumulation.
    # For N=100, 4950 pairs. A python loop might take ~1-5ms. Acceptable.
    
    # Extract indices
    i_idx, j_idx = iu
    
    for k in range(len(pair_dists)):
        i = i_idx[k]
        j = j_idx[k]
        d = pair_dists[k]
        if d < 1e-8:
            continue
        # Weight is -p_pairs[k]
        w = -p_pairs[k]
        
        diff_vec = centers[i] - centers[j]
        dir_vec = diff_vec / d
        
        grad[i] += w * dir_vec
        grad[j] -= w * dir_vec
        
    # Walls:
    # V_wall(i, k, sign) = 1 - 2*sign*c_ik
    # dV/dc_ik = -2*sign
    # d(obj)/dc_ik = -p_wall * (-2*sign) = 2*sign*p_wall
    
    p_walls = p[len(pair_dists):]
    
    # p_walls corresponds to:
    # First N*3: (1 - 2*c) for all i, k. sign = +1? 
    # Wait, my construction:
    # wall_vals = concat(1-2c, 1+2c)
    # So first half is 1-2c (sign=+1 in formula 1-2*sign*c? No.
    # If constraint is c <= 0.5 - r -> 2r <= 1 - 2c. Here sign is effectively +1 for the term 2c.
    # Let's just use the derivative directly.
    
    # Part 1: 1 - 2*c
    # dV/dc = -2
    # d(obj)/dc = -p * (-2) = 2*p
    # Shape of p_walls[:N*3] is (N, 3) flattened.
    grad += 2.0 * p_walls[:N*3].reshape(N, 3)
    
    # Part 2: 1 + 2*c
    # dV/dc = 2
    # d(obj)/dc = -p * (2) = -2*p
    grad -= 2.0 * p_walls[N*3:].reshape(N, 3)
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy:
    # 1. Multiple random restarts.
    # 2. For each restart:
    #    a. Start with random or grid.
    #    b. Run L-BFGS-B with increasing beta (annealing) to sharpen the min.
    #    c. Evaluate hard radius.
    
    # Initializations
    inits = []
    
    # Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # Random
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(-0.45, 0.45, (N, 3)))
        
    # FCC Lattice subset (approx)
    # Generate FCC points in a box and take closest N to center?
    # Just use random for now to keep it simple, grid is good.
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        centers = init.copy()
        
        # Annealing schedule for beta
        # Start with small beta (smooth), increase to large beta (sharp)
        betas = [5.0, 20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            # Define objective and gradient wrapper
            def obj_grad(x):
                x = x.reshape(-1, 3)
                val, grad = _soft_min_obj_and_grad(x, beta)
                return val, grad.flatten()
            
            # Bounds: keep centers well inside to avoid numerical issues with walls
            # The wall constraints in soft-min handle the boundary, but L-BFGS-B
            # works better with explicit bounds if we want to enforce them strictly.
            # However, the soft-min already penalizes going out.
            # If we use bounds [-0.5, 0.5], the gradient might point outwards if beta is low.
            # Let's use bounds [-0.49, 0.49] to be safe? 
            # No, the soft-min objective will naturally keep them in if beta is high enough.
            # But for low beta, they might drift. 
            # Let's use bounds [-0.5, 0.5] and rely on the potential.
            # Actually, if a sphere goes out, dist to wall becomes negative.
            # 1 - 2c becomes > 1. 
            # Wait, if c > 0.5, 1 - 2c < 0. 
            # min(V) becomes negative. 
            # We maximize min(V). So negative V is bad.
            # So the gradient will push back.
            
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                res = minimize(
                    obj_grad, 
                    centers.flatten(), 
                    method='L-BFGS-B', 
                    jac=True, 
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                centers = res.x.reshape(N, 3)
            except Exception as e:
                # If optimization fails, keep current centers
                pass
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    # If we have time, try a few more random restarts with higher beta directly
    while time.time() < deadline - 10.0:
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        # Quick polish with high beta
        def obj_grad_high(x):
            x = x.reshape(-1, 3)
            val, grad = _soft_min_obj_and_grad(x, 200.0)
            return val, grad.flatten()
            
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(
                obj_grad_high, 
                centers.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            centers = res.x.reshape(N, 3)
        except:
            pass
            
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        best_centers = inits[0].copy()
        
    return best_centers

# EVOLVE-BLOCK-END
