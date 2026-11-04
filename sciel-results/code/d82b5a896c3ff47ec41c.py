import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 100
HALF = 0.5
TOTAL_TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Wall distances
    wall_dists = HALF - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    if min_wall_dist < 0:
        return 0.0
    
    # Pairwise distances
    dists = pdist(centers)
    min_pair_dist = np.min(dists)
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _soft_min_obj_grad(x, beta):
    """
    Minimize the log-sum-exp of negative distances to approximate maximizing the minimum distance.
    Objective: log( sum exp(-beta * d) )
    This is a smooth surrogate for -min(d). Minimizing this maximizes the soft-min distance.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    # Using pdist for efficiency, but we need gradients.
    # For N=100, N(N-1)/2 = 4950 pairs.
    # We can compute gradients in a loop or vectorized.
    # Vectorized gradient computation:
    # d_ij = ||c_i - c_j||
    # grad_x_i += -beta * exp(-beta * d_ij) / Z * (c_i - c_j) / d_ij
    
    # To avoid O(N^2) memory in a single array if possible, or just use it since N=100 is small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal and upper/lower triangle to avoid double counting and self-distance
    # We sum over all i<j.
    # The term in the sum is exp(-beta * d_ij).
    # Let's compute the sum Z_pair
    # To be efficient, we only need the upper triangle.
    triu_indices = np.triu_indices(N, k=1)
    pair_dists = dists[triu_indices]
    
    # Wall distances
    wall_dists = HALF - np.abs(centers)
    # We sum over all i and k (3 coords per sphere)
    # Total terms: 4950 pairs + 300 wall terms = 5250 terms.
    
    # Calculate Z = sum(exp(-beta * d))
    # For numerical stability, we can subtract max exponent, but since we minimize log(Z),
    # and d > 0, exp(-beta*d) is small.
    # Actually, if d is small, exp is close to 1.
    
    # Let's combine all distances into one array for the log-sum-exp
    all_dists = np.concatenate([pair_dists, wall_dists.flatten()])
    
    # Clip distances to avoid log(0) or issues if centers coincide
    all_dists = np.maximum(all_dists, 1e-10)
    
    exponents = -beta * all_dists
    # Numerical stability for logsumexp
    max_exp = np.max(exponents)
    # If max_exp is very negative, exp is 0. But d is bounded by 1.5, beta ~ 10-100.
    # exp(-100 * 1.5) is 0. exp(-100 * 0.1) is 0.00004.
    # It's safer to do:
    # Z = exp(max_exp) * sum(exp(exponents - max_exp))
    # log(Z) = max_exp + log(sum(exp(exponents - max_exp)))
    
    shifted = exponents - max_exp
    sum_exp = np.sum(np.exp(shifted))
    log_Z = max_exp + np.log(sum_exp)
    
    # Gradient
    # d log(Z) / d x = (1/Z) * sum( d exp(-beta d) / d x )
    # = (1/Z) * sum( exp(-beta d) * (-beta) * d d / d x )
    # = -beta/Z * sum( exp(-beta d) * d d / d x )
    
    # We need the weights w_k = exp(-beta * d_k) / Z
    # w_k = exp(shifted_k) / sum_exp
    
    weights = np.exp(shifted) / sum_exp
    
    grad = np.zeros_like(centers)
    
    # Pairwise part
    # For each pair (i, j), dist = d_ij
    # grad[i] += -beta * w_pair * (c_i - c_j) / d_ij
    # grad[j] += -beta * w_pair * (c_j - c_i) / d_ij
    
    # We have pair_dists and weights for pairs (first 4950 weights)
    w_pairs = weights[:len(pair_dists)]
    
    # Vectorized update for pairs
    # diff is (N, N, 3). We need to add contributions.
    # This might be slow in Python loop. Let's try to vectorize or use a loop if N is small enough.
    # N=100, 4950 pairs. A Python loop might be slow.
    # Let's use the full matrix approach for gradient accumulation.
    
    # grad_from_pairs[i] = sum_j w_ij * (c_i - c_j) / d_ij
    # We can compute a matrix of coefficients.
    
    # Create a matrix W (N, N) where W[i, j] = w_pairs for pair (i, j) if i<j, 0 otherwise.
    # Actually, we need to handle the division by d.
    
    # Let's stick to a loop for clarity and correctness, as 5000 iterations is fast in NumPy if vectorized, but loop is safer for index mapping.
    # However, 4950 iterations in Python is ~1ms. This is called many times.
    # Better: Vectorized using the diff matrix.
    
    # diff[i, j] = c_i - c_j
    # dists[i, j] = d_ij
    # We need sum_j w_ij * diff[i, j] / dists[i, j]
    
    # Construct full weight matrix for pairs (symmetric)
    W_full = np.zeros((N, N))
    W_full[triu_indices] = w_pairs
    W_full[triu_indices[1], triu_indices[0]] = w_pairs # Symmetric
    
    # Coeff matrix C[i, j] = W_full[i, j] / dists[i, j]
    # Avoid division by zero on diagonal
    dists_safe = np.where(dists > 1e-10, dists, 1.0)
    C = W_full / dists_safe
    np.fill_diagonal(C, 0.0)
    
    # grad_pairs[i] = -beta * sum_j C[i, j] * (c_i - c_j)
    # = -beta * ( sum_j C[i, j] * c_i - sum_j C[i, j] * c_j )
    # = -beta * ( c_i * sum_j C[i, j] - (C @ c)_i )
    
    row_sums = C.sum(axis=1)
    C_c = C @ centers # (N, 3)
    
    grad_pairs = -beta * (centers * row_sums[:, np.newaxis] - C_c)
    grad += grad_pairs
    
    # Wall part
    # w_walls = weights[len(pair_dists):]
    # wall_dists is (N, 3)
    # grad_wall[i, k] = -beta * w_wall_ik * sign(c_ik)
    
    w_walls = weights[len(pair_dists):].reshape(N, 3)
    signs = np.sign(centers)
    # If center is 0, sign is 0, but gradient of |x| is undefined. 
    # In practice, if c is very close to 0, the force should be towards 0? 
    # d_wall = 0.5 - |c|. d(d_wall)/dc = -sign(c).
    # So grad energy = -beta * w * (-sign(c)) = beta * w * sign(c).
    # Wait, obj = log(sum exp(-beta d)).
    # d obj / d c = (1/Z) * exp(-beta d) * (-beta) * d d / d c
    # d d_wall / d c = -sign(c)
    # So grad = -beta/Z * exp(-beta d) * (-sign(c)) = beta/Z * exp(-beta d) * sign(c)
    
    grad_walls = beta * w_walls * signs
    
    grad += grad_walls
    
    return log_Z, grad.flatten()

def _minimize_soft_min(centers_init, beta, time_limit=5.0):
    """Minimize the soft-min objective."""
    x0 = centers_init.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        val, _ = _soft_min_obj_grad(x, beta)
        return val
        
    def grad(x):
        _, g = _soft_min_obj_grad(x, beta)
        return g
        
    start_t = time.time()
    stop_flag = [False]
    
    def callback(xk):
        if time.time() > start_t + time_limit:
            stop_flag[0] = True
        return stop_flag[0]
        
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=bounds, options={
                'maxiter': 200,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'callback': callback,
                'disp': False
            }
        )
        best_x = res.x
    except Exception:
        best_x = x0
        
    best_centers = best_x.reshape(N, 3)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

def _riesz_energy_and_grad(centers, s=6.0):
    """Compute Riesz s-energy and gradient for initial spreading."""
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = centers[i] - centers[j]
            d2 = np.dot(d_vec, d_vec)
            if d2 < 1e-20:
                d2 = 1e-20
            d = np.sqrt(d2)
            energy += d ** (-s)
            coeff = -s * d ** (-s - 2)
            grad[i] += coeff * d_vec
            grad[j] -= coeff * d_vec
            
    # Wall repulsion
    for i in range(n):
        for k in range(3):
            c = centers[i, k]
            d_wall = 0.5 - abs(c)
            if d_wall < 1e-10:
                d_wall = 1e-10
            energy += d_wall ** (-s)
            sign_c = np.sign(c)
            if sign_c == 0:
                sign_c = 1.0
            grad[i, k] += (-s) * d_wall ** (-s - 1) * sign_c
            
    return energy, grad

def _minimize_riesz(centers_init, s=6.0, time_limit=5.0):
    """Minimize Riesz energy to spread spheres."""
    x0 = centers_init.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        c = x.reshape(N, 3)
        e, _ = _riesz_energy_and_grad(c, s)
        return e
        
    def grad(x):
        c = x.reshape(N, 3)
        _, g = _riesz_energy_and_grad(c, s)
        return g.flatten()
        
    start_t = time.time()
    stop_flag = [False]
    
    def callback(xk):
        if time.time() > start_t + time_limit:
            stop_flag[0] = True
        return stop_flag[0]
        
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=bounds, options={
                'maxiter': 200,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'callback': callback,
                'disp': False
            }
        )
        best_x = res.x
    except Exception:
        best_x = x0
        
    best_centers = best_x.reshape(N, 3)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

def _generate_initial_positions():
    """Generate diverse initial configurations."""
    inits = []
    
    # 5x5x4 grid
    m1, m2, m3 = 5, 5, 4
    t1 = (np.arange(m1) + 0.5) / m1 - 0.5
    t2 = (np.arange(m2) + 0.5) / m2 - 0.5
    t3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in t3 for y in t2 for x in t1])
    inits.append(grid[:N].copy())
    
    # 4x5x5 grid
    m1, m2, m3 = 4, 5, 5
    t1 = (np.arange(m1) + 0.5) / m1 - 0.5
    t2 = (np.arange(m2) + 0.5) / m2 - 0.5
    t3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in t3 for y in t2 for x in t1])
    inits.append(grid[:N].copy())
    
    # Random uniform
    rng = np.random.RandomState(42)
    inits.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    # Jittered grid
    base = inits[0].copy()
    jitter = rng.normal(0, 0.015, (N, 3))
    inits.append(np.clip(base + jitter, -0.499, 0.499))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME_LIMIT
    
    best_radius = 0.0
    best_centers = None
    
    inits = _generate_initial_positions()
    
    # Phase 1: Spread with Riesz Energy
    # We run Riesz minimization on each init to get a good starting point for Soft-Min
    riesz_results = []
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
            
        time_left = deadline - time.time() - 30.0
        if time_left < 5.0:
            break
            
        # Use a portion of the time for Riesz
        t_limit = min(10.0, time_left * 0.5)
        opt_centers = _minimize_riesz(init_centers, s=6.0, time_limit=t_limit)
        riesz_results.append(opt_centers)
        
        r = _compute_radius(opt_centers)
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
            
    if best_centers is None:
        best_centers = inits[0]
        best_radius = _compute_radius(best_centers)
        
    # Phase 2: Refine with Soft-Min (Log-Sum-Exp)
    # We try multiple betas to sharpen the minimum
    # Start with the best Riesz result and a couple of others
    candidates = riesz_results if riesz_results else [best_centers]
    
    # Ensure we don't process too many candidates
    candidates = candidates[:4]
    
    for beta in [10.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 20.0:
            break
            
        for cand in candidates:
            if time.time() > deadline - 10.0:
                break
                
            time_left = deadline - time.time() - 5.0
            if time_left < 2.0:
                continue
                
            t_limit = min(5.0, time_left * 0.8)
            opt_centers = _minimize_soft_min(cand, beta, time_limit=t_limit)
            
            r = _compute_radius(opt_centers)
            if r > best_radius:
                best_radius = r
                best_centers = opt_centers.copy()
                
    # Phase 3: Final Polish
    # Use high beta soft-min or high s Riesz for a final squeeze
    if time.time() < deadline - 5.0 and best_centers is not None:
        time_left = deadline - time.time() - 2.0
        if time_left > 2.0:
            # Try a final high-precision soft-min
            polished = _minimize_soft_min(best_centers, beta=500.0, time_limit=time_left)
            r_final = _compute_radius(polished)
            if r_final > best_radius:
                best_radius = r_final
                best_centers = polished
    
    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
