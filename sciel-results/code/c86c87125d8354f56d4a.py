"""Ant colony orienteering heuristic: Nelder-Mead optimized density-prize hybrid."""
import numpy as np
import time

_DEADLINE = time.time() + 320.0  # 10s safety margin before 330s kill
_params_cache = None

def _simulate_aco(prizes, dist, max_len, heuristic, n, rng):
    """
    Simulate ACO and return the highest total prize collected by any ant in any iteration.
    Optimized for speed: pre-computed distances to home, vectorized where possible.
    """
    num_ants = 20
    num_iters = 30
    pheromone_decay = 0.9
    pheromone_gain_divisor = 10.0

    pheromone = np.ones((n, n))
    dist_to_home = dist[:, 0].copy()
    
    # Pre-compute feasibility masks for each node i, assuming we are at i and have used 0 length?
    # No, feasibility depends on path_len. 
    # However, we can pre-compute dist[i, :] + dist_to_home for quick checking.
    # feasible if path_len + dist[i,j] + dist[j,0] <= max_len
    # => dist[i,j] + dist[j,0] <= max_len - path_len
    
    # Matrix of dist[i,j] + dist[j,0] for all i,j
    # dist[i,j] is dist[i]
    # dist[j,0] is dist_to_home[j]
    # cost_matrix[i,j] = dist[i,j] + dist_to_home[j]
    cost_matrix = dist + dist_to_home[None, :]
    
    global_best = 0.0

    for it in range(num_iters):
        if time.time() > _DEADLINE - 15:
            break
        iter_best = 0.0

        for ant in range(num_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                remaining = max_len - path_len
                # We need dist[current, j] + dist_to_home[j] <= remaining
                # This is cost_matrix[current, :] <= remaining
                feasible_mask = cost_matrix[current, :] <= remaining + 1e-9
                feasible_mask[visited] = False
                feasible_mask[0] = False

                if not np.any(feasible_mask):
                    break

                candidates = np.where(feasible_mask)[0]
                # Get weights
                # Pheromone[current, candidates] * Heuristic[current, candidates]
                weights = pheromone[current, candidates] * heuristic[current, candidates]
                
                total_w = weights.sum()
                if total_w <= 0:
                    break

                # Normalize and choose
                probs = weights / total_w
                # rng.choice is a bit slow, use cumulative sum if needed, but numpy is fine for small n
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]

                path_len += dist[current, next_node]
                total_prize += prizes[next_node]
                visited[next_node] = True
                current = next_node

            if total_prize > iter_best:
                iter_best = total_prize

        pheromone *= pheromone_decay
        pheromone += iter_best / pheromone_gain_divisor

        if iter_best > global_best:
            global_best = iter_best

    return global_best

def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix.
    params: (alpha, beta, gamma, delta)
    alpha: exponent for d_ij
    beta: exponent for d_j0
    gamma: weight for local prize density
    delta: weight for direct prize
    """
    n = len(prizes)
    alpha, beta, gamma, delta = params
    
    d_ij = dist + 1e-6
    d_j0 = dist[:, 0] + 1e-6
    d_j0_row = d_j0[None, :] # (1, n)
    
    # Local Prize Density: For each node j, sum of prizes of nodes k 
    # such that d[j,k] + d[k,0] <= max_len (conservative reachability from j)
    # We compute this once.
    dist_to_home = dist[:, 0]
    local_density = np.zeros(n)
    # To speed this up, we can vectorize the feasibility check per j
    # But n=50 is small, so O(n^2) is fine.
    for j in range(n):
        # Nodes k reachable from j and returnable to home within max_len
        # Condition: dist[j,k] + dist[k,0] <= max_len
        feasible_k = (dist[j, :] + dist_to_home) <= max_len + 1e-9
        feasible_k[j] = False # Exclude self
        if np.any(feasible_k):
            local_density[j] = np.sum(prizes[feasible_k])
    
    # Normalize local density to [0, 1] to keep scale comparable
    max_ld = np.max(local_density)
    if max_ld > 0:
        local_density_norm = local_density / max_ld
    else:
        local_density_norm = np.zeros(n)
        
    # Direct Prize
    prize = prizes[None, :] # (1, n)
    
    # Combine Prize and Density
    # Value_j = delta * Prize_j + gamma * LocalDensity_j
    # We want this to be positive.
    value = delta * prize + gamma * local_density_norm[None, :]
    
    # Ensure value is positive
    value = np.maximum(value, 1e-9)
    
    # Denominator: d_ij^alpha * d_j0^beta
    denom = (d_ij ** alpha) * (d_j0_row ** beta)
    
    h = value / denom
    
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h

def _optimize_parameters():
    """
    Optimize heuristic parameters using Nelder-Mead on a few representative instances.
    """
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    # Generate representative instances
    n_instances = 5
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed * 11 + 5)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    # Initial guess for parameters: (alpha, beta, gamma, delta)
    # Typical good values: alpha~1, beta~1, gamma~0.5, delta~1.0
    x0 = np.array([1.0, 1.0, 0.5, 1.0])
    
    # Nelder-Mead implementation (simplified, using scipy if available, otherwise manual)
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
        # Fallback to simple grid/random if scipy not available, but scipy is standard in these tasks usually.
        # Let's assume scipy is available as per "Python standard library, NumPy and SciPy are available."

    eval_rng = np.random.RandomState(42)
    
    def objective(params):
        if time.time() > _DEADLINE - 30:
            return 1e9 # Penalize if running out of time
        
        alpha, beta, gamma, delta = params
        # Clamp parameters to reasonable ranges to avoid numerical issues
        alpha = np.clip(alpha, 0.1, 3.0)
        beta = np.clip(beta, 0.1, 3.0)
        gamma = np.clip(gamma, 0.0, 2.0)
        delta = np.clip(delta, 0.1, 2.0)
        
        clamped_params = (alpha, beta, gamma, delta)
        
        total_score = 0.0
        for prizes, dist, max_len in instances:
            n = len(prizes)
            heuristic = _make_heuristic(prizes, dist, max_len, clamped_params)
            score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
            total_score += score
            
        return -total_score / len(instances) # Minimize negative score

    best_params = x0
    best_score = -1.0
    
    if use_scipy:
        # Run Nelder-Mead
        # maxiter needs to be low to fit in time. 
        # Each evaluation is ~5 instances * 30 iters * 20 ants. 
        # 5 * 30 * 20 = 3000 ant runs. 
        # Each ant run is ~50 steps max. 150,000 steps.
        # This is fast in numpy. 
        # Let's allow ~50 iterations of Nelder-Mead.
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-4, 'disp': False})
            if res.success or res.fun < objective(x0):
                best_params = np.array(res.x)
                best_score = -res.fun
            else:
                # Keep initial if it failed
                pass
        except Exception:
            pass

    # If scipy failed or didn't improve, do a small random search around x0 as fallback
    if best_score < objective(x0): # objective returns negative, so lower is better
         # Actually, let's just run a few random perturbations if we have time
         current_best_params = x0
         current_best_score = objective(x0)
         
         for _ in range(20):
             if time.time() > _DEADLINE - 20:
                 break
             perturbation = np.random.normal(0, 0.2, size=4)
             candidate = x0 + perturbation
             candidate[0] = np.clip(candidate[0], 0.1, 3.0)
             candidate[1] = np.clip(candidate[1], 0.1, 3.0)
             candidate[2] = np.clip(candidate[2], 0.0, 2.0)
             candidate[3] = np.clip(candidate[3], 0.1, 2.0)
             
             score = objective(candidate)
             if score < current_best_score:
                 current_best_score = score
                 current_best_params = candidate

         best_params = current_best_params
         best_score = -current_best_score

    _params_cache = tuple(best_params)
    return _params_cache

def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    if _params_cache is None:
        params = _optimize_parameters()
    else:
        params = _params_cache
    
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
