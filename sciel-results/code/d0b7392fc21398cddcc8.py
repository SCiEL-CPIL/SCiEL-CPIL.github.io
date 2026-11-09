import numpy as np
import time

def _run_aco_vectorized(prizes, dist, max_len, heuristic, seed):
    """
    Run ACO with vectorized feasibility and probability checks.
    Returns the best total prize found.
    """
    n = len(prizes)
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix
    pheromone = np.ones((n, n), dtype=float)
    # Heuristic matrix, ensuring positive values
    heur = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Pre-compute distances to origin for quick feasibility checks
    # dist_to_origin[j] = dist[j, 0]
    dist_to_origin = dist[:, 0]
    
    best_overall = 0.0
    
    for it in range(30):
        best_iter = 0.0
        
        for ant in range(20):
            # State
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                # Find feasible nodes
                # Condition: not visited AND length + dist[current, j] + dist[j, 0] <= max_len
                # dist[current, :] gives distance from current to all j
                # dist_to_origin gives distance from all j to 0
                # feasible_mask[j] = (not visited_mask[j]) AND (length + dist[current, j] + dist_to_origin[j] <= max_len)
                
                # Vectorized feasibility check
                # Note: dist[current, :] is a vector of size n
                # dist_to_origin is a vector of size n
                # We need to mask out visited nodes
                feasible_mask = (~visited_mask) & (length + dist[current, :] + dist_to_origin <= max_len)
                
                # Get indices of feasible nodes
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    break
                
                # Calculate probabilities
                # p[j] = pheromone[current, j] * heur[current, j]
                # We only care about feasible_indices
                probs = pheromone[current, feasible_indices] * heur[current, feasible_indices]
                
                s = probs.sum()
                if s <= 0:
                    break
                
                probs = probs / s
                
                # Select next node
                idx = int(rng.choice(len(feasible_indices), p=probs))
                j = int(feasible_indices[idx])
                
                # Update state
                length += dist[current, j]
                total_prize += prizes[j]
                visited_mask[j] = True
                current = j
            
        if total_prize > best_iter:
            best_iter = total_prize
        if total_prize > best_overall:
            best_overall = total_prize
            
        # Pheromone update
        pheromone = pheromone * 0.9 + best_iter / 10.0
        
    return best_overall

def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix based on parameters.
    H[i, j] = (prize[j] + p_add)^p_exp / (dist[i, j] + ret_weight * dist[j, 0] + eps)^d_exp
    """
    eps = 1e-9
    p = prizes[None, :].astype(float)
    d = dist.astype(float)
    ret = dist[:, 0][None, :].astype(float) # Distance from j to 0, broadcasted to rows
    
    a = params.get('d_exp', 1.0)
    b = params.get('ret_weight', 0.0)
    c = params.get('p_exp', 1.0)
    k = params.get('p_add', 0.0)
    
    # Denominator: (d + b * ret + eps)^a
    denom_base = d + b * ret + eps
    denom = denom_base ** a
    
    # Numerator: (p + k)^c
    numer = (p + k) ** c
    
    H = numer / denom
    
    # Ensure finite and positive
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def _seed_heuristic(prizes, dist):
    H = (prizes[None, :] + 1e-9) / (dist + 1e-9)
    return np.where(np.isfinite(H) & (H > 0), H, 1e-9)

def heuristics(prizes, distance_matrix, max_len):
    try:
        start = time.time()
        deadline = start + 320.0 # 10s buffer
        
        prizes = np.asarray(prizes, dtype=float)
        dist = np.asarray(distance_matrix, dtype=float)
        n = len(prizes)
        
        # Initial coarse grid of parameters
        # d_exp: distance exponent
        # ret_weight: weight of return distance in denominator
        # p_exp: prize exponent
        # p_add: prize offset
        
        coarse_params = [
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 1.5, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 2.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 1.0, 'ret_weight': 0.5, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 1.0, 'ret_weight': 1.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 1.5, 'p_add': 0.0},
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 2.0, 'p_add': 0.0},
            {'d_exp': 1.5, 'ret_weight': 0.5, 'p_exp': 1.5, 'p_add': 0.0},
            {'d_exp': 2.0, 'ret_weight': 0.5, 'p_exp': 1.5, 'p_add': 0.0},
            {'d_exp': 1.0, 'ret_weight': 0.2, 'p_exp': 1.2, 'p_add': 0.1},
        ]
        
        best_score = 0.0
        best_params = None
        best_H = None
        
        # Phase 1: Coarse Search
        # We have 10 candidates. Let's evaluate each with 2 seeds first to find the promising region.
        # 10 candidates * 2 seeds * (30 iter * 20 ants * ~5 steps) ops.
        # Vectorized ACO is much faster. Let's try 3 seeds per candidate.
        
        for params in coarse_params:
            if time.time() > deadline - 30.0:
                break
            
            H = _make_heuristic(prizes, dist, max_len, params)
            scores = []
            # Use 3 seeds for coarse search
            for s in range(3):
                if time.time() > deadline - 30.0:
                    break
                score = _run_aco_vectorized(prizes, dist, max_len, H, s)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    best_H = H
        
        if best_params is None:
            # Fallback if no time was left or error
            return _seed_heuristic(prizes, dist)
        
        # Phase 2: Local Refinement
        # Perform a local search around best_params
        # Perturb each parameter slightly and test
        
        # Define small perturbations
        # d_exp: +/- 0.2
        # ret_weight: +/- 0.1
        # p_exp: +/- 0.2
        # p_add: +/- 0.05
        
        num_refine_candidates = 20
        refine_params_list = []
        
        # Generate neighbors
        base_d = best_params['d_exp']
        base_r = best_params['ret_weight']
        base_p = best_params['p_exp']
        base_a = best_params['p_add']
        
        for delta_d in [-0.2, 0.2]:
            for delta_r in [-0.1, 0.1]:
                for delta_p in [-0.2, 0.2]:
                    refine_params_list.append({
                        'd_exp': max(0.1, base_d + delta_d),
                        'ret_weight': max(0.0, base_r + delta_r),
                        'p_exp': max(0.1, base_p + delta_p),
                        'p_add': base_a
                    })
        
        # Also try changing p_add
        for delta_a in [-0.05, 0.05]:
            refine_params_list.append({
                'd_exp': base_d,
                'ret_weight': base_r,
                'p_exp': base_p,
                'p_add': max(0.0, base_a + delta_a)
            })
            
        # Shuffle to ensure we don't just test the same ones if we run out of time
        # Actually, we want to test the most promising ones first? 
        # The grid above is symmetric. Let's just iterate.
        
        for params in refine_params_list:
            if time.time() > deadline - 20.0:
                break
            
            H = _make_heuristic(prizes, dist, max_len, params)
            scores = []
            # Use 2 seeds for refinement to save time, or 3 if time allows
            num_seeds = 2
            for s in range(num_seeds):
                if time.time() > deadline - 20.0:
                    break
                score = _run_aco_vectorized(prizes, dist, max_len, H, s)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    best_H = H
        
        # Phase 3: Final Polish
        # If we have time, run the best params with more seeds to confirm and potentially get a slightly better H?
        # The H is deterministic given params. So more seeds just confirms the score.
        # We just need to return the H corresponding to the best params.
        
        # Ensure the final H is valid
        if best_H is None:
            return _seed_heuristic(prizes, dist)
            
        H_final = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
        return H_final

    except Exception:
        try:
            prizes = np.asarray(prizes, dtype=float)
            dist = np.asarray(distance_matrix, dtype=float)
            return _seed_heuristic(prizes, dist)
        except Exception:
            try:
                n = len(prizes)
            except Exception:
                n = 50
            return np.ones((n, n))
