"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco_fast(prizes, distance_matrix, max_len, heuristic, seed):
    """
    Fast simulation of the ACO colony. Returns best prize collected by any ant.
    """
    np.random.seed(seed)
    
    n = len(prizes)
    if n <= 1:
        return 0.0
    
    n_ants = 20
    n_iter = 30
    evaporation = 0.9
    deposition = 1.0 / 10.0
    
    pheromone = np.ones((n, n), dtype=np.float64)
    h = np.maximum(heuristic, 1e-9)
    
    best_total = 0.0
    d_j0 = distance_matrix[:, 0]
    
    for it in range(n_iter):
        iter_best = 0.0
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                remaining = max_len - length
                if remaining <= 0:
                    break
                
                # Vectorized feasibility check
                feasible_mask = (remaining - d_j0 >= 0) & (~visited)
                if not np.any(feasible_mask):
                    break
                
                # Need d[current, j] + d[j, 0] <= remaining
                eligible_mask = feasible_mask & (distance_matrix[current] <= (remaining - d_j0))
                eligible_mask[0] = False  # node 0 is start
                
                if not np.any(eligible_mask):
                    break
                
                eligible = np.where(eligible_mask)[0]
                
                # Compute attractiveness
                att = pheromone[current, eligible] * h[current, eligible]
                
                total_att = att.sum()
                if total_att <= 0 or not np.isfinite(total_att):
                    break
                
                # Normalize and sample
                probs = att / total_att
                probs = np.maximum(probs, 0)
                ps = probs.sum()
                if ps <= 0:
                    break
                probs = probs / ps
                
                r = np.random.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(eligible):
                    idx = len(eligible) - 1
                
                next_j = eligible[idx]
                
                length += distance_matrix[current, next_j]
                total_prize += prizes[next_j]
                visited[next_j] = True
                current = next_j
            
            if total_prize > iter_best:
                iter_best = total_prize
        
        if iter_best > best_total:
            best_total = iter_best
        
        # Update pheromone
        pheromone = pheromone * evaporation
        pheromone += deposition * iter_best
    
    return best_total


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness heuristic matrix using a 'regret' or opportunity-cost metric.
    
    The heuristic model combines:
    1. A base 'value density' term: prize[j] / local_density[j]
    2. A 'regret' term: inversely proportional to the maximum prize of other reachable nodes.
    
    h[i,j] = (prize[j] / density[j])^alpha * (1 + d[i,j])^(-beta) * (1 + d[j,0])^(-gamma) * (1 + max_prize_reachable[i,j])^(-delta)
    
    We optimize alpha, beta, gamma, delta, and k (for density) using a derivative-free search.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
    
    start_time = time.time()
    time_limit = 315.0  # Safety margin before 330s kill
    
    d_j0 = distance_matrix[:, 0]
    eps = 1e-9
    
    # Precompute base terms
    prize_term = np.maximum(prizes, eps)
    
    # Precompute local densities for a few fixed k values to speed up search
    # We will precompute for k in [3, 5, 7, 10]
    k_values = [3, 5, 7, 10]
    densities = {}
    for k in k_values:
        local_density = np.ones(n)
        for j in range(n):
            if j == 0:
                continue
            dists = distance_matrix[j].copy()
            dists[j] = np.inf
            k_eff = min(k, n - 1)
            if k_eff <= 0:
                local_density[j] = 1.0
                continue
            sorted_indices = np.argsort(dists)
            top_k_indices = sorted_indices[:k_eff]
            # Use uniform weights for density to keep it simple and fast
            weighted_avg_dist = np.mean(dists[top_k_indices])
            local_density[j] = max(weighted_avg_dist, eps)
        densities[k] = local_density

    def build_heuristic(k, alpha, beta, gamma, delta):
        """
        Build heuristic matrix.
        h[i,j] = (prize[j] / density[j])^alpha * (1 + d[i,j])^(-beta) * (1 + d[j,0])^(-gamma) * (1 + regret[i,j])^(-delta)
        
        regret[i,j] = max prize of other nodes reachable from i given we go to j?
        Actually, the prompt suggests: "inversely proportional to the maximum prize value of any other unvisited node k that is also reachable from i within the remaining length budget"
        
        Since the heuristic is static, we must approximate this.
        A good static approximation for "regret" of choosing j from i is:
        max_prize_of_other_reachable_nodes_from_i - prize[j]
        Or simply: max prize of nodes reachable from i (excluding j and 0).
        
        Let's define reachability from i as nodes k where d[i,k] + d[k,0] <= max_len.
        Then regret[i,j] = max(0, max_{k in Reach(i), k!=j, k!=0} prize[k] - prize[j])
        
        This encourages moving to high prize nodes that are NOT overshadowed by other reachable high prize nodes?
        Wait, the prompt says: "biasing ants away from choices that lead to lower opportunity costs compared to immediate alternatives."
        This is slightly ambiguous. Usually, regret-based heuristics (like in MCTS or regret matching) prefer actions that have the least regret.
        Regret of action j = best_possible_value - value(j).
        If we choose j, we lose the chance to choose the best alternative.
        So, if there is a very high prize node k reachable from i, and we choose a low prize node j, the regret is high.
        We want to MINIMIZE regret. So we want to MAXIMIZE the heuristic.
        Therefore, h[i,j] should be inversely proportional to regret.
        
        Regret[i,j] = max_{k in Reach(i)} prize[k] - prize[j]
        If prize[j] is the max, regret is 0 (or small). h is high.
        If prize[j] is low, regret is high. h is low.
        This effectively makes h[i,j] proportional to prize[j] if the max is fixed for a given i.
        But the max depends on i.
        
        Let's compute max_reachable_prize[i] = max(0, max_{k in Reach(i)} prize[k])
        Then regret[i,j] = max_reachable_prize[i] - prize[j]
        h[i,j] = ... * (1 + regret[i,j])^(-delta)
        
        If delta > 0, nodes with high regret (low prize relative to local max) are penalized.
        This reinforces the "go for the best available" strategy.
        """
        local_density = densities.get(k, np.ones(n))
        
        # Precompute max reachable prize for each node i
        # Reach(i) = {k | d[i,k] + d[k,0] <= max_len, k != 0}
        # Note: The heuristic is static, so we assume "remaining length" is max_len for the first step, or an average.
        # Using max_len is a reasonable upper bound for reachability from start, but for intermediate nodes, it varies.
        # However, since h is static, we must use a static proxy.
        # Let's use the set of nodes reachable from i with total path length d[i,k] + d[k,0] <= max_len.
        # This is the condition for being able to go i->k->0.
        
        max_reachable_prize = np.zeros(n)
        for i in range(n):
            # Nodes k such that d[i,k] + d[k,0] <= max_len
            # d[i,k] is distance_matrix[i, k]
            # d[k,0] is d_j0[k]
            reachable_mask = (distance_matrix[i] + d_j0) <= max_len
            reachable_mask[0] = False # exclude node 0
            if np.any(reachable_mask):
                max_reachable_prize[i] = np.max(prize_term[reachable_mask])
            else:
                max_reachable_prize[i] = 0.0
        
        # Compute regret for all pairs (i, j)
        # regret[i, j] = max_reachable_prize[i] - prize[j]
        # If j is not reachable from i, it shouldn't be chosen, but the mask handles that in ACO.
        # For the heuristic matrix, we compute it for all pairs.
        regret = max_reachable_prize[:, None] - prize_term[None, :]
        regret = np.maximum(regret, 0.0)
        
        # Base value term
        value_term = (prize_term / local_density) ** alpha
        
        # Distance decay
        dist_factor = (1.0 + distance_matrix) ** (-beta)
        
        # Return distance penalty
        return_factor = (1.0 + d_j0) ** (-gamma)
        
        # Regret penalty
        regret_factor = (1.0 + regret) ** (-delta)
        
        h = value_term[None, :] * dist_factor * return_factor[None, :] * regret_factor
        
        # Ensure positive
        h = np.maximum(h, 1e-9)
        
        return h

    def evaluate_params(k, alpha, beta, gamma, delta, seed=42):
        """Evaluate parameters by running ACO simulation."""
        h = build_heuristic(k, alpha, beta, gamma, delta)
        return _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=seed)
    
    best_score = -1.0
    best_h = None
    best_params = None
    
    # Phase 1: Random search
    np.random.seed(12345)
    
    n_random = 150
    evaluated = 0
    
    # Ranges
    alpha_range = (0.5, 3.0)
    beta_range = (0.5, 3.0)
    gamma_range = (0.5, 2.0)
    delta_range = (0.5, 3.0)
    
    for _ in range(n_random):
        if time.time() - start_time > time_limit:
            break
        
        k = k_values[np.random.randint(len(k_values))]
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        gamma = np.random.uniform(*gamma_range)
        delta = np.random.uniform(*delta_range)
        
        score = evaluate_params(k, alpha, beta, gamma, delta, seed=42)
        evaluated += 1
        
        if score > best_score:
            best_score = score
            best_h = build_heuristic(k, alpha, beta, gamma, delta)
            best_params = (k, alpha, beta, gamma, delta)
    
    # Phase 2: Local refinement
    if best_params is not None:
        base_k, base_alpha, base_beta, base_gamma, base_delta = best_params
        
        # Perturbations
        for da in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            for db in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                for dg in [-0.1, 0.0, 0.1]:
                    for dd in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                        if time.time() - start_time > time_limit:
                            break
                        
                        a = max(0.1, base_alpha + da)
                        b = max(0.1, base_beta + db)
                        g = max(0.1, base_gamma + dg)
                        d = max(0.1, base_delta + dd)
                        
                        # Also try changing k slightly? k is discrete.
                        # Let's just stick to continuous params for refinement to save time.
                        
                        score = evaluate_params(base_k, a, b, g, d, seed=42)
                        evaluated += 1
                        
                        if score > best_score:
                            best_score = score
                            best_h = build_heuristic(base_k, a, b, g, d)
                            best_params = (base_k, a, b, g, d)
        
        # Try changing k if time permits
        if time.time() - start_time < time_limit * 0.8:
            for k in k_values:
                if k == base_k:
                    continue
                score = evaluate_params(k, base_alpha, base_beta, base_gamma, base_delta, seed=42)
                evaluated += 1
                if score > best_score:
                    best_score = score
                    best_h = build_heuristic(k, base_alpha, base_beta, base_gamma, base_delta)
                    best_params = (k, base_alpha, base_beta, base_gamma, base_delta)
    
    # Fallback
    if best_h is None:
        best_h = build_heuristic(5, 1.0, 1.0, 1.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
