"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco_fast(prizes, distance_matrix, max_len, heuristic, seed):
    """
    Fast simulation of the ACO colony. Returns best prize collected by any ant.
    Uses vectorized operations where possible.
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
    dm = distance_matrix
    pz = prizes
    
    for it in range(n_iter):
        iter_best = 0.0
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=np.bool_)
            visited[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                remaining = max_len - length
                if remaining <= 0:
                    break
                
                # Eligible: unvisited, not node 0, and length + d[current,j] + d[j,0] <= max_len
                # i.e., d[current,j] <= remaining - d[j,0]
                rem_dj0 = remaining - d_j0
                eligible_mask = (rem_dj0 >= 0) & (~visited)
                eligible_mask[0] = False
                eligible_mask &= (dm[current] <= rem_dj0)
                
                if not eligible_mask.any():
                    break
                
                eligible = np.flatnonzero(eligible_mask)
                
                att = pheromone[current, eligible] * h[current, eligible]
                
                total_att = att.sum()
                if total_att <= 0 or not np.isfinite(total_att):
                    break
                
                probs = att / total_att
                r = np.random.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(eligible):
                    idx = len(eligible) - 1
                
                next_j = int(eligible[idx])
                
                length += dm[current, next_j]
                total_prize += pz[next_j]
                visited[next_j] = True
                current = next_j
            
            if total_prize > iter_best:
                iter_best = total_prize
        
        if iter_best > best_total:
            best_total = iter_best
        
        pheromone *= evaporation
        pheromone += deposition * iter_best
    
    return best_total


def _build_heuristic(prizes, distance_matrix, max_len, alpha, scale, p_beta, dual_gamma, budget_beta, regret_beta=1.0):
    """
    Build heuristic matrix with parameters.
    Adds a regret term based on the maximum prize of other reachable nodes.
    h[i,j] = (prize[j] / M[i,j])^regret_beta * prize[j]^beta * exp(-d[j,0]/scale) / (d[i,j]+eps)^alpha
    where M[i,j] is the max prize of k != j reachable from i.
    """
    eps = 1e-9
    n = len(prizes)
    d_j0 = distance_matrix[:, 0]
    
    prize_term = np.maximum(prizes, eps)
    prize_pow = prize_term ** p_beta
    
    return_cost = np.exp(-d_j0 / scale)
    
    if dual_gamma > 0:
        dual_pen = (d_j0 + eps) ** (-dual_gamma)
    else:
        dual_pen = np.ones(n)
    
    node_attr = prize_pow * return_cost * dual_pen  # shape (n,)
    
    d_plus = distance_matrix + eps
    dist_pen = d_plus ** (-alpha)  # shape (n, n)
    
    if budget_beta > 0:
        budget_pen = np.exp(-budget_beta * distance_matrix / max_len)
        dist_pen = dist_pen * budget_pen
    
    h = node_attr[None, :] * dist_pen
    
    # Compute regret term
    # Reachable[i] = nodes j such that d[i,j] + d[j,0] <= max_len
    # For each i, find max and second max prize in Reachable[i]
    regret_term = np.ones((n, n))
    
    reach_mask = (distance_matrix + d_j0[None, :]) <= max_len
    # reach_mask[i, j] is True if j is reachable from i
    
    for i in range(n):
        reachable_j = np.where(reach_mask[i])[0]
        if len(reachable_j) == 0:
            continue
        prizes_in_reach = prizes[reachable_j]
        # Find indices of top 2 prizes
        # argsort gives indices in sorted order
        sorted_idx = np.argsort(prizes_in_reach)[::-1]
        max_prize = prizes_in_reach[sorted_idx[0]]
        if len(sorted_idx) > 1:
            sec_max_prize = prizes_in_reach[sorted_idx[1]]
        else:
            sec_max_prize = 0.0
        
        # For each j in reachable_j:
        # If prize[j] == max_prize, M[i,j] = sec_max_prize
        # Else M[i,j] = max_prize
        for j in reachable_j:
            if prizes[j] == max_prize:
                M = sec_max_prize + eps
            else:
                M = max_prize + eps
            
            # Regret factor: (prize[j] / M)^regret_beta
            # If j is the best, factor > 1. If j is not best, factor < 1.
            regret_term[i, j] = (prizes[j] / M) ** regret_beta
    
    h = h * regret_term
    h = np.maximum(h, 1e-9)
    
    return h


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness heuristic matrix.
    
    Uses a dual-distance penalty structure with budget-aware edge penalty:
    h[i,j] = prize[j]^beta * exp(-d[j,0]/scale) / (d[i,j]+eps)^alpha * (d[j,0]+eps)^(-gamma) * exp(-bb*d[i,j]/max_len)
    
    The budget_beta term captures that long edges consume disproportionate budget,
    leaving less room for the return trip.
    
    Parameter search via fast ACO simulation.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
    
    start_time = time.time()
    time_limit = 280.0
    
    # Parameter grid: (alpha, scale, p_beta, dual_gamma, budget_beta, regret_beta)
    # We add regret_beta to the search space.
    # To keep the grid size manageable, we will test regret_beta in [0.0, 0.5, 1.0, 2.0]
    # by expanding the base grid.
    base_params = [
        (1.0, 0.5, 1.0, 0.0, 0.0),
        (1.5, 0.5, 1.0, 0.0, 0.0),
        (2.0, 0.5, 1.0, 0.0, 0.0),
        (1.0, 0.7, 1.0, 0.0, 0.0),
        (1.5, 0.7, 1.0, 0.0, 0.0),
        (2.0, 0.7, 1.0, 0.0, 0.0),
        (1.0, 0.5, 1.5, 0.0, 0.0),
        (1.5, 0.5, 1.5, 0.0, 0.0),
        (2.0, 0.5, 1.5, 0.0, 0.0),
        (1.0, 0.5, 1.0, 0.5, 0.0),
        (1.5, 0.5, 1.0, 0.5, 0.0),
        (2.0, 0.5, 1.0, 0.5, 0.0),
        (1.0, 0.5, 1.0, 0.0, 0.5),
        (1.5, 0.5, 1.0, 0.0, 0.5),
        (2.0, 0.5, 1.0, 0.0, 0.5),
        (1.0, 0.5, 1.0, 0.5, 0.5),
        (1.5, 0.5, 1.0, 0.5, 0.5),
        (2.0, 0.5, 1.0, 0.5, 0.5),
    ]
    
    param_sets = []
    for bp in base_params:
        for rb in [0.0, 0.5, 1.0, 2.0]:
            param_sets.append((bp[0], bp[1], bp[2], bp[3], bp[4], rb))
    
    # Shuffle for diversity
    np.random.seed(42)
    perm = np.random.permutation(len(param_sets))
    param_sets = [param_sets[i] for i in perm]
    
    best_score = -1.0
    best_h = None
    best_params = None
    
    # First pass: test each param set with seed 42
    for alpha, scale, p_beta, dual_gamma, budget_beta, regret_beta in param_sets:
        if time.time() - start_time > time_limit:
            break
        
        h = _build_heuristic(prizes, distance_matrix, max_len, alpha, scale, p_beta, dual_gamma, budget_beta, regret_beta)
        
        score = _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=42)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
            best_params = (alpha, scale, p_beta, dual_gamma, budget_beta, regret_beta)
    
    # Second pass: local refinement around best params if time permits
    if best_h is not None and time.time() - start_time < time_limit * 0.50:
        best_alpha, best_scale, best_p_beta, best_dual, best_bb, best_rb = best_params
        local_candidates = []
        for da in [-0.15, -0.05, 0.05, 0.15]:
            for ds in [-0.1, -0.05, 0.05, 0.1]:
                for dpb in [-0.15, 0.15]:
                    for ddg in [-0.15, -0.05, 0.05, 0.15]:
                        for dbb in [-0.15, -0.05, 0.05, 0.15]:
                            for drb in [-0.25, -0.1, 0.1, 0.25]:
                                la = best_alpha + da
                                ls = best_scale + ds
                                lpb = best_p_beta + dpb
                                ldg = best_dual + ddg
                                lbb = best_bb + dbb
                                lrb = best_rb + drb
                                if la > 0.1 and ls > 0.1 and lpb > 0.1 and ldg >= 0 and lbb >= 0 and lrb >= 0:
                                    local_candidates.append((la, ls, lpb, ldg, lbb, lrb))
        
        # Limit local candidates to avoid timeout
        max_local = 100
        if len(local_candidates) > max_local:
            local_candidates = local_candidates[:max_local]
        
        for alpha, scale, p_beta, dual_gamma, budget_beta, regret_beta in local_candidates:
            if time.time() - start_time > time_limit * 0.82:
                break
            
            h = _build_heuristic(prizes, distance_matrix, max_len, alpha, scale, p_beta, dual_gamma, budget_beta, regret_beta)
            
            score = _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=42)
            
            if score > best_score:
                best_score = score
                best_h = h.copy()
                best_params = (alpha, scale, p_beta, dual_gamma, budget_beta, regret_beta)
    
    # Third pass: verify robustness with additional seeds (monitoring only)
    if best_h is not None and time.time() - start_time < time_limit * 0.88:
        for test_seed in [123, 7, 99, 12345]:
            if time.time() - start_time > time_limit:
                break
            s = _simulate_aco_fast(prizes, distance_matrix, max_len, best_h, seed=test_seed)
    
    # Fallback if no heuristic was built
    if best_h is None:
        best_h = _build_heuristic(prizes, distance_matrix, max_len, 1.0, 0.5, 1.0, 0.5, 0.5, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
