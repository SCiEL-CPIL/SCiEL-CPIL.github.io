"""QAP solver for tai256c with iterated local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145  # leave 5s buffer for safety
    
    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng(42)
    
    # Precompute invariants
    a_sum = a + a.T
    diag_a = np.diag(a)
    
    def _compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    def _local_search(p, time_limit):
        """Vectorized 2-opt local search."""
        p = p.copy()
        D = b[np.ix_(p, p)].copy()
        current_cost = np.sum(a * D)
        diag_D = np.diag(D).copy()
        
        # Precompute for faster delta computation
        a_sum_D = a_sum @ D  # shape (n, n): a_sum[i] @ D[:, j] for all i, j
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute deltas for swapping position i with all j
                # S_j = (a_sum[i] - a_sum[j]) dot (D[j] - D[i])
                #     = a_sum[i]@D[j] - a_sum[i]@D[i] - a_sum[j]@D[j] + a_sum[j]@D[i]
                
                ai_dot_Di = np.dot(a_sum[i], D[i])
                ai_dot_D_all = a_sum[i] @ D  # shape (n,)
                aj_dot_Dj = np.sum(a_sum * D, axis=1)  # shape (n,)
                aj_dot_Di = a_sum @ D[i]  # shape (n,)
                
                S_all = ai_dot_D_all - ai_dot_Di - aj_dot_Dj + aj_dot_Di
                
                # T_i_j = (a_sum[i,i] - a_sum[j,i]) * (D[j,i] - D[i,i])
                coeff_i_all = a_sum[i, i] - a_sum[:, i]
                term_i_all = coeff_i_all * (D[:, i] - D[i, i])
                
                # T_j_i = (a_sum[i,j] - a_sum[j,j]) * (D[j,j] - D[i,j])
                coeff_j_all = a_sum[i, :] - np.diag(a_sum)
                term_j_all = coeff_j_all * (diag_D - D[i, :])
                
                # diag_term_j = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
                diag_term_all = (diag_a[i] - diag_a) * (diag_D - D[i, i])
                
                # cross_term_j = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
                cross_term_all = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
                
                deltas = S_all - term_i_all - term_j_all + diag_term_all + cross_term_all
                deltas[i] = 0  # Exclude self-swap
                
                # Find best improving move
                j = np.argmin(deltas)
                if deltas[j] < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    diag_D[i], diag_D[j] = diag_D[j], diag_D[i]
                    current_cost += deltas[j]
                    improved = True
        
        return p, current_cost
    
    def _greedy_init():
        """Greedy construction: place facilities one by one at best location."""
        p = np.empty(n, dtype=int)
        used_locs = set()
        placed = np.zeros(n, dtype=bool)
        
        # Start with facility that has highest total flow
        flow_sums = a.sum(axis=1)
        for step in range(n):
            if step == 0:
                f = np.argmax(flow_sums)
            else:
                # For unplaced facilities, compute cost of placing at each unused location
                best_cost = np.inf
                best_f, best_loc = -1, -1
                for f in range(n):
                    if placed[f]:
                        continue
                    for loc in range(n):
                        if loc in used_locs:
                            continue
                        # Compute incremental cost
                        inc_cost = 0
                        for g in range(n):
                            if placed[g]:
                                inc_cost += a[f, g] * b[loc, p[g]]
                        if inc_cost < best_cost:
                            best_cost = inc_cost
                            best_f, best_loc = f, loc
                f, loc = best_f, best_loc
            
            p[f] = loc
            used_locs.add(loc)
            placed[f] = True
        
        return p
    
    def _rank_match_init():
        """Initialize by matching facility ranks to location ranks based on row sums."""
        fac_order = np.argsort(a.sum(axis=1))
        loc_order = np.argsort(b.sum(axis=1))
        p = np.empty(n, dtype=int)
        p[fac_order] = loc_order
        return p
    
    def _perturb(p, k, rng):
        """Randomly swap k pairs of positions."""
        p = p.copy()
        for _ in range(k):
            i, j = rng.choice(n, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Rank-matching
    inits.append(_rank_match_init())
    
    # 2. Random permutations with different seeds
    for seed in [42, 123, 456]:
        r = np.random.default_rng(seed)
        inits.append(r.permutation(n))
    
    # 3. Greedy construction (may be slow, so limit time)
    greedy_time_limit = start_time + 10
    p_greedy = _greedy_init()
    inits.append(p_greedy)
    
    # Run local search on each initialization
    for idx, p_init in enumerate(inits):
        if time.time() - start_time > time_budget - 30:
            break
        
        remaining = time_budget - (time.time() - start_time) - 25
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search(p_init, start_time + remaining)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Iterated Local Search with adaptive perturbation
    stagnation = 0
    perturbation_k = 5
    
    while time.time() - start_time < time_budget - 5:
        # Adaptive perturbation: increase if stagnating
        if stagnation > 10:
            perturbation_k = min(15, perturbation_k + 2)
        elif stagnation < 3:
            perturbation_k = max(3, perturbation_k - 1)
        
        # Perturb best solution
        p_pert = _perturb(best_p, perturbation_k, rng)
        
        # Local search
        remaining = time_budget - (time.time() - start_time) - 3
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search(p_pert, start_time + remaining)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
            stagnation = 0
        else:
            stagnation += 1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
