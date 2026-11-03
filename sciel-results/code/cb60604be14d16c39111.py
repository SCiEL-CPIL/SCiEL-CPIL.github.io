"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    # p is a list/array of length n; facility i -> location p[i]
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient: (a * b[np.ix_(p, p)]).sum()
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _greedy_initial(a, b):
    """Greedy construction: place facilities one by one to minimize incremental cost."""
    n = len(a)
    # Incremental cost of placing facility i at location k given current assignment
    # We build assignment p where p[i] = location for facility i
    # Use a priority approach: for each unassigned facility, compute best location
    
    # Precompute for efficiency
    # When we place facility i at location k, the additional cost is:
    #   sum_j a[i,j] * b[k, p[j]] + sum_j a[j,i] * b[p[j], k]
    # where j are already placed facilities
    
    # This is O(n^3) naive, which for n=256 is ~16M ops, feasible.
    
    # Better: maintain a running "attraction" vector for each location.
    # Let's use a simpler greedy: sort facilities by total flow, assign to best location.
    
    # Actually, let's do a proper greedy:
    # For each step, for each unassigned facility i and each unassigned location k,
    # compute the marginal cost. Pick the (i,k) pair with minimum marginal cost.
    # This is O(n^3) total which is fine for n=256.
    
    # Simpler and faster: assign in order of facility importance.
    # Facility i's "importance" = sum(a[i,:]) + sum(a[:,i])
    # For each facility in order of importance, assign to the location that minimizes
    # the partial cost contribution.
    
    importance = a.sum(axis=1) + a.sum(axis=0)
    order = np.argsort(-importance, kind='stable')
    
    p = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    used_loc = np.zeros(n, dtype=bool)
    
    # For each facility in order, compute cost of assigning to each free location
    # Cost of assigning facility i to location k:
    #   sum_{j: assigned} a[i,j] * b[k, p[j]] + a[j,i] * b[p[j], k]
    # We can maintain for each location k a "load" vector.
    
    # Let's maintain: for each location k, the accumulated cost contribution from 
    # already-assigned facilities.
    # When facility j is assigned to location p[j], it contributes to location k:
    #   a[j,k_facility] * b[p[j], k]  -- no, this is getting complicated.
    
    # Let me just do the straightforward O(n^3) greedy.
    # For n=256, that's 256^3 = 16.7M operations, each simple. Should be fast enough.
    
    for i in order:
        best_k = -1
        best_cost = float('inf')
        for k in range(n):
            if used_loc[k]:
                continue
            # Compute marginal cost of assigning i to k
            # = sum_{j: p[j] >= 0} a[i,j]*b[k,p[j]] + a[j,i]*b[p[j],k]
            # Only over already assigned j
            assigned_idx = np.where(assigned)[0]
            if len(assigned_idx) == 0:
                cost = 0
            else:
                # a[i, assigned_idx] * b[k, p[assigned_idx]]
                # a[assigned_idx, i] * b[p[assigned_idx], k]
                c1 = np.sum(a[i, assigned_idx] * b[k, p[assigned_idx]])
                c2 = np.sum(a[assigned_idx, i] * b[p[assigned_idx], k])
                cost = c1 + c2
            if cost < best_cost:
                best_cost = cost
                best_k = k
        p[i] = best_k
        used_loc[best_k] = True
        assigned[i] = True
    
    return p.tolist()


def _local_search_2opt(a, b, p, time_limit=10.0):
    """2-opt local search: swap two facility locations, accept if cost improves."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _qap_cost(a, b, p)
    
    start_time = time.time()
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for faster delta computation
    # Delta of swapping p[i] and p[j]:
    # Let ki = p[i], kj = p[j]
    # New cost - old cost = 
    #   a[i,j]*(b[kj,ki] - b[ki,kj]) + a[j,i]*(b[ki,kj] - b[kj,ki])
    #   + sum_{m != i,j} a[i,m]*(b[kj,p[m]] - b[ki,p[m]]) + a[m,i]*(b[p[m],kj] - b[p[m],ki])
    #   + sum_{m != i,j} a[j,m]*(b[ki,p[m]] - b[kj,p[m]]) + a[m,j]*(b[p[m],ki] - b[p[m],kj])
    
    # This is O(n) per swap evaluation. With n=256, checking all pairs is O(n^3) per full pass.
    # 256^3 = 16.7M, which might be too slow for multiple passes.
    
    # Let's do a more efficient version: for each i, compute the delta for swapping i with each j.
    # We can vectorize this.
    
    # For a fixed i with location ki, the delta of swapping i and j (location kj) is:
    # delta = a[i,j]*(b[kj,ki]-b[ki,kj]) + a[j,i]*(b[ki,kj]-b[kj,ki])
    #         + sum_{m != i,j} [a[i,m]*(b[kj,p[m]]-b[ki,p[m]]) + a[m,i]*(b[p[m],kj]-b[p[m],ki])]
    #         + sum_{m != i,j} [a[j,m]*(b[ki,p[m]]-b[kj,p[m]]) + a[m,j]*(b[p[m],ki]-b[p[m],kj])]
    
    # Let's define for each location k and facility i:
    # row_i_k = sum_m a[i,m] * b[k, p[m]]  (cost of i at k from i's row)
    # col_i_k = sum_m a[m,i] * b[p[m], k]  (cost of i at k from i's col)
    # Then total cost contribution of facility i at location k (excluding self) is row_i_k + col_i_k
    # But we need to be careful about double counting.
    
    # Actually, let's just do a randomized hill climbing with limited iterations.
    # Given time constraints, let's do a simpler approach: random swaps with acceptance.
    
    np.random.seed(42)
    
    while time.time() - start_time < time_limit:
        # Try a random swap
        i, j = np.random.choice(n, 2, replace=False)
        ki, kj = p[i], p[j]
        
        # Compute delta
        # Old contributions involving i or j:
        # We need to recompute the change in total cost.
        
        # Total cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
        # When we swap p[i] and p[j], the locations ki and kj are exchanged for facilities i and j.
        
        # The change involves all terms where u or v is i or j.
        # Terms with u=i, v=j: a[i,j]*b[kj,ki] (new) vs a[i,j]*b[ki,kj] (old)
        # Terms with u=j, v=i: a[j,i]*b[ki,kj] (new) vs a[j,i]*b[kj,ki] (old)
        # Terms with u=i, v=m (m!=i,j): a[i,m]*b[kj,p[m]] (new) vs a[i,m]*b[ki,p[m]] (old)
        # Terms with u=m, v=i (m!=i,j): a[m,i]*b[p[m],kj] (new) vs a[m,i]*b[p[m],ki] (old)
        # Terms with u=j, v=m (m!=i,j): a[j,m]*b[ki,p[m]] (new) vs a[j,m]*b[kj,p[m]] (old)
        # Terms with u=m, v=j (m!=i,j): a[m,j]*b[p[m],ki] (new) vs a[m,j]*b[p[m],kj] (old)
        
        # delta = a[i,j]*(b[kj,ki]-b[ki,kj]) + a[j,i]*(b[ki,kj]-b[kj,ki])
        #         + sum_{m!=i,j} a[i,m]*(b[kj,p[m]]-b[ki,p[m]]) + a[m,i]*(b[p[m],kj]-b[p[m],ki])
        #         + sum_{m!=i,j} a[j,m]*(b[ki,p[m]]-b[kj,p[m]]) + a[m,j]*(b[p[m],ki]-b[p[m],kj])
        
        # Vectorized:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        pm = p[mask]
        
        delta = (a[i,j] * (b[kj,ki] - b[ki,kj]) + 
                 a[j,i] * (b[ki,kj] - b[kj,ki]) +
                 np.sum(a[i, mask] * (b[kj, pm] - b[ki, pm])) + 
                 np.sum(a[mask, i] * (b[pm, kj] - b[pm, ki])) +
                 np.sum(a[j, mask] * (b[ki, pm] - b[kj, pm])) + 
                 np.sum(a[mask, j] * (b[pm, ki] - b[pm, kj])))
        
        if delta < 0:
            p[i], p[j] = kj, ki
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p.tolist(), int(best_cost)


def solve(a, b):
    """Solve the QAP instance using greedy construction + local search."""
    n = len(a)
    total_start = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy initial solution
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    # Phase 2: Local search from greedy
    time_remaining = deadline - (time.time() - total_start)
    if time_remaining > 5:
        p_ls, cost_ls = _local_search_2opt(a, b, p_greedy, time_limit=min(time_remaining - 2, 30.0))
        if cost_ls < cost_greedy:
            p_best = p_ls
            cost_best = cost_ls
        else:
            p_best = p_greedy
            cost_best = cost_greedy
    else:
        p_best = p_greedy
        cost_best = cost_greedy
    
    # Phase 3: If time remains, try a second greedy with different ordering + local search
    time_remaining = deadline - (time.time() - total_start)
    if time_remaining > 10:
        # Different greedy: random perturbation of importance order
        np.random.seed(123)
        importance = a.sum(axis=1) + a.sum(axis=0)
        # Perturb
        perturbed = importance + np.random.normal(0, 0.1 * np.std(importance), n)
        order = np.argsort(-perturbed, kind='stable')
        
        # Rebuild p with this order
        p2 = np.full(n, -1, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        used_loc = np.zeros(n, dtype=bool)
        
        for i in order:
            best_k = -1
            best_cost = float('inf')
            for k in range(n):
                if used_loc[k]:
                    continue
                assigned_idx = np.where(assigned)[0]
                if len(assigned_idx) == 0:
                    cost = 0
                else:
                    c1 = np.sum(a[i, assigned_idx] * b[k, p2[assigned_idx]])
                    c2 = np.sum(a[assigned_idx, i] * b[p2[assigned_idx], k])
                    cost = c1 + c2
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            p2[i] = best_k
            used_loc[best_k] = True
            assigned[i] = True
        
        cost2 = _qap_cost(a, b, p2)
        if cost2 < cost_best:
            p_best = p2.tolist()
            cost_best = cost2
        
        # Local search from best
        time_remaining = deadline - (time.time() - total_start)
        if time_remaining > 5:
            p_ls2, cost_ls2 = _local_search_2opt(a, b, p_best, time_limit=min(time_remaining - 2, 20.0))
            if cost_ls2 < cost_best:
                p_best = p_ls2
                cost_best = cost_ls2
    
    return p_best
# EVOLVE-BLOCK-END
