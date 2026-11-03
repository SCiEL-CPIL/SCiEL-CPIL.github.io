"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for assignment p."""
    n = len(p)
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _greedy_initial(a, b):
    """Greedy construction: place facilities one by one at the best location."""
    n = len(a)
    # Precompute location sums for quick evaluation
    # Cost of placing facility i at location j, given already placed facilities:
    #   sum_{k placed} a[i,k]*b[j, p[k]] + a[k,i]*b[p[k], j]
    # We'll build incrementally.
    
    p = np.full(n, -1, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    
    # For efficiency, maintain for each unplaced facility i and unplaced location j:
    # incremental cost contribution from already-placed facilities
    # inc[i, j] = sum_{k placed} (a[i,k] + a[k,i]) * b[j, p[k]]
    # Actually more carefully:
    # When we place facility i at location j, the new cost terms involving i are:
    #   sum_{k placed} a[i,k] * b[j, p[k]] + sum_{k placed} a[k,i] * b[p[k], j]
    # Since b is symmetric in typical QAP instances but not guaranteed, keep both.
    
    # Precompute a_sym = a + a.T for the combined flow
    # Actually the cost is sum_{i,j} a[i,j]*b[p[i],p[j]], so when placing i at j:
    # contribution = sum_{k placed} a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j]
    # Let's maintain a matrix M[i, j] = sum_{k placed} (a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j])
    
    # This is O(n^3) naive but with incremental updates O(n^3) total.
    # For n=256, n^3 = 16.7M which is fine.
    
    # M[i, j] for unplaced i and unplaced j
    M = np.zeros((n, n), dtype=np.int64)
    
    for _ in range(n):
        # For each unplaced facility i, find best unplaced location j minimizing M[i,j]
        best_i = -1
        best_j = -1
        best_cost = np.iinfo(np.int64).max
        for i in range(n):
            if placed[i]:
                continue
            for j in range(n):
                if p[j] != -1:
                    continue
                c = M[i, j]
                if c < best_cost:
                    best_cost = c
                    best_i = i
                    best_j = j
        
        # Place facility best_i at location best_j
        p[best_i] = best_j
        placed[best_i] = True
        
        # Update M for remaining unplaced facilities
        for i in range(n):
            if placed[i]:
                continue
            # M[i, j] += a[i, best_i] * b[j, best_j] + a[best_i, i] * b[best_j, j]
            # for all unplaced j
            for j in range(n):
                if p[j] != -1:
                    continue
                M[i, j] += a[i, best_i] * b[j, best_j] + a[best_i, i] * b[best_j, j]
    
    # p[i] = location of facility i. But we need permutation where p[i] is the location.
    # Wait, the interface says: Facility i is placed at location p[i].
    # So p[i] should be the location index. Let's verify.
    # "Facility i is placed at location p[i]"
    # Cost: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # So p is a list where p[i] = location assigned to facility i.
    
    return p.tolist()


def _local_search_2opt(a, b, p, deadline):
    """2-opt local search: swap two locations and accept if cost decreases."""
    n = len(p)
    p_arr = np.array(p, dtype=np.int64)
    
    # Precompute current cost
    bp = b[np.ix_(p_arr, p_arr)]
    current_cost = int(np.sum(a * bp))
    
    # Precompute for fast delta evaluation:
    # When swapping locations of facilities i and j (i.e., p[i] <-> p[j]):
    # The change in cost involves terms where i or j appears.
    # Let's define:
    #   For a pair (i, j) where we swap their locations:
    #   delta = sum_{k != i, j} [a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]]
    #           - sum_{k != i, j} [a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]] + a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]]
    #           + a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    #           - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]
    
    # This is O(n) per swap evaluation, O(n^2) per pass, O(n^3) per full improvement.
    # For n=256, n^3 = 16.7M per pass. With multiple passes, this could be slow.
    
    # Let's use a vectorized approach for the delta computation.
    
    best_cost = current_cost
    best_p = p_arr.copy()
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # Randomize order to avoid bias
        order = list(range(n))
        random.shuffle(order)
        
        for idx_i in range(n):
            i = order[idx_i]
            if time.time() >= deadline:
                break
            for idx_j in range(idx_i + 1, n):
                j = order[idx_j]
                
                # Compute delta for swapping p[i] and p[j]
                pi = p_arr[i]
                pj = p_arr[j]
                
                # Terms involving i and j with each other
                old_ii = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
                new_ii = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
                
                # Terms involving i with all k != i, j
                # Old: sum_k a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi]
                # New: sum_k a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj]
                # Delta_i = sum_k [a[i,k]*b[pj,p[k]] - a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pj] - a[k,i]*b[p[k],pi]]
                #         = sum_k [a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])]
                
                # Similarly for j
                # Delta_j = sum_k [a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])]
                
                # Combined delta for all k != i, j:
                # sum_k [a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
                #        + a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])]
                # = sum_k [(a[i,k] - a[j,k])*(b[pj,p[k]] - b[pi,p[k]])
                #        + (a[k,i] - a[k,j])*(b[p[k],pj] - b[p[k],pi])]
                
                # Let's compute this vectorized
                # d_b1[k] = b[pj, p[k]] - b[pi, p[k]]
                # d_b2[k] = b[p[k], pj] - b[p[k], pi]
                # delta_k = sum_k (a[i,k] - a[j,k]) * d_b1[k] + (a[k,i] - a[k,j]) * d_b2[k]
                
                # Exclude i and j from k
                d_b1 = b[pj, p_arr] - b[pi, p_arr]
                d_b2 = b[p_arr, pj] - b[p_arr, pi]
                delta_k = np.sum((a[i, :] - a[j, :]) * d_b1 + (a[:, i] - a[:, j]) * d_b2)
                
                # But this includes k=i and k=j, which we need to handle separately
                # Actually let's just compute the full delta including all k, then subtract the i,j terms and add back correctly.
                
                # Simpler: compute delta directly
                # Old contribution of i and j to total cost (excluding their mutual term):
                # For i: sum_{k != i} a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi]
                # For j: sum_{k != j} a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
                # Mutual: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
                
                # New:
                # For i: sum_{k != i} a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj]
                # For j: sum_{k != j} a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
                # Mutual: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
                
                # Let's just compute the full delta using the vectorized formula and verify
                delta = int(delta_k + new_ii - old_ii)
                
                if delta < 0:
                    # Accept swap
                    p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p_arr.copy()
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best_p.tolist(), best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple initializations and local searches
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Strategy 1: Rank-matching initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1_list = p1.tolist()
    
    cost1, _ = _local_search_2opt(a, b, p1_list, deadline)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = cost1
    
    if time.time() > deadline:
        return best_p
    
    # Strategy 2: Greedy initialization
    if time.time() < deadline - 30:
        p2 = _greedy_initial(a, b)
        cost2, _ = _local_search_2opt(a, b, p2, deadline)
        if cost2 < best_cost:
            best_cost = cost2
            best_p = cost2
    
    if time.time() > deadline:
        return best_p
    
    # Strategy 3: Random initialization with local search
    if time.time() < deadline - 20:
        p3 = list(range(n))
        random.shuffle(p3)
        cost3, _ = _local_search_2opt(a, b, p3, deadline)
        if cost3 < best_cost:
            best_cost = cost3
            best_p = cost3
    
    if time.time() > deadline:
        return best_p
    
    # Strategy 4: Another random restart
    if time.time() < deadline - 15:
        p4 = list(range(n))
        random.shuffle(p4)
        cost4, _ = _local_search_2opt(a, b, p4, deadline)
        if cost4 < best_cost:
            best_cost = cost4
            best_p = cost4
    
    return best_p
# EVOLVE-BLOCK-END
