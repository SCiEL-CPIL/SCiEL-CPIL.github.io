"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Precompute some useful quantities
    # For a swap of p[i] and p[j], the change in cost can be computed in O(n)
    # Let's define helper functions for efficient cost update
    
    def initial_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    # Efficient swap delta computation
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # If we swap p[i] and p[j], let x = p[i], y = p[j]
    # The terms that change are those involving i or j as first index, or p[i] or p[j] as second index
    # Actually, let's think about this more carefully.
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # After swapping p[i] <-> p[j]:
    # New cost = sum_{k,l} a[k,l] * b[p'[k], p'[l]]
    # where p' differs from p only at positions i and j.
    # 
    # The change only affects terms where k in {i,j} or p'[k] in {x,y} (i.e., k in {i,j} since p is a permutation)
    # Wait, no. p'[k] = p[k] for k not in {i,j}, p'[i] = y, p'[j] = x.
    # So b[p'[k], p'[l]] changes when k in {i,j} OR l in {i,j}.
    # 
    # Delta = sum_{k,l} a[k,l] * (b[p'[k], p'[l]] - b[p[k], p[l]])
    # This is non-zero only when k in {i,j} or l in {i,j}.
    # 
    # Let's break it down:
    # Terms with k=i, l not in {i,j}: a[i,l]*(b[y, p[l]] - b[x, p[l]])
    # Terms with k=j, l not in {i,j}: a[j,l]*(b[x, p[l]] - b[y, p[l]])
    # Terms with k not in {i,j}, l=i: a[k,i]*(b[p[k], y] - b[p[k], x])
    # Terms with k not in {i,j}, l=j: a[k,j]*(b[p[k], x] - b[p[k], y])
    # Terms with k=i, l=i: a[i,i]*(b[y,y] - b[x,x])
    # Terms with k=j, l=j: a[j,j]*(b[x,x] - b[y,y])
    # Terms with k=i, l=j: a[i,j]*(b[y,x] - b[x,y])
    # Terms with k=j, l=i: a[j,i]*(b[x,y] - b[y,x])
    
    # Let's precompute row sums of a weighted by b rows/cols for faster updates
    # Actually, for n=100, O(n) per swap is fine. Let's just compute delta in O(n).
    
    def swap_delta(p, i, j):
        """Compute change in cost if we swap p[i] and p[j]."""
        x = p[i]
        y = p[j]
        if x == y:
            return 0
        
        # We need to compute the change efficiently.
        # Let's precompute for each location k: 
        #   row_contrib[k] = sum_l a[k,l] * b[p[k], p[l]]  -- no, this isn't quite right for the swap.
        
        # Let's just do it directly in O(n)
        # Delta = sum over all affected terms
        # Affected terms: those where first index is i or j, or second index is i or j.
        
        delta = 0
        
        # For each k not in {i, j}:
        #   Term (k, i): a[k,i] * (b[p[k], y] - b[p[k], x])
        #   Term (k, j): a[k,j] * (b[p[k], x] - b[p[k], y])
        #   Term (i, k): a[i,k] * (b[y, p[k]] - b[x, p[k]])
        #   Term (j, k): a[j,k] * (b[x, p[k]] - b[y, p[k]])
        
        # For k in {i, j}:
        #   Term (i, i): a[i,i] * (b[y,y] - b[x,x])
        #   Term (j, j): a[j,j] * (b[x,x] - b[y,y])
        #   Term (i, j): a[i,j] * (b[y,x] - b[x,y])
        #   Term (j, i): a[j,i] * (b[x,y] - b[y,x])
        
        # Let's vectorize the k not in {i,j} part
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        ks = np.where(mask)[0]
        
        # sum_k a[k,i] * (b[p[k], y] - b[p[k], x])
        # = sum_k a[k,i] * b[p[k], y] - sum_k a[k,i] * b[p[k], x]
        
        pk = p[ks]
        
        term1 = np.sum(a[ks, i] * (b[pk, y] - b[pk, x]))
        term2 = np.sum(a[ks, j] * (b[pk, x] - b[pk, y]))
        term3 = np.sum(a[i, ks] * (b[y, pk] - b[x, pk]))
        term4 = np.sum(a[j, ks] * (b[x, pk] - b[y, pk]))
        
        # Diagonal and cross terms
        term5 = a[i, i] * (b[y, y] - b[x, x])
        term6 = a[j, j] * (b[x, x] - b[y, y])
        term7 = a[i, j] * (b[y, x] - b[x, y])
        term8 = a[j, i] * (b[x, y] - b[y, x])
        
        delta = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
        return int(delta)
    
    # 1. Initial Heuristic: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = initial_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Tabu Search with efficient delta computation
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # Tabu list: store (i, j, iteration) for recent swaps
    tabu_list = []
    tabu_tenure = 20  # Number of iterations to keep in tabu list
    best_iter = 0
    
    # Parameters for acceptance
    T = 500.0
    T_min = 1.0
    alpha = 0.999
    
    iteration = 0
    
    while True:
        if time.time() - start_time > timeout:
            break
        
        # Pick random i, j
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        # Check tabu
        is_tabu = False
        for (ti, tj, titer) in tabu_list:
            if (ti == i and tj == j) or (ti == j and tj == i):
                if iteration - titer < tabu_tenure:
                    is_tabu = True
                    break
        
        # Compute delta
        delta = swap_delta(p, i, j)
        
        # Acceptance criterion
        accept = False
        if delta < 0:
            accept = True
        elif not is_tabu:
            # Accept if not tabu and probabilistic acceptance
            if random.random() < np.exp(-delta / T):
                accept = True
        
        if accept:
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            iteration += 1
            
            # Update tabu list
            tabu_list.append((i, j, iteration))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            
            # Update best
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                best_iter = iteration
        else:
            # Reject: don't swap
            iteration += 1
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Occasionally restart tabu list if stuck
        if iteration - best_iter > 500:
            # Reset to best solution
            p = best_p.copy()
            current_cost = best_cost
            tabu_list = []
            T = 500.0
            best_iter = iteration
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
