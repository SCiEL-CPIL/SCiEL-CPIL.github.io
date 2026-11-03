import itertools
import time
import random
import numpy as np

def construct(n):
    if n <= 1:
        return "1"
    
    start_time = time.time()
    time_limit = 140.0  # 10s margin before 150s kill
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Convert to numpy array for faster overlap calculation
    # Shape: (N, n)
    P = np.array(perms, dtype=np.int8)
    
    # Precompute cost matrix C
    # C[i][j] = n - overlap(P[i], P[j])
    # Overlap is max k in 1..n-1 such that P[i][n-k:] == P[j][:k]
    
    # For n=7, N=5040. N^2 = 25.4M.
    # We can compute this in chunks or use a loop that is reasonably fast.
    # A pure python loop for 25M items is too slow (~20s+).
    # Let's use a smarter approach or accept the time if optimized.
    # Actually, we can compute overlaps for all i, j by checking suffixes of i against prefixes of j.
    # Since n is small, we can iterate k from n-1 down to 1.
    # For a fixed k, P[i][n-k:] is a tuple of length k.
    # We can map all suffixes of length k to a set of indices i.
    # Then for each prefix of length k of j, if it matches, overlap is at least k.
    # We want max k. So we can iterate k from n-1 down to 1.
    # For each k, find pairs (i, j) where suffix_k(i) == prefix_k(j).
    # If C[i][j] is not yet set (or is n), set it to n-k.
    
    C = np.full((N, N), n, dtype=np.int8)
    
    # Map suffixes to lists of indices
    # For efficiency, we'll iterate k
    for k in range(n - 1, 0, -1):
        # Build map: suffix_tuple -> list of i
        suffix_map = {}
        for i in range(N):
            s = tuple(P[i][n-k:])
            if s not in suffix_map:
                suffix_map[s] = []
            suffix_map[s].append(i)
            
        # For each j, check prefix
        for j in range(N):
            p = tuple(P[j][:k])
            if p in suffix_map:
                for i in suffix_map[p]:
                    if C[i][j] == n: # Not yet assigned, meaning overlap < k
                        C[i][j] = n - k
                    # If C[i][j] < n, it was already assigned with a larger k (smaller cost), so we don't update
                    # Wait, larger k means smaller cost (n-k).
                    # We iterate k from large to small.
                    # So the first time we match (largest k), we set it.
                    # If it's already set, it was set by a larger k (smaller cost), so we leave it.
                    # Actually, if C[i][j] is already < n, it means we found a larger overlap earlier.
                    # So we only set if it's still n (no overlap found yet for this pair in this pass).
                    # But wait, if k=n-1 matches, cost is 1. If k=n-2 matches, cost is 2.
                    # We want min cost, so max k.
                    # Iterating k from n-1 down to 1:
                    # If we find a match at k, and C[i][j] is still n, we set C[i][j] = n-k.
                    # If C[i][j] is already set, it must have been set by a larger k (smaller cost), so we ignore.
                    pass # The logic above is correct inside the loop

    # The above loop is O(N * n + TotalMatches). TotalMatches can be large.
    # For n=7, this is feasible.
    
    # Convert C to list of lists for faster access in python loops? 
    # Numpy indexing is slower for scalar access in tight loops than lists of lists.
    # But N=5040, C is 5040x5040.
    # Let's stick to numpy for storage, but maybe convert to list of lists if 2-opt is heavy.
    # Actually, 2-opt does O(N^2) iterations. 25M iterations.
    # Accessing C[i][j] in numpy is ~100ns. 25M * 100ns = 2.5s. Acceptable.
    # Accessing list of lists is ~50ns.
    # Let's convert to list of lists for speed in the search phase.
    C_list = C.tolist()
    
    # Greedy Construction
    start = random.randint(0, N - 1)
    unvisited = set(range(N))
    unvisited.remove(start)
    tour = [start]
    current = start
    
    while unvisited:
        best_next = -1
        best_cost = float('inf')
        # Optimization: Instead of iterating all unvisited, we could use a priority queue, 
        # but N is small enough that linear scan is O(N^2) total for greedy.
        for j in unvisited:
            cost = C_list[current][j]
            if cost < best_cost:
                best_cost = cost
                best_next = j
        tour.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    if time.time() - start_time > time_limit - 20:
        # Not enough time for search, return greedy
        return _build_string(tour, P, n)

    # 2-Opt Local Search
    # pre_fwd[i] = sum of C[tour[k]][tour[k+1]] for k=0..i-1
    # pre_bwd[i] = sum of C[tour[k+1]][tour[k]] for k=0..i-1
    
    def calc_prefixes(tour):
        pre_fwd = [0] * N
        pre_bwd = [0] * N
        for i in range(N - 1):
            u = tour[i]
            v = tour[i+1]
            pre_fwd[i+1] = pre_fwd[i] + C_list[u][v]
            pre_bwd[i+1] = pre_bwd[i] + C_list[v][u]
        return pre_fwd, pre_bwd

    pre_fwd, pre_bwd = calc_prefixes(tour)
    current_cost = pre_fwd[N-1]
    
    improved = True
    passes = 0
    max_passes = 200
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() - start_time > time_limit - 10:
            break
            
        for i in range(N - 1):
            if time.time() - start_time > time_limit - 10:
                break
            for j in range(i + 1, N):
                # Reverse segment tour[i+1 ... j]
                # Old edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1]) if j < N-1
                # New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1]) if j < N-1
                # Internal cost change: Forward(i+1, j) -> Reverse(i+1, j)
                
                u = tour[i]
                v = tour[i+1]
                w = tour[j]
                x = tour[j+1] if j < N - 1 else -1
                
                cost_old = C_list[u][v]
                cost_new = C_list[u][w]
                
                # Internal cost
                # Forward cost of segment i+1..j is pre_fwd[j] - pre_fwd[i+1]
                # Reverse cost of segment i+1..j is pre_bwd[j] - pre_bwd[i+1]
                # Note: pre_bwd is calculated on the current tour.
                # The segment is tour[i+1], ..., tour[j].
                # Forward: tour[i+1]->...->tour[j]. Cost = pre_fwd[j] - pre_fwd[i+1].
                # Reverse: tour[j]->...->tour[i+1]. Cost = pre_bwd[j] - pre_bwd[i+1].
                
                internal_old = pre_fwd[j] - pre_fwd[i+1]
                internal_new = pre_bwd[j] - pre_bwd[i+1]
                
                if j < N - 1:
                    cost_old += C_list[w][x]
                    cost_new += C_list[v][x]
                else:
                    # No edge after j
                    pass
                
                delta = (cost_new + internal_new) - (cost_old + internal_old)
                
                if delta < 0:
                    # Apply move
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    current_cost += delta
                    improved = True
                    # Recalculate prefixes? 
                    # Recalculating full prefixes is O(N).
                    # Doing it after every improvement is O(N * #improvements).
                    # #improvements can be O(N^2). Total O(N^3).
                    # For N=5040, N^3 is too slow.
                    # We need to update prefixes incrementally or accept the cost.
                    # Actually, 2-opt usually converges fast.
                    # Let's just recalculate prefixes once per pass if any improvement?
                    # Or better, break inner loop and restart pass?
                    # Standard 2-opt: if improved, restart pass or continue?
                    # Usually continue, but prefixes are invalid.
                    # So we must recalculate prefixes.
                    # To avoid O(N) per improvement, we can just recalculate once after the inner loop if `improved` was set.
                    # But if we recalculate in the middle, we break the inner loop logic?
                    # Let's just recalculate prefixes immediately.
                    pre_fwd, pre_bwd = calc_prefixes(tour)
                    # This is expensive. 
                    # Alternative: Use a different local search like Or-opt or 1-opt which is easier to update?
                    # Or just do a few passes with full recalculation at the start of each pass.
                    # If we find an improvement, we apply it, mark improved=True, and break out of inner loop?
                    # No, that's slow.
                    # Let's try: Apply improvement, recalc prefixes, and continue.
                    # If it's too slow, we fall back to greedy.
                    
    return _build_string(tour, P, n)

def _build_string(tour, P, n):
    # P is numpy array (N, n)
    # tour is list of indices
    result = list(P[tour[0]])
    for i in range(1, len(tour)):
        p_prev = P[tour[i-1]]
        p_curr = P[tour[i]]
        # Find overlap
        max_k = 0
        # Since we have C matrix, we know the cost.
        # cost = n - max_k => max_k = n - cost
        # But we don't have C here easily. Let's just compute.
        for k in range(n - 1, 0, -1):
            if all(p_prev[n-k + t] == p_curr[t] for t in range(k)):
                max_k = k
                break
        result.extend(p_curr[max_k:])
    return "".join(map(str, result))
