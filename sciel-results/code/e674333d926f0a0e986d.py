import time
import sys
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    MAX_VAL = 10000
    MAX_SUM = 20000
    
    # Precompute smallest prime factors up to MAX_SUM
    spf = list(range(MAX_SUM + 1))
    for i in range(2, isqrt(MAX_SUM) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Get prime factors as a set for a number
    def prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # Precompute prime factor sets for all sums 2..MAX_SUM
    # And for all numbers 1..MAX_VAL
    # To save memory and time, we can just compute on the fly or cache.
    # Caching is better for repeated checks.
    
    # We will try increasing sets of primes.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    best_result = None
    best_prime_count = float('inf')
    
    # Try combinations of the first few primes
    # Start with small sets
    # 2 primes: {2,3}
    # 3 primes: {2,3,5}, {2,3,7}, etc.
    # We prioritize sets containing 2, as it's the most common factor in sums.
    
    import itertools
    
    found = False
    
    # Try k=2, 3, 4, 5 primes
    for k in range(2, 7):
        if time.time() > deadline:
            break
            
        # Generate combinations of primes
        # To bias towards smaller primes, we iterate through combinations of the list
        # But we should try {2,3} first, then {2,3,5}, etc.
        # A simple way: iterate through all combinations, but sort by max prime or sum?
        # Or just hardcode a priority order for the first few.
        
        if k == 2:
            prime_sets = [{2, 3}, {2, 5}, {2, 7}, {3, 5}]
        elif k == 3:
            prime_sets = [{2, 3, 5}, {2, 3, 7}, {2, 5, 7}, {2, 3, 11}, {3, 5, 7}]
        elif k == 4:
            prime_sets = [{2, 3, 5, 7}, {2, 3, 5, 11}, {2, 3, 7, 11}, {2, 5, 7, 11}]
        elif k == 5:
            prime_sets = [{2, 3, 5, 7, 11}, {2, 3, 5, 7, 13}, {2, 3, 5, 11, 13}]
        elif k == 6:
            prime_sets = [{2, 3, 5, 7, 11, 13}, {2, 3, 5, 7, 11, 17}]
        else:
            # Fallback to generic small primes
            prime_sets = [set(primes_list[:k])]
            
        for p_set in prime_sets:
            if time.time() > deadline:
                break
                
            # 1. Find all numbers <= MAX_VAL that are "smooth" with respect to p_set
            # i.e., their prime factors are a subset of p_set
            candidates = []
            for n in range(1, MAX_VAL + 1):
                pf = prime_factors(n)
                if pf.issubset(p_set):
                    candidates.append(n)
            
            if len(candidates) < 12:
                continue
            
            # 2. Build graph
            # n_cand = len(candidates)
            # We need to check if sum of two candidates is also smooth w.r.t p_set
            
            n_cand = len(candidates)
            
            # Optimization: If n_cand is huge, limit search?
            # For {2,3}, count is small. For {2,3,5,7}, count is larger.
            # Let's cap candidates to the first 500 or so to keep graph size manageable?
            # Or just use all if < 1000.
            if n_cand > 1000:
                # Heuristic: smaller numbers are more likely to form small sums
                # which are more likely to be smooth?
                # Actually, density of smooth numbers decreases.
                # Let's just take the first 1000.
                candidates = candidates[:1000]
                n_cand = len(candidates)
            
            # Precompute adjacency
            # adj[i][j] = True if candidates[i] + candidates[j] is smooth w.r.t p_set
            # Use a list of sets for efficiency
            adj = [set() for _ in range(n_cand)]
            
            for i in range(n_cand):
                ci = candidates[i]
                for j in range(i + 1, n_cand):
                    cj = candidates[j]
                    s = ci + cj
                    if s > MAX_SUM:
                        continue
                    pf_s = prime_factors(s)
                    if pf_s.issubset(p_set):
                        adj[i].add(j)
                        adj[j].add(i)
            
            # 3. Find clique of size 12
            # Backtracking search
            # Order candidates by degree? Descending degree helps find cliques faster.
            # But we need to keep track of original values.
            
            # Let's use a simple recursive backtracker with pruning.
            # State: current clique (list of indices), start index for next candidate
            # To speed up, sort candidates by degree descending.
            
            degrees = [len(adj[i]) for i in range(n_cand)]
            # Create list of (degree, index)
            indexed = list(enumerate(range(n_cand)))
            indexed.sort(key=lambda x: degrees[x[1]], reverse=True)
            # Re-map? Or just iterate in this order?
            # If we iterate in this order, we need to be careful with "start index" logic
            # because the order is permuted.
            # Better: keep original indices, but visit them in sorted order.
            # We can create a `visit_order` list.
            visit_order = [idx for _, idx in indexed]
            
            # To make "start index" work with permutation, we can just use a set of "forbidden" or "available"
            # Or simpler: standard backtracking on the original indices, but checking in sorted order?
            # No, standard clique finding works best if we process vertices in a specific order
            # and only consider vertices with higher index in that order.
            
            # Let's map original indices to positions in visit_order.
            # pos_in_order[orig_idx] = new_pos
            pos_in_order = [0] * n_cand
            for new_pos, orig_idx in enumerate(visit_order):
                pos_in_order[orig_idx] = new_pos
            
            # Rebuild adj in the new order? Or just check pos_in_order?
            # It's cleaner to create a new adjacency list `new_adj` for the permuted vertices.
            new_adj = [set() for _ in range(n_cand)]
            for i in range(n_cand):
                orig_i = visit_order[i]
                for j in adj[orig_i]:
                    orig_j = j
                    if pos_in_order[orig_j] > i: # Only add edge to higher position to avoid duplicates in search?
                         # Actually, for clique finding, we usually look at neighbors with index > current.
                         # So we just need to know if a neighbor exists.
                         pass
            
            # Let's just use the original adj and pos_in_order in the search.
            # Search for clique of size 12.
            
            def find_clique(current_clique, start_pos):
                if len(current_clique) == 12:
                    return current_clique[:]
                if start_pos >= n_cand:
                    return None
                
                # Pruning: if remaining vertices + current size < 12, return None
                if len(current_clique) + (n_cand - start_pos) < 12:
                    return None
                
                # Iterate from start_pos to end
                for pos in range(start_pos, n_cand):
                    orig_idx = visit_order[pos]
                    
                    # Check if orig_idx is connected to all in current_clique
                    compatible = True
                    for c_idx in current_clique:
                        # c_idx is original index
                        if orig_idx not in adj[c_idx]:
                            compatible = False
                            break
                    
                    if compatible:
                        current_clique.append(orig_idx)
                        result = find_clique(current_clique, pos + 1)
                        if result:
                            return result
                        current_clique.pop()
                        
                return None
            
            clique_indices = find_clique([], 0)
            
            if clique_indices:
                result_numbers = [candidates[i] for i in clique_indices]
                # Verify distinctness (guaranteed by set construction) and bounds
                if len(set(result_numbers)) == 12 and all(1 <= x <= 10000 for x in result_numbers):
                    # Verify prime count
                    primes_found = set()
                    for i in range(12):
                        for j in range(i + 1, 12):
                            primes_found |= prime_factors(result_numbers[i] + result_numbers[j])
                    
                    if len(primes_found) < best_prime_count:
                        best_prime_count = len(primes_found)
                        best_result = result_numbers
                        found = True
                        # Since we are trying smaller prime sets first, 
                        # if we found a solution with k primes, we can stop 
                        # if k is the minimum possible? 
                        # Minimum is 1? (Only prime 2). 
                        # If we found 2, we should check if 1 is possible.
                        # But 1 is impossible for 12 distinct ints.
                        # So if we find a solution with prime count 2, it's likely optimal or near optimal.
                        # However, we iterate k=2, then k=3. 
                        # If k=2 yields a solution, prime count is 2. 
                        # If k=3 yields a solution, prime count is 3.
                        # So finding a solution for k=2 is better than k=3.
                        # We can break out of all loops if we find a solution for k=2?
                        # Yes, because any solution for k=3 has >=3 primes, while solution for k=2 has exactly 2 (subset of {2,3}).
                        # Wait, the union of prime factors of the sums is the metric.
                        # If sums only have factors from {2,3}, the union is a subset of {2,3}.
                        # It could be {2} or {2,3}.
                        # If we find a solution where all sums are even (only 2), that's 1 prime.
                        # But we assumed sums have factors in {2,3}.
                        # So the actual count might be 1 or 2.
                        # We should calculate the actual count.
                        # If we find a solution for k=2, the max count is 2.
                        # If we find a solution for k=3, the max count is 3.
                        # So we should prioritize k=2 solutions.
                        # If we find one, we can return it immediately? 
                        # Only if we are sure no k=1 solution exists (impossible) 
                        # and no other k=2 solution has count 1 (which is better than 2).
                        # Actually, if we find a k=2 solution with count 2, 
                        # and we haven't checked all k=2 sets, maybe another k=2 set gives count 1?
                        # Count 1 means all sums are powers of 2.
                        # This is a very specific structure.
                        # Let's just return the first valid solution found for the smallest k.
                        # Because any solution for k=2 has count <= 2.
                        # Any solution for k=3 has count <= 3.
                        # So a k=2 solution is always better or equal to a k=3 solution.
                        # And within k=2, we want min count.
                        # Let's collect all k=2 solutions? No, time limit.
                        # Let's just return the first one found. It's likely optimal.
                        break
            
            if found:
                break
                
        if found:
            break
    
    if best_result is None:
        # Fallback: Random or simple heuristic
        # Just return 1..12?
        # 1+2=3, 1+3=4(2), 2+3=5. Primes: 2,3,5...
        # Let's try a greedy construction for a small set like {2,3,5}
        # But if the main loop failed, it means no clique of 12 was found in the smooth sets.
        # This might happen if the graph is too sparse.
        # Let's try a different candidate set: all numbers, but filter by sum property?
        # That's expensive.
        
        # Fallback: Return a known good set if we have one, or generate one.
        # Example: Powers of 2? 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048.
        # Sums: 1+2=3 (3), 1+4=5 (5), 2+4=6 (2,3).
        # Primes: 2, 3, 5, 7 (1+8=9? No 1+8=9=3^2. 1+16=17).
        # So powers of 2 give many primes.
        
        # Let's try: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24.
        # All even. Sums are even.
        # 2+4=6 (2,3), 2+6=8 (2), 2+8=10 (2,5).
        # Primes: 2, 3, 5, 7 (2+12=14), 11 (2+20=22), 13 (2+24=26), 17 (4+24=28? No 4+22=26).
        # 10+12=22 (11).
        # This set has primes {2,3,5,7,11,13,17,19}.
        
        # Let's just return the first 12 even numbers.
        return list(range(2, 25, 2))
        
    return best_result

if __name__ == "__main__":
    construct()
