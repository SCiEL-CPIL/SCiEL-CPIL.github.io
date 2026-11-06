import time
import random
from collections import defaultdict

def _is_smooth(n, primes):
    """Check if n is smooth wrt the given set of primes."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def _generate_smooth_numbers(max_val, primes):
    """Generate all numbers <= max_val that are smooth wrt primes."""
    smooth = set()
    primes_list = list(primes)
    
    def gen(current, idx):
        if current > max_val:
            return
        if current > 1:
            smooth.add(current)
        for i in range(idx, len(primes_list)):
            p = primes_list[i]
            if current * p > max_val:
                break
            gen(current * p, i)
            
    gen(1, 0)
    return smooth

def _find_clique(n_size, max_node, neighbors, deadline):
    """
    Find a clique of size n_size in the graph defined by neighbors.
    Nodes are 1..max_node.
    neighbors is a dict: node -> set of neighbors.
    """
    # Sort nodes by degree descending to prune faster
    nodes = list(range(1, max_node + 1))
    nodes.sort(key=lambda x: len(neighbors[x]), reverse=True)
    
    # We want to pick n_size nodes.
    # Use backtracking.
    current_clique = []
    
    # To speed up, we can use a heuristic: always try to add nodes that are connected to all in current_clique.
    
    def backtrack(start_idx, available):
        if time.time() > deadline:
            return None
        
        if len(current_clique) == n_size:
            return current_clique[:]
        
        if not available:
            return None
            
        # Heuristic: try nodes with higher degree within 'available'
        # But 'available' is a set. We can just iterate.
        # Optimization: if the number of available nodes is less than needed, fail.
        if len(available) < n_size - len(current_clique):
            return None
            
        for v in available:
            # Check if v can be added
            # v must be connected to all nodes in current_clique
            # But since we filter 'available' to only contain nodes connected to all in current_clique, 
            # this is implicitly true if we maintain 'available' correctly.
            
            current_clique.append(v)
            
            # New available: intersection of available (excluding v) and neighbors[v]
            new_available = available - {v}
            new_available &= neighbors[v]
            
            res = backtrack(0, new_available)
            if res is not None:
                return res
                
            current_clique.pop()
            
        return None

    # Initial available: all nodes
    # But we should probably not just iterate 1..max_node in backtrack, 
    # but rather pass a sorted list or set.
    # Using a set for 'available' is efficient for intersection.
    
    initial_available = set(nodes)
    return backtrack(0, initial_available)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 12
    max_val = 10000
    
    # List of prime sets to try, ordered by size
    # We want to minimize the union of primes.
    # If we use K with primes P_K, and sums are P-sum smooth, total primes = P_K U P_sum.
    # Usually we set P_sum to contain P_K.
    
    prime_sets_to_try = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
    ]
    
    # Candidate K values
    # K should have prime factors in the first prime set we try?
    # Actually, if we pick P_sum = {2,3,5,7}, and K=210 (primes 2,3,5,7), total is 4.
    # If we pick K=60 (primes 2,3,5), and P_sum={2,3,5,7}, total is 4.
    # If we pick K=30 (primes 2,3,5), and P_sum={2,3,5,7}, total is 4.
    # If we pick K=12 (primes 2,3), and P_sum={2,3,5,7}, total is 4.
    
    # Larger K means smaller max_m, which might make finding a clique harder or easier?
    # Smaller max_m means fewer nodes in graph. Easier to search, but fewer choices.
    # K=210 -> max_m=47.
    # K=60 -> max_m=166.
    # K=30 -> max_m=333.
    
    # Let's try K values that are highly composite or have small prime factors.
    k_candidates = [210, 60, 30, 12, 42, 84, 180, 360, 90, 120, 240, 420, 840]
    
    best_solution = None
    best_score = float('inf')
    
    for primes in prime_sets_to_try:
        if time.time() > deadline:
            break
            
        # Generate smooth numbers for sums
        # Max sum of multipliers depends on K.
        # We will calculate max_m for each K.
        
        for K in k_candidates:
            if time.time() > deadline:
                break
                
            # Check if primes(K) are subset of primes
            # If not, the total prime count will be larger.
            # We want to minimize total distinct primes.
            # Let's just compute the effective prime set: effective_primes = set(primes) U set(primes(K))
            # And check if we can find a clique using sums that are smooth wrt effective_primes?
            # No, the sums m_i+m_j don't need to be smooth wrt effective_primes.
            # The sums a_i+a_j = K(m_i+m_j).
            # Primes(a_i+a_j) = primes(K) U primes(m_i+m_j).
            # We want |primes(K) U primes(m_i+m_j)| to be small.
            # So we should restrict m_i+m_j to be smooth wrt a set P_sum.
            # Then total primes = |primes(K) U P_sum|.
            
            # Let's try to keep P_sum = primes (the current set we are iterating).
            # Then total primes = |set(primes) U set(primes(K))|.
            
            # Calculate primes of K
            k_primes = set()
            temp_k = K
            for p in primes:
                while temp_k % p == 0:
                    k_primes.add(p)
                    temp_k //= p
            if temp_k > 1:
                # K has a prime factor not in 'primes'
                # This increases the total count.
                # We can still proceed, but the score will be higher.
                pass
            
            total_primes_count = len(set(primes) | k_primes)
            
            # If this count is already >= best_score, skip
            if total_primes_count >= best_score:
                continue
                
            max_m = max_val // K
            if max_m < n:
                continue
                
            max_sum = 2 * max_m
            smooth_sums = _generate_smooth_numbers(max_sum, primes)
            
            # Build graph
            neighbors = defaultdict(set)
            for u in range(1, max_m + 1):
                for s in smooth_sums:
                    v = s - u
                    if 1 <= v <= max_m and v != u:
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        
            # Find clique
            # Limit search time per K
            clique = _find_clique(n, max_m, neighbors, deadline)
            
            if clique is not None:
                # Verify
                # Construct result
                result = [K * m for m in clique]
                # Check distinctness
                if len(set(result)) != n:
                    continue
                if any(x > max_val for x in result):
                    continue
                
                # Calculate actual score to be sure
                actual_primes = set()
                for i in range(n):
                    for j in range(i+1, n):
                        s = result[i] + result[j]
                        # Factor s
                        temp_s = s
                        for p in range(2, int(temp_s**0.5)+1):
                            if temp_s % p == 0:
                                actual_primes.add(p)
                                while temp_s % p == 0:
                                    temp_s //= p
                        if temp_s > 1:
                            actual_primes.add(temp_s)
                
                score = len(actual_primes)
                if score < best_score:
                    best_score = score
                    best_solution = result
                    
                # If we found a solution with the minimum possible primes for this set, 
                # we might want to try smaller prime sets? 
                # But we are iterating prime sets in increasing order of size.
                # So the first one we find is likely optimal for that prime set size.
                # However, a larger K with smaller prime set might yield better?
                # No, we are iterating primes sets first.
                # If we find a solution with 4 primes, that's likely the best we can do (since 3 primes 2,3,5 might not allow 12 distinct sums to be smooth).
                # Let's break out of loops if we find a very good score?
                # Minimum possible is 2? (Sums all powers of 2? Impossible for distinct positives).
                # 3? (Sums smooth wrt 2,3).
                # Let's just continue to see if we can find 3 primes.
                # But 4 is already very good.
                
                if best_score == 4:
                    # Hard to beat 4. 
                    # We can break out of K loop, but maybe try next prime set? No, next is larger.
                    # So we can break out of everything?
                    # Actually, maybe a different K with same primes gives a "cleaner" set? Score is same.
                    # Let's break to save time.
                    return result

    if best_solution is None:
        # Fallback
        # Just return some valid set
        # Use the current program's logic or a simple one
        # Simple: multiples of 2
        return list(range(2, 2*n+2, 2))
        
    return best_solution
