import time
import itertools
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    N = 12
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL
    
    # Generate primes up to MAX_SUM
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve(MAX_SUM)
    
    # Precompute smallest prime factor (SPF) for fast factorization
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def is_smooth(n, allowed_primes):
        # Check if all prime factors of n are in allowed_primes
        # allowed_primes should be a set for O(1) lookup
        while n > 1:
            p = spf[n]
            if p not in allowed_primes:
                return False
            while n % p == 0:
                n //= p
        return True

    def get_prime_count(n):
        count = 0
        while n > 1:
            p = spf[n]
            count += 1
            while n % p == 0:
                n //= p
        return count

    def find_clique(allowed_primes):
        # Generate smooth sums
        smooth_sums = []
        for s in range(2, MAX_SUM + 1):
            if is_smooth(s, allowed_primes):
                smooth_sums.append(s)
        
        # If there are too few smooth sums, it's unlikely to form a large clique
        # Each node needs to connect to N-1 others.
        # The number of potential edges is roughly N * (N-1) / 2 * (avg_degree_factor)
        
        # Build adjacency structure
        # We use bitsets (integers) for fast intersection checks in backtracking
        # Node i is represented by its index i.
        # adj[i] is a bitset where bit j is 1 if i and j are connected.
        
        adj = [0] * (MAX_VAL + 1)
        
        for s in smooth_sums:
            min_a = max(1, s - MAX_VAL)
            max_a = min(MAX_VAL, s - 1)
            for a in range(min_a, max_a + 1):
                b = s - a
                if b != a:
                    # Add edge a-b
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        
        # Calculate degrees
        degrees = [adj[i].bit_count() for i in range(1, MAX_VAL + 1)]
        
        # Filter candidates with degree >= N-1
        candidates = [i for i in range(1, MAX_VAL + 1) if degrees[i] >= N - 1]
        
        if len(candidates) < N:
            return None
            
        # Sort candidates by degree descending to find large cliques faster
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        
        # Limit search space to top M candidates to avoid explosion
        # If a clique exists, it's likely among high-degree nodes.
        # For N=12, top 500-1000 should be sufficient if a solution exists.
        M = 1000
        search_space = candidates[:M]
        
        # Map original node to index in search_space for faster bit operations?
        # No, we can just use the original node indices in the bitsets.
        
        found_clique = []
        
        def backtrack(start_idx, current_clique, common_neighbors):
            # common_neighbors is a bitset of nodes in search_space 
            # that are connected to all nodes in current_clique
            if len(current_clique) == N:
                found_clique.extend(current_clique)
                return True
            
            # Pruning: if not enough nodes left in common_neighbors
            if common_neighbors.bit_count() < N - len(current_clique):
                return False
                
            # Iterate through candidates in search_space starting from start_idx
            # We need to check if candidate is in common_neighbors
            
            # Optimization: Iterate over set bits in common_neighbors that are >= start_idx in search_space order?
            # Since search_space is sorted, we can just loop i from start_idx to end
            # and check if search_space[i] is in common_neighbors.
            
            for i in range(start_idx, len(search_space)):
                node = search_space[i]
                # Check if node is in common_neighbors
                if not (common_neighbors & (1 << node)):
                    continue
                
                # Add node to clique
                current_clique.append(node)
                
                # Update common neighbors: intersect with adj[node]
                new_common = common_neighbors & adj[node]
                
                # Recurse
                # The next start index is i+1 because we are picking from a sorted list 
                # and want to avoid permutations (clique is a set)
                if backtrack(i + 1, current_clique, new_common):
                    return True
                    
                current_clique.pop()
            
            return False
        
        # Initial common neighbors: all nodes in search_space
        # Note: search_space nodes are the candidates. 
        # We initialize common_neighbors to include all nodes in search_space.
        initial_common = 0
        for node in search_space:
            initial_common |= (1 << node)
            
        if backtrack(0, [], initial_common):
            return found_clique[:N]
        return None

    # Strategy: Try small prime sets first.
    # The number of distinct primes in the sums will be the size of the set P 
    # IF all sums are P-smooth. 
    # However, the question asks to minimize the number of distinct primes 
    # dividing the sums. It does NOT require ALL sums to be smooth over a fixed set P 
    # such that |P| is minimized. 
    # Wait, "union all prime factors and minimize the number of distinct primes".
    # This is equivalent to finding a set of numbers such that the set of primes 
    # appearing in any pair sum is as small as possible.
    # If we find a clique in the graph defined by P, then all pair sums are P-smooth.
    # The union of prime factors will be a subset of P.
    # So minimizing |P| such that a clique exists is a valid strategy to find a lower bound.
    
    # Let's try P = {2, 3}
    P1 = {2, 3}
    res = find_clique(P1)
    if res:
        return res
    
    if time.time() > deadline:
        return list(range(1, 13))
        
    # Let's try P = {2, 3, 5}
    P2 = {2, 3, 5}
    res = find_clique(P2)
    if res:
        return res
    
    if time.time() > deadline:
        return list(range(1, 13))

    # Let's try P = {2, 3, 5, 7}
    P3 = {2, 3, 5, 7}
    res = find_clique(P3)
    if res:
        return res
    
    if time.time() > deadline:
        return list(range(1, 13))
        
    # If all specific smooth searches fail, try a heuristic local search
    # Start with a random set and try to swap elements to reduce prime count
    
    import random
    
    def evaluate_set(nums):
        primes_found = set()
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                s = nums[i] + nums[j]
                # Factor s
                n = s
                while n > 1:
                    p = spf[n]
                    primes_found.add(p)
                    while n % p == 0:
                        n //= p
        return len(primes_found)
    
    best_set = list(range(1, 13))
    best_score = evaluate_set(best_set)
    
    # Local search
    current_set = best_set[:]
    current_score = best_score
    
    # Precompute prime factors for all possible sums? 
    # Too slow. Just evaluate on the fly.
    
    for _ in range(5000): # Iterations
        if time.time() > deadline:
            break
            
        # Pick a random index to replace
        idx = random.randint(0, N-1)
        # Pick a random replacement
        new_val = random.randint(1, MAX_VAL)
        if new_val in current_set:
            continue
            
        old_val = current_set[idx]
        current_set[idx] = new_val
        
        new_score = evaluate_set(current_set)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            # Revert
            current_set[idx] = old_val
            
    return best_set

# EVOLVE-BLOCK-END

