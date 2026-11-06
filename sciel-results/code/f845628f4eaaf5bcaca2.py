import math
from itertools import combinations

def construct():
    MAX_VAL = 10000
    N = 12
    
    # Precompute smallest prime factor for numbers up to 2 * MAX_VAL
    MAX_SUM = 2 * MAX_VAL
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
                
    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # Strategy: 
    # 1. Try various base multipliers L.
    # 2. For each L, define a set of allowed primes P_allowed.
    # 3. Construct a graph of valid multipliers k such that k*L <= MAX_VAL.
    # 4. Edge (k1, k2) exists if prime factors of (k1+k2) are subset of P_allowed U prime_factors(L).
    # 5. Find a clique of size N in this graph.
    
    # Candidate L values: highly composite numbers
    # 210 = 2*3*5*7
    # 420 = 2^2*3*5*7
    # 840 = 2^3*3*5*7
    # 1260 = 2^2*3^2*5*7
    # 2520 = 2^3*3^2*5*7
    # 5040 = 2^4*3^2*5*7
    # 7560 = 2^3*3^3*5*7
    # 10080 > 10000 (but k can be 1)
    
    L_candidates = [210, 420, 840, 1260, 2520, 5040, 7560, 30, 60, 120, 360, 720, 180, 90, 45, 15, 10, 5, 2, 1]
    
    # We will try to find a clique.
    # To make it efficient, we'll use a simple backtracking clique finder.
    
    def find_clique_size_n(nodes, adj, n):
        """
        nodes: list of node indices
        adj: dict of adjacency lists
        n: target clique size
        Returns a list of node indices forming a clique, or None.
        """
        if n == 0:
            return []
        if not nodes:
            return None
        # Sort nodes by degree (heuristic)
        nodes_sorted = sorted(nodes, key=lambda x: len(adj[x]), reverse=True)
        
        # Recursive search
        # To avoid deep recursion and timeout, we limit branching or use a greedy check first.
        # Given the small size of N=12 and potential node count (~50 for L=210), 
        # a backtracking search with pruning is feasible if the graph is dense.
        
        def backtrack(current_clique, candidates):
            if len(current_clique) == n:
                return current_clique
            if not candidates:
                return None
            
            # Pruning: if remaining candidates + current < n, fail
            if len(current_clique) + len(candidates) < n:
                return None
            
            # Try adding the first candidate
            # We can iterate through candidates
            for i in range(len(candidates)):
                node = candidates[i]
                # Check if node is connected to all in current_clique
                if all(node in adj[c] for c in current_clique):
                    # New candidates must be connected to node and be after i (to avoid duplicates)
                    new_candidates = [c for c in candidates[i+1:] if c in adj[node]]
                    result = backtrack(current_clique + [node], new_candidates)
                    if result is not None:
                        return result
            return None
        
        return backtrack([], nodes_sorted)

    best_result = None
    best_prime_count = float('inf')
    
    # We'll iterate through L candidates
    # For each L, we'll try expanding the set of allowed primes.
    # Start with the primes of L.
    
    for L in L_candidates:
        max_k = MAX_VAL // L
        if max_k < N:
            continue
            
        base_primes = get_prime_factors(L)
        
        # Try sets of allowed primes.
        # Start with base_primes, then add small primes one by one.
        # Small primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
        
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        # We want to find a clique.
        # Let's try allowed_primes = base_primes U {small primes up to K}
        
        found_for_L = False
        for num_extra_primes in range(0, len(small_primes) + 1):
            allowed_primes = base_primes.union(set(small_primes[:num_extra_primes]))
            
            # Build graph
            # Nodes are k in 1..max_k
            # We only care about k such that L*k <= MAX_VAL
            nodes = list(range(1, max_k + 1))
            
            # Precompute sum factors for all pairs? 
            # max_k is at most 10000 (if L=1). 
            # If L=1, max_k=10000. Graph is 10000 nodes. Too big for naive clique.
            # If L=210, max_k=47. Graph is 47 nodes. Very small.
            # So we prioritize large L.
            
            if max_k > 200:
                # For small L, the graph is too big for simple clique search.
                # We can restrict nodes to a subset or use a different heuristic.
                # For now, skip very small L or use a greedy approach.
                # Let's just skip L < 100 to keep the graph small enough for backtracking.
                # Or we can sample nodes.
                # Let's stick to L >= 100 for the clique approach.
                continue
            
            adj = {k: set() for k in nodes}
            
            # Build adjacency
            # Optimization: precompute factors of sums
            # Sum range: 2 to 2*max_k
            # For L=210, max_k=47, sum range 2..94.
            
            for i in range(len(nodes)):
                k1 = nodes[i]
                for j in range(i+1, len(nodes)):
                    k2 = nodes[j]
                    s = k1 + k2
                    factors = get_prime_factors(s)
                    if factors.issubset(allowed_primes):
                        adj[k1].add(k2)
                        adj[k2].add(k1)
                        
            # Find clique of size N
            clique = find_clique_size_n(nodes, adj, N)
            
            if clique:
                # Convert back to a_i
                result = [L * k for k in clique]
                # Verify distinctness and bounds
                if len(set(result)) == N and all(1 <= x <= MAX_VAL for x in result):
                    # Calculate actual prime count
                    prime_set = set()
                    for i in range(N):
                        for j in range(i+1, N):
                            prime_set |= get_prime_factors(result[i] + result[j])
                    count = len(prime_set)
                    if count < best_prime_count:
                        best_prime_count = count
                        best_result = result
                        # If we found a very good solution, we can stop early?
                        # Minimum possible is 1 (all sums power of 2? Impossible for distinct positive ints? 
                        # 2,4,8... sums 6,10,12... factors 2,3,5. So >1.
                        # 2, 4, 8, 16, 32... sums have 3,5,17...
                        # Let's not stop too early, but 4-5 is good.
                        if count <= 4:
                            return result
                found_for_L = True
                # If we found a clique, we can try to improve by adding more primes?
                # No, adding more primes only allows more edges, potentially finding a clique if it wasn't found before.
                # But we iterate num_extra_primes from 0 up.
                # So if we found a clique with few extra primes, that's likely optimal for this L.
                # We can break inner loop? No, maybe a clique exists with more primes but allows a different L?
                # Actually, for a fixed L, if a clique exists with P, it exists with P U {p}.
                # So the first found clique (smallest num_extra_primes) is best for this L.
                break
                
    if best_result:
        return best_result
        
    # Fallback: Random search if clique approach fails
    # This should rarely happen, but good for robustness.
    import random
    best_score = float('inf')
    best = None
    for _ in range(1000):
        cand = random.sample(range(1, MAX_VAL+1), N)
        prime_set = set()
        for i in range(N):
            for j in range(i+1, N):
                prime_set |= get_prime_factors(cand[i] + cand[j])
        score = len(prime_set)
        if score < best_score:
            best_score = score
            best = cand
    return best
