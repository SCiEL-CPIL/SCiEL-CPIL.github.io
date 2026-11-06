def construct():
    import time
    from math import gcd
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 145
    
    N = 511
    weights = [0.0] + [1.0 / (N + 1 - a) for a in range(1, N + 1)]
    
    # Precompute prime factors as bitmasks
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    prime_index = {p: i for i, p in enumerate(primes)}
    masks = [0] * (N + 1)
    for a in range(1, N + 1):
        m = a
        mask = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask |= (1 << prime_index[m])
        masks[a] = mask
    
    # Build conflict graph using bitsets
    # conflict[a] is a set of numbers that conflict with a (not coprime)
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if masks[a] & masks[b]:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Sort by weight descending for better branching
    sorted_indices = list(range(1, N + 1))
    sorted_indices.sort(key=lambda x: weights[x], reverse=True)
    
    # Convert conflict sets to sorted lists for faster iteration
    conflict_lists = [sorted(conflict[a]) for a in range(N + 1)]
    
    # Branch and bound
    best_score = 0.0
    best_set = set()
    
    # Upper bound: sum of weights of all remaining candidates
    # We use a greedy upper bound based on the conflict structure
    
    def upper_bound(candidates, current_score):
        """Compute an upper bound on achievable score from candidates."""
        # Simple bound: sum of all weights
        # Better bound: account for conflicts by taking max weight from each conflict group
        # For now, use sum of weights as a loose bound
        return current_score + sum(weights[a] for a in candidates)
    
    def branch_and_bound(candidates, current_mask, current_set, current_score, depth=0):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        
        # Upper bound check
        ub = current_score + sum(weights[a] for a in candidates)
        if ub <= best_score:
            return
        
        # Choose the next vertex to branch on
        # Pick the one with highest weight among candidates
        # Or pick the one that conflicts with the most other candidates
        v = max(candidates, key=lambda x: weights[x])
        
        # Branch 1: Include v
        new_candidates = [a for a in candidates if a != v and a not in conflict[v]]
        new_set = current_set | {v}
        branch_and_bound(new_candidates, current_mask | masks[v], new_set, current_score + weights[v], depth + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude v
        new_candidates2 = [a for a in candidates if a != v]
        branch_and_bound(new_candidates2, current_mask, current_set, current_score, depth + 1)
    
    # Start with all numbers as candidates
    initial_candidates = sorted_indices[:]
    branch_and_bound(initial_candidates, 0, set(), 0.0)
    
    return list(best_set)
