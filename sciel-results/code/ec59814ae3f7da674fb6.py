# EVOLVE-BLOCK-START
def construct():
    import math
    from math import gcd
    
    n = 512
    # Precompute weights
    weights = [0.0] * (n)
    for a in range(1, n):
        weights[a] = 1.0 / (n - a)
        
    # Precompute conflicts: conflicts[a] is a set of b such that gcd(a,b) > 1
    # We only care about b in 1..511
    conflicts = [set() for _ in range(n)]
    for a in range(1, n):
        for b in range(a + 1, n):
            if gcd(a, b) > 1:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Initialize with 1 and all primes
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
        
    selected = set()
    selected.add(1)
    for p in range(2, n):
        if is_prime(p):
            selected.add(p)
            
    # Hill Climbing with Multi-Swaps
    improved = True
    while improved:
        improved = False
        # Iterate through candidates
        # We can iterate in order of potential gain or just 1..511
        # Sorting by weight might help find big jumps faster, but 1..511 is fine
        for a in range(1, n):
            if a in selected:
                continue
                
            # Find conflicts with current selected set
            c_set = selected.intersection(conflicts[a])
            
            if not c_set:
                # Can add directly
                # Gain is weights[a] which is always > 0
                selected.add(a)
                improved = True
                break # Restart
            else:
                # Need to remove c_set
                cost = sum(weights[b] for b in c_set)
                gain = weights[a] - cost
                if gain > 1e-9:
                    # Perform swap
                    selected.difference_update(c_set)
                    selected.add(a)
                    improved = True
                    break # Restart
                    
    return list(selected)
# EVOLVE-BLOCK-END
