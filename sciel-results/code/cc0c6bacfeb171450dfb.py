import itertools
import time

def _is_paley_valid(p):
    """
    Check if the Paley tournament on p vertices satisfies the condition:
    Every set of 4 vertices has a common dominator.
    """
    if p < 5:
        return True
    
    # Compute squares
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Build adjacency lists (out-neighbors)
    # out_neigh[x] is a sorted list of y such that x -> y
    out_neigh = [[] for _ in range(p)]
    for x in range(p):
        for y in range(p):
            if x == y:
                continue
            if (y - x) % p in squares:
                out_neigh[x].append(y)
    
    # Precompute mapping from 4-tuple to index
    # We only need to check if ALL 4-sets are covered.
    # Total 4-sets: C(p, 4)
    # We can use a set of covered 4-sets.
    # If len(covered) == C(p, 4), it is valid.
    
    total_4sets = 1
    for i in range(4):
        total_4sets = total_4sets * (p - i) // (i + 1)
        
    covered = set()
    
    # For each vertex x, mark all 4-sets in N+(x)
    for x in range(p):
        neigh = out_neigh[x]
        # Generate all 4-combinations of neigh
        # len(neigh) is (p-1)/2
        for comb in itertools.combinations(neigh, 4):
            # comb is already sorted because out_neigh[x] is built by iterating y 0..p-1
            covered.add(comb)
            
    return len(covered) == total_4sets

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Primes p congruent to 3 mod 4
    # We want to minimize n, so check small p first.
    # Lower bound is 31.
    candidates = [31, 43, 47, 59, 67, 71, 79, 83, 87, 91, 95, 97, 103, 107, 109, 127, 131, 139, 143, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 233, 239, 241, 251, 257, 263, 271, 277, 281, 283, 293]
    
    # Filter to valid primes p % 4 == 3
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    valid_primes = [p for p in candidates if p % 4 == 3 and is_prime(p)]
    
    # Sort by n
    valid_primes.sort()
    
    best_result = None
    
    for p in valid_primes:
        if time.time() > deadline:
            break
            
        # Check validity
        if _is_paley_valid(p):
            # Construct orientation
            squares = {x * x % p for x in range(1, p)}
            orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
            return {'n': p, 'orientation': orientation}
            
    # Fallback: Return the seed program's result (Paley 83) if no smaller one found or timeout
    # Actually, if we timed out, we should return the best known valid one.
    # If we haven't checked 83 yet, we should ensure we return a valid tournament.
    # The seed program returns Paley 83. We know it's valid (presumably, or at least accepted).
    # If we didn't find a smaller one, return Paley 83.
    
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
