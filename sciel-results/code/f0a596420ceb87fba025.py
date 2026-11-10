"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs, reverse=True)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # 5s safety margin
    
    n = 5040
    target = 4979
    divs = _get_divisors(n)
    # divs is sorted descending
    
    # Greedy fallback (same as original but using precomputed divisors)
    remainder = target
    greedy_chosen = []
    for d in divs:
        if d <= remainder:
            greedy_chosen.append(d)
            remainder -= d
    if remainder != 0:
        # Should not happen, but safety
        return greedy_chosen
    
    best_solution = greedy_chosen
    
    # Try to find a solution with fewer divisors
    # Enumerate combinations by increasing size
    max_k = len(best_solution)
    
    for k in range(1, max_k):
        if time.time() > deadline:
            break
        # Check all combinations of size k
        found = False
        for combo in combinations(divs, k):
            if time.time() > deadline:
                found = False
                break
            s = sum(combo)
            if s == target:
                best_solution = list(combo)
                found = True
                break
            # Early termination: if sum already exceeds target, skip
            # (divs is sorted descending, so combos are generated in some order)
            # Actually combinations generates in lexicographic order of indices,
            # not by sum. We can't easily prune here without more structure.
        if found:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
