"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _get_candidates():
    """Generate candidate numbers: primes and prime powers p^k where p^k <= 256."""
    candidates = set()
    for p in range(2, 257):
        if _is_prime(p):
            pk = p
            while pk <= 256:
                candidates.add(pk)
                pk *= p
    return sorted(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    candidates = _get_candidates()
    n = len(candidates)
    
    # Precompute conflict matrix: conflict[i][j] = True if candidates[i] divides candidates[j] * candidates[k] for some k != i,j
    # Actually, the condition is: for every three DISTINCT members a, b, c, a does NOT divide b*c.
    # So if a divides b*c, we cannot have all three a, b, c in the set.
    # This is a 3-uniform hypergraph independence problem.
    # For greedy: we want to add numbers such that no triple (a,b,c) in the set has a | b*c.
    
    # Let's build a function to check if a new number x can be added to set S.
    # x can be added if for all a,b in S with a != b, a != x, b != x, it is NOT the case that a | x*b or x | a*b or b | a*x.
    # Wait, the condition is symmetric in the sense that any of the three could be the divisor.
    # If we add x to S, we need to check triples (x, a, b) for all a,b in S, a != b.
    # The triple (x, a, b) is invalid if x | a*b OR a | x*b OR b | x*a.
    
    def can_add(x, S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                if x % 1 == 0:  # always true, just to structure
                    pass
                # Check if x divides a*b
                if (a * b) % x == 0:
                    return False
                # Check if a divides x*b
                if (x * b) % a == 0:
                    return False
                # Check if b divides x*a
                if (x * a) % b == 0:
                    return False
        return True

    # Greedy approach: sort candidates by some heuristic.
    # Numbers with larger prime factors might be safer? Or smaller numbers?
    # Let's try sorting by number of divisors descending, or just by value.
    # Actually, primes are safest. Prime powers p^k are also quite safe.
    # Let's try adding in order of increasing "conflict potential".
    # A simple heuristic: sort by value. Primes and small prime powers will be added first.
    
    candidates_sorted = sorted(candidates)
    
    # Try multiple orderings and keep the best
    best_set = set()
    orderings = [
        lambda c: c,  # ascending
        lambda c: -c,  # descending
        lambda c: (c.bit_length(), c),  # by bit length then value
    ]
    
    for ordering in orderings:
        S = set()
        candidates_ordered = sorted(candidates, key=ordering)
        for x in candidates_ordered:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        if len(S) > len(best_set):
            best_set = S.copy()
        if time.time() > deadline:
            break
    
    # Convert to list of Python ints
    result = [int(x) for x in best_set]
    return sorted(result)

# EVOLVE-BLOCK-END
