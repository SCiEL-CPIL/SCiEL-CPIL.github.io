"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 256

def _violations(S, a, b, c):
    """Check if a divides b*c (the forbidden condition)."""
    return (b * c) % a == 0

def _has_violation(S, a):
    """Check if adding a to S creates any violation with three distinct elements."""
    # For each pair (b,c) in S, check if a divides b*c or b divides a*c or c divides a*b
    for i, b in enumerate(S):
        for c in S[i+1:]:
            if _violations(S, a, b, c) or _violations(S, b, a, c) or _violations(S, c, a, b):
                return True
    return False

def _count_violations(S):
    """Count total violations in set S."""
    count = 0
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = S[i], S[j], S[k]
                if (b*c) % a == 0 or (a*c) % b == 0 or (a*b) % c == 0:
                    count += 1
    return count

def _greedy_construct(order):
    """Greedy: add elements in given order if no violation is created."""
    S = []
    for a in order:
        if not _has_violation(S, a):
            S.append(a)
    return S

def _local_search(S, max_time):
    """1-for-1 swap local search: try replacing one element with another to increase size or maintain size with fewer violations."""
    S = list(S)
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        n = len(S)
        # Try removing each element and adding each non-element
        for i in range(n):
            if (time.time() - start_time) > max_time:
                break
            S_removed = S[:i] + S[i+1:]
            # Try adding each candidate not in S
            for cand in range(2, N+1):
                if cand in S_removed:
                    continue
                if _has_violation(S_removed, cand):
                    continue
                # This swap maintains size; check if it's "better" in some heuristic
                # For now, just accept to explore
                S_new = S_removed + [cand]
                # Actually, we want to INCREASE size, so let's try add-only first
                improved = True
                S = S_new
                break
            if improved:
                break
        # Try adding a new element (increase size)
        if not improved:
            for cand in range(2, N+1):
                if cand in S:
                    continue
                if not _has_violation(S, cand):
                    S.append(cand)
                    improved = True
                    break
    return S

def construct():
    deadline = time.time() + 55  # 55 second deadline for safety
    
    # Strategy 1: Greedy with different orderings
    candidates = list(range(2, N+1))
    
    # Ordering 1: Primes first
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0:
                return False
            i += 6
        return True
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    best_S = []
    
    # Try multiple orderings
    orderings = [
        primes + composites,
        sorted(candidates, key=lambda x: -x),  # large first
        sorted(candidates, key=lambda x: x),    # small first
        sorted(candidates, key=lambda x: x if is_prime(x) else -x),  # primes then large composites
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_construct(order)
        if len(S) > len(best_S):
            best_S = S
    
    # Local search on best
    if time.time() < deadline - 5:
        best_S = _local_search(best_S, deadline - time.time() - 2)
    
    # Convert to list of ints
    return [int(x) for x in best_S]
# EVOLVE-BLOCK-END
