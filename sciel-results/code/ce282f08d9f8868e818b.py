import itertools
import math
import time

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    vals = list(set(values))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if _gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return (num_distinct, is_valid) for given A, B."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    if len(distinct) < 2:
        return 0, False
    if not _is_pairwise_coprime(list(distinct)):
        return 0, False
    return len(distinct), True

def construct():
    # Deadline guard
    deadline = time.time() + 50
    
    # Parity constraint: A odd, B even (or vice versa)
    # We try A odd, B even first
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    best_count = 0
    best_AB = None
    
    # Phase 1: Deterministic search on small numbers
    # We limit the search to the first 30 odds and 30 evens to keep it fast
    # 30 choose 3 is 4060. 4060 * 4060 is ~16 million. This might be too slow in Python.
    # Let's try a smaller range first, e.g., first 15 odds/evens.
    # 15 choose 3 is 455. 455 * 455 is ~200k. This is very fast.
    
    small_odds = odds[:20]
    small_evens = evens[:20]
    
    # Iterate over combinations
    for A in itertools.combinations(small_odds, 3):
        if time.time() > deadline:
            break
        for B in itertools.combinations(small_evens, 3):
            count, valid = _score(A, B)
            if valid and count > best_count:
                best_count = count
                best_AB = (list(A), list(B))
                if best_count == 9:
                    return best_AB
        if time.time() > deadline:
            break
            
    # Phase 2: If not found, expand range or use random search
    if best_AB is None or best_count < 9:
        # Use random search with a tighter loop
        import random
        random.seed(42)
        
        for _ in range(100000):
            if time.time() > deadline:
                break
            A = list(random.sample(odds, 3))
            B = list(random.sample(evens, 3))
            count, valid = _score(A, B)
            if valid and count > best_count:
                best_count = count
                best_AB = (A, B)
                if best_count == 9:
                    break
    
    if best_AB is None:
        # Fallback to a known valid solution (even if not max)
        return [[1, 7, 13], [4, 10, 16]]
        
    return best_AB
