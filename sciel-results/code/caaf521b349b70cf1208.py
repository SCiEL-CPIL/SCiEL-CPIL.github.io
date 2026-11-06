import time
import math
import random

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    We store them as a sorted list of tuples (a, b, c) where a < b < c is NOT required.
    Actually, from 1/a = 1/b + 1/c, we can derive that a < b and a < c (since 1/a > 1/b and 1/a > 1/c).
    Also, without loss of generality, we can assume b <= c.
    The equation gives c = ab/(b-a). So for each pair (a, b) with b > a, if b-a divides ab, then c = ab/(b-a) is an integer.
    We need c in [1, N] and c distinct from a and b.
    """
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
    return triples

def _check_valid(S, triples):
    """Check if set S has no valid triple from the precomputed list."""
    S_set = set(S)
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    triples = _precompute_triples(N)
    
    # Strategy 1: Find the largest valid contiguous suffix {K, K+1, ..., 180}
    # Since if {K..N} is invalid, all {J..N} for J < K are also invalid (supersets)
    best_suffix = None
    for K in range(1, N + 1):
        S = set(range(K, N + 1))
        if _check_valid(S, triples):
            best_suffix = S
            break  # We want the smallest K (largest set)
    
    if best_suffix is None:
        best_suffix = set(range(1, N + 1))  # Fallback, though unlikely
    
    best_set = set(best_suffix)
    best_size = len(best_set)
    
    # Strategy 2: Greedy random search to try to add more elements
    # Start from the best suffix and try to add elements from 1 to K-1
    if time.time() < deadline - 5.0:
        # Try to greedily add elements in random order
        candidates = list(range(1, min(K, N + 1)))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if x not in best_set:
                new_set = best_set | {x}
                if _check_valid(new_set, triples):
                    best_set = new_set
                    best_size += 1
    
    # Strategy 3: Local search - try removing one element and adding multiple others
    if time.time() < deadline - 5.0:
        # Try to remove one element from the current set and see if we can add two or more
        current = list(best_set)
        for i in range(len(current)):
            if time.time() > deadline - 3.0:
                break
            x_remove = current[i]
            reduced_set = best_set - {x_remove}
            
            # Try to add elements not in reduced_set
            available = [y for y in range(1, N + 1) if y not in reduced_set]
            random.shuffle(available)
            
            improved = False
            for x_add in available:
                if time.time() > deadline - 3.0:
                    break
                new_set = reduced_set | {x_add}
                if _check_valid(new_set, triples):
                    best_set = new_set
                    best_size = len(best_set)
                    improved = True
                    break
            
            if improved:
                break
    
    # Strategy 4: Try a different starting point - maybe not a contiguous suffix
    # Let's try a greedy construction from scratch with randomization
    if time.time() < deadline - 10.0:
        for trial in range(10):
            if time.time() > deadline - 5.0:
                break
            
            S = set()
            elements = list(range(1, N + 1))
            random.shuffle(elements)
            
            for x in elements:
                if time.time() > deadline - 5.0:
                    break
                new_set = S | {x}
                if _check_valid(new_set, triples):
                    S = new_set
            
            if len(S) > best_size:
                best_set = S
                best_size = len(S)
    
    return list(best_set)

# EVOLVE-BLOCK-END
