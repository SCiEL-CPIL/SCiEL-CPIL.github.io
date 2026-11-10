import time
import random
from itertools import combinations_with_replacement

def _compute_scores(A):
    """Compute |A+A| and |A*A| for set A (with repetition allowed)."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def _objective(A):
    s, p = _compute_scores(A)
    return max(s, p)

def _local_search(A, budget_seconds):
    """Local search with swaps to minimize max(|A+A|, |A*A|)."""
    start = time.time()
    A = list(A)
    best = _objective(A)
    n = len(A)
    max_val = 10000
    
    # Try swapping each element with others
    while time.time() - start < budget_seconds:
        improved = False
        for i in range(n):
            if time.time() - start > budget_seconds:
                break
            for j in range(n):
                if i == j:
                    continue
                old = A[i]
                A[i] = A[j]
                # Check distinctness
                if len(set(A)) < n:
                    A[i] = old
                    continue
                obj = _objective(A)
                if obj < best:
                    best = obj
                    improved = True
                else:
                    A[i] = old
            # Also try replacing A[i] with a random value
            for _ in range(5):
                if time.time() - start > budget_seconds:
                    break
                old = A[i]
                A[i] = random.randint(1, max_val)
                if len(set(A)) < n:
                    A[i] = old
                    continue
                obj = _objective(A)
                if obj < best:
                    best = obj
                    improved = True
                else:
                    A[i] = old
        if not improved:
            # Try random restarts of some elements
            for _ in range(10):
                if time.time() - start > budget_seconds:
                    break
                i = random.randint(0, n-1)
                old = A[i]
                A[i] = random.randint(1, max_val)
                if len(set(A)) < n:
                    A[i] = old
                    continue
                obj = _objective(A)
                if obj < best:
                    best = obj
                    improved = True
                else:
                    A[i] = old
            if not improved:
                break
    
    return A, best

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Initial candidate: geometric progression with base 2, but capped
    # {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096} - only 13 elements
    # Let's try a mixed approach
    
    # Strategy 1: Powers of 2 (13 elements) + some others
    # Strategy 2: Arithmetic progression
    # Strategy 3: Geometric with small base
    
    # Let's try several initializations
    candidates = []
    
    # AP: 1,2,...,20
    candidates.append(list(range(1, 21)))
    
    # GP: powers of 2 up to 8192, then fill in
    gp2 = [2**i for i in range(13)]  # 1 to 4096
    # Add more: 5, 7, 11, 13, 17, 19, 23, 29, 31 (primes not powers of 2)
    extras = [x for x in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] 
              if x not in gp2 and x <= 10000]
    candidates.append(gp2 + extras[:20-len(gp2)])
    
    # Mixed: small AP + geometric tail
    mixed = list(range(1, 11)) + [2**i for i in range(10, 14)]  # 1-10, 1024, 2048, 4096, 8192
    mixed = [x for x in mixed if x <= 10000]
    if len(mixed) < 20:
        mixed += [x for x in range(11, 21) if x not in mixed]
    candidates.append(mixed[:20])
    
    # Another: {1,2,3,...,10} union {100,200,...,290}
    mixed2 = list(range(1, 11)) + list(range(100, 300, 20))
    candidates.append(mixed2[:20])
    
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    for cand in candidates:
        if len(cand) != 20 or len(set(cand)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in cand):
            continue
        s = _objective(cand)
        if s < best_score:
            best_score = s
            best_A = list(cand)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = _objective(best_A)
    
    # Local search with time budget
    remaining = deadline - time.time()
    if remaining > 2:
        best_A, best_score = _local_search(best_A, remaining - 2)
    
    # Final validation
    assert len(best_A) == 20
    assert len(set(best_A)) == 20
    assert all(1 <= x <= 10000 for x in best_A)
    
    return best_A
