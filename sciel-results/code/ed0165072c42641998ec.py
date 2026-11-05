import time
import random
from itertools import combinations

DEADLINE = 85  # seconds; evaluator kills at 90s, keep margin

def _sumset_size(A):
    """Compute |A+A| where A is a sorted list of distinct positive ints."""
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai + A[j])
    return len(s)

def _prodset_size(A):
    """Compute |A*A| where A is a sorted list of distinct positive ints."""
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai * A[j])
    return len(s)

def _score(A):
    return max(_sumset_size(A), _prodset_size(A))

def _local_search(A, deadline, rng):
    """Hill-climbing with random restarts on element swaps/moves."""
    A = sorted(A)
    best_score = _score(A)
    best_A = list(A)
    n = len(A)
    max_val = 10000
    
    # Precompute some structure
    while time.time() < deadline:
        improved = False
        # Try swapping each pair of elements with random candidates
        for i in range(n):
            for j in range(i+1, n):
                # Try replacing A[i] with a random value
                for _ in range(3):
                    new_val = rng.randint(1, max_val)
                    if new_val in A:
                        continue
                    new_A = list(A)
                    new_A[i] = new_val
                    new_A = sorted(new_A)
                    # Quick check: if new_val is too large, product set will blow up
                    if new_val > max_val:
                        continue
                    sc = _score(new_A)
                    if sc < best_score:
                        best_score = sc
                        best_A = new_A
                        A = new_A
                        improved = True
                        break
        if not improved:
            # Try a random perturbation: replace 1-2 elements
            k = rng.randint(1, 2)
            new_A = list(A)
            for _ in range(k):
                idx = rng.randint(0, n-1)
                new_A[idx] = rng.randint(1, max_val)
            # Check distinctness
            if len(set(new_A)) != n:
                continue
            new_A = sorted(new_A)
            sc = _score(new_A)
            if sc < best_score:
                best_score = sc
                best_A = new_A
                A = new_A
            elif time.time() > deadline:
                break
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_A = None
    best_score = float('inf')
    
    # Strategy 1: Start with small consecutive integers
    A0 = list(range(1, 21))
    s0 = _score(A0)
    if s0 < best_score:
        best_score = s0
        best_A = list(A0)
    
    # Strategy 2: Geometric progression with ratio 2
    # A = {a, 2a, 4a, ..., a*2^19}
    # But a*2^19 must be <= 10000, so a <= 1 (since 2^19 = 524288)
    # So only a=1: {1, 2, 4, 8, ..., 2^19} but 2^19=524288 > 10000
    # So geometric with ratio 2 won't work for 20 elements <= 10000
    
    # Strategy 3: Use small numbers with some structure
    # Try sets like {1,2,3,...,k} union some larger numbers
    # Or try to minimize both sum and product sets
    
    # Greedy constructive: build element by element
    # Start with 1, then add elements that minimize incremental max(|A+A|,|A*A|)
    A = [1]
    candidates = list(range(2, 10001))
    rng.shuffle(candidates)
    
    # Actually, let's try a different approach: 
    # Use sets that are "multiplicatively structured" to keep product set small
    # and "additively structured" to keep sum set small
    
    # Try: A = {a_i} where a_i are chosen to have many additive and multiplicative relations
    
    # Let's try a hybrid: start with small consecutive, then optimize
    for base_size in range(10, 21):
        A_base = list(range(1, base_size + 1))
        # Extend to 20 elements by adding larger numbers
        remaining = 20 - base_size
        if remaining <= 0:
            continue
        
        # Try different ways to extend
        # Option A: add consecutive numbers
        A_ext = list(range(1, 21))
        sc = _score(A_ext)
        if sc < best_score:
            best_score = sc
            best_A = list(A_ext)
        
        # Option B: add numbers that are multiples of existing ones
        # This can keep product set smaller
        A_try = list(range(1, base_size + 1))
        for _ in range(remaining):
            # Pick a number that's a multiple of some element
            best_cand = None
            best_cand_score = float('inf')
            for cand in range(base_size + 1, min(base_size * 3 + 20, 10001)):
                if cand in A_try:
                    continue
                new_A = sorted(A_try + [cand])
                # Quick score estimate
                sc_est = _score(new_A)
                if sc_est < best_cand_score:
                    best_cand_score = sc_est
                    best_cand = cand
                if time.time() > deadline:
                    break
            if best_cand is not None:
                A_try.append(best_cand)
        
        if len(A_try) == 20 and len(set(A_try)) == 20:
            A_try = sorted(A_try)
            sc = _score(A_try)
            if sc < best_score:
                best_score = sc
                best_A = list(A_try)
        
        if time.time() > deadline - 5:
            break
    
    # Local search on the best found
    if best_A is not None:
        best_A, best_score = _local_search(best_A, deadline - 3, rng)
    
    # Final local search with more time if available
    if time.time() < deadline - 1:
        best_A, best_score = _local_search(best_A, deadline - 1, rng)
    
    return best_A
