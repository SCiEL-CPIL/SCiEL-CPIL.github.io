"""Improved program: precompute pairwise-coprime sum sets, then backtrack over difference structure."""
import time
import math
import random
from itertools import combinations, product

# Deadline guard
_DEADLINE = 90.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = list(set(vals))
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if _gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _sumset(A, B):
    """Compute the set of all sums a+b for a in A, b in B."""
    return {a + b for a in A for b in B}

def _score(A, B):
    """Return (is_valid, cardinality) for the sumset A+B."""
    s = _sumset(A, B)
    if not _pairwise_coprime(list(s)):
        return False, 0
    return True, len(s)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Baseline
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # baseline has 5 distinct sums
    
    # Verify baseline
    valid, card = _score(best_A, best_B)
    if valid:
        best_score = card
    
    deadline = _DEADLINE - 2.0  # 2s safety margin
    
    # Strategy 1: Systematic search with pruning
    # We search over A first (sorted, distinct, in [1,128]), then B.
    # To make this tractable, we use structured families.
    
    # Family 1: Arithmetic progressions A = [a, a+d, a+2d], B = [b, b+e, b+2e]
    # Sums form a 3x3 grid: {a+b, a+b+e, a+b+2e, a+d+b, a+d+b+e, a+d+b+2e, a+2d+b, a+2d+b+e, a+2d+b+2e}
    # The distinct sums depend on whether d and e have common structure.
    
    # Strategy 2: Random search with local improvement
    random.seed(42)
    
    def random_AB():
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        return A, B
    
    def local_search(A, B, max_iters=200):
        """Try to improve by swapping one element at a time."""
        A = list(A)
        B = list(B)
        cur_valid, cur_card = _score(A, B)
        if not cur_valid:
            return A, B, 0
        
        for _ in range(max_iters):
            if _elapsed() > deadline:
                break
            
            improved = False
            # Try modifying A
            for i in range(3):
                old = A[i]
                for new_val in random.sample(range(1, 129), 20):
                    if new_val in A:
                        continue
                    A[i] = new_val
                    A.sort()
                    v, c = _score(A, B)
                    if v and c > cur_card:
                        cur_card = c
                        improved = True
                        break
                    A[i] = old
                if improved:
                    break
            
            # Try modifying B
            if not improved:
                for j in range(3):
                    old = B[j]
                    for new_val in random.sample(range(1, 129), 20):
                        if new_val in B:
                            continue
                        B[j] = new_val
                        B.sort()
                        v, c = _score(A, B)
                        if v and c > cur_card:
                            cur_card = c
                            improved = True
                            break
                        B[j] = old
                    if improved:
                        break
            
            if not improved:
                break
        
        return A, B, cur_card
    
    # Phase 1: Random search with local improvement
    n_random = 500
    for i in range(n_random):
        if _elapsed() > deadline:
            break
        
        A, B = random_AB()
        v, c = _score(A, B)
        if v:
            A, B, c = local_search(A, B, max_iters=100)
            if c > best_score:
                best_score = c
                best_A, best_B = A, B
                if best_score == 9:
                    break
        else:
            # Try to repair: adjust one element
            for _ in range(30):
                if _elapsed() > deadline:
                    break
                # Randomly perturb
                if random.random() < 0.5:
                    idx = random.randint(0, 2)
                    A[idx] = random.randint(1, 128)
                    if len(set(A)) != 3:
                        continue
                    A.sort()
                else:
                    idx = random.randint(0, 2)
                    B[idx] = random.randint(1, 128)
                    if len(set(B)) != 3:
                        continue
                    B.sort()
                v, c = _score(A, B)
                if v:
                    A, B, c = local_search(A, B, max_iters=100)
                    if c > best_score:
                        best_score = c
                        best_A, best_B = A, B
                        if best_score == 9:
                            break
                if best_score == 9:
                    break
    
    # Phase 2: Structured search over arithmetic progression families
    if best_score < 9 and _elapsed() < deadline:
        # A = [a, a+d, a+2d], B = [b, b+e, b+2e]
        # Sums: base + {0, e, 2e, d, d+e, d+2e, 2d, 2d+e, 2d+2e}
        # For all 9 to be distinct: d, e, 2d, 2e, d+e, d+2e, 2d+e, 2d+2e must all be distinct
        # This requires d and e to be incommensurate in a certain sense
        # Actually for distinctness we need: d != e, d != 2e, 2d != e, etc.
        # Simpler: just check
        
        a_range = range(1, 50)
        d_range = range(1, 40)
        b_range = range(1, 50)
        e_range = range(1, 40)
        
        for a in a_range:
            if _elapsed() > deadline:
                break
            for d in d_range:
                if _elapsed() > deadline:
                    break
                A = [a, a + d, a + 2 * d]
                if A[2] > 128 or len(set(A)) != 3:
                    continue
                for b in b_range:
                    if _elapsed() > deadline:
                        break
                    for e in e_range:
                        if _elapsed() > deadline:
                            break
                        B = [b, b + e, b + 2 * e]
                        if B[2] > 128 or len(set(B)) != 3:
                            continue
                        v, c = _score(A, B)
                        if v and c > best_score:
                            best_score = c
                            best_A, best_B = A, B
                            if best_score == 9:
                                break
                    if best_score == 9:
                        break
                if best_score == 9:
                    break
            if best_score == 9:
                break
    
    # Phase 3: More aggressive random search if still not 9
    if best_score < 9 and _elapsed() < deadline:
        for i in range(2000):
            if _elapsed() > deadline:
                break
            A, B = random_AB()
            v, c = _score(A, B)
            if v:
                A, B, c = local_search(A, B, max_iters=50)
                if c > best_score:
                    best_score = c
                    best_A, best_B = A, B
                    if best_score == 9:
                        break
    
    return [best_A, best_B]
