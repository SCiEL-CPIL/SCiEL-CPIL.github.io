"""Two Sidon sets with disjoint difference sets - greedy with local search."""
import time
from itertools import combinations_with_replacement, combinations

def _pair_sums_unique(els):
    """Check if all pair sums (with replacement) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _nonzero_diffs(els):
    """Return set of nonzero differences (b-a for a<b in els)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(els[j] - els[i])
    return diffs

def _is_valid_pair(A, B):
    """Check if A and B are valid Sidon sets with disjoint nonzero differences."""
    if not _pair_sums_unique(A):
        return False
    if not _pair_sums_unique(B):
        return False
    if A and B:
        if _nonzero_diffs(A) & _nonzero_diffs(B):
            return False
    return True

def _greedy_construct(n=256, max_size=64, order=None):
    """Greedy construction: try to add each element to A or B."""
    if order is None:
        order = list(range(1, n+1))
    
    A = []
    B = []
    
    # Precompute for efficiency
    for x in order:
        # Try adding to A
        if len(A) < max_size:
            A_test = A + [x]
            if _pair_sums_unique(A_test):
                dA = _nonzero_diffs(A_test)
                if not B or not (dA & _nonzero_diffs(B)):
                    A = A_test
                    continue
        # Try adding to B
        if len(B) < max_size:
            B_test = B + [x]
            if _pair_sums_unique(B_test):
                dB = _nonzero_diffs(B_test)
                if not A or not (dB & _nonzero_diffs(A)):
                    B = B_test
                    continue
        # Try adding to A even if B exists (recheck)
        if len(A) < max_size:
            A_test = A + [x]
            if _pair_sums_unique(A_test):
                dA = _nonzero_diffs(A_test)
                dB = _nonzero_diffs(B) if B else set()
                if not (dA & dB):
                    A = A_test
                    continue
    
    return A, B

def _local_search(A, B, n=256, max_size=64, max_iters=200):
    """Try local swaps to improve combined size."""
    A = list(A)
    B = list(B)
    current_size = len(A) + len(B)
    
    for _ in range(max_iters):
        improved = False
        
        # Try moving an element from one set to the other
        for i, a in enumerate(A):
            if len(B) < max_size:
                A_new = A[:i] + A[i+1:]
                B_new = B + [a]
                if _is_valid_pair(A_new, B_new):
                    # Check if this helps (same size but maybe enables more additions)
                    A, B = A_new, B_new
                    improved = True
                    break
        if improved:
            continue
            
        for i, b in enumerate(B):
            if len(A) < max_size:
                B_new = B[:i] + B[i+1:]
                A_new = A + [b]
                if _is_valid_pair(A_new, B_new):
                    A, B = A_new, B_new
                    improved = True
                    break
        if improved:
            continue
        
        # Try swapping elements between A and B
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                A_new = A[:i] + A[i+1:] + [b]
                B_new = B[:j] + B[j+1:] + [a]
                if _is_valid_pair(A_new, B_new):
                    A, B = A_new, B_new
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try adding a new element from [1,n] not in A or B
        used = set(A) | set(B)
        for x in range(1, n+1):
            if x in used:
                continue
            if len(A) < max_size:
                A_new = A + [x]
                if _is_valid_pair(A_new, B):
                    A = A_new
                    improved = True
                    break
            if len(B) < max_size:
                B_new = B + [x]
                if _is_valid_pair(A, B_new):
                    B = B_new
                    improved = True
                    break
        if improved:
            continue
            
        break  # No improvement possible
    
    return A, B

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s kill
    
    n = 256
    max_size = 64
    
    best_A, best_B = [], []
    best_size = 0
    
    # Try multiple orderings
    import random
    random.seed(42)
    
    # First: natural order
    A, B = _greedy_construct(n, max_size, order=list(range(1, n+1)))
    A, B = _local_search(A, B, n, max_size, max_iters=100)
    if len(A) + len(B) > best_size:
        best_A, best_B = A, B
        best_size = len(A) + len(B)
    
    if time.time() - start_time > deadline:
        return [best_A, best_B]
    
    # Try reverse order
    A, B = _greedy_construct(n, max_size, order=list(range(n, 0, -1)))
    A, B = _local_search(A, B, n, max_size, max_iters=100)
    if len(A) + len(B) > best_size:
        best_A, best_B = A, B
        best_size = len(A) + len(B)
    
    if time.time() - start_time > deadline:
        return [best_A, best_B]
    
    # Try random orderings
    for trial in range(20):
        if time.time() - start_time > deadline:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        A, B = _greedy_construct(n, max_size, order=order)
        A, B = _local_search(A, B, n, max_size, max_iters=50)
        if len(A) + len(B) > best_size:
            best_A, best_B = A, B
            best_size = len(A) + len(B)
    
    if time.time() - start_time > deadline:
        return [best_A, best_B]
    
    # Final local search on best
    best_A, best_B = _local_search(best_A, best_B, n, max_size, max_iters=200)
    
    return [best_A, best_B]
