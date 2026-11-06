"""Integer sum-product extremal sets: geometric progression with local search."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _sumset_size(A):
    """Compute |A+A| with equal summands allowed."""
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] + A[j])
    return len(s)

def _prodset_size(A):
    """Compute |A*A| with equal factors allowed."""
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] * A[j])
    return len(s)

def _score(A):
    return max(_sumset_size(A), _prodset_size(A))

def _valid(A, max_val=10000):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(a < 1 for a in A):
        return False
    if any(a > max_val for a in A):
        return False
    return True

def _gen_geo(a, p, q, n=20, max_val=10000):
    """Generate geometric progression: a, a*p/q, a*p^2/q^2, ... rounded to integers."""
    A = []
    for i in range(n):
        val = a * (p ** i) // (q ** i)
        if val <= 0 or val > max_val:
            return None
        A.append(val)
    if len(set(A)) != n:
        return None
    if any(a < 1 for a in A):
        return None
    return A

def _local_search(A, max_time, deadline):
    """Hill-climbing local search with swap and replace moves."""
    A = sorted(A)
    best_score = _score(A)
    
    while time.time() < deadline:
        improved = False
        n = len(A)
        
        # Try all single-element replacements
        for i in range(n):
            old = A[i]
            # Try nearby values
            for delta in range(1, 50):
                for new_val in [old + delta, old - delta]:
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in A:
                        continue
                    A[i] = new_val
                    sc = _score(A)
                    if sc < best_score:
                        best_score = sc
                        improved = True
                    A[i] = old
                if time.time() >= deadline:
                    return sorted(A), best_score
        
        # Try all swaps
        for i in range(n):
            for j in range(i+1, n):
                A[i], A[j] = A[j], A[i]
                sc = _score(A)
                if sc < best_score:
                    best_score = sc
                    improved = True
                A[i], A[j] = A[j], A[i]
                if time.time() >= deadline:
                    return sorted(A), best_score
        
        if not improved:
            break
    
    return sorted(A), best_score

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    best_A = list(range(1, 21))
    best_score = _score(best_A)
    
    # Phase 1: Grid search over geometric progressions with rational ratios
    # Try various (a, p, q) combinations
    for p in range(2, 20):
        for q in range(1, p):
            if time.time() >= deadline:
                break
            # Find valid starting values
            for a in range(1, 500):
                if time.time() >= deadline:
                    break
                A = _gen_geo(a, p, q, 20, 10000)
                if A is None:
                    continue
                sc = _score(A)
                if sc < best_score:
                    best_score = sc
                    best_A = list(A)
    
    if time.time() >= deadline:
        return best_A
    
    # Phase 2: Local search from best geometric progression
    A, sc = _local_search(best_A, 30, deadline)
    if sc < best_score:
        best_score = sc
        best_A = A
    
    if time.time() >= deadline:
        return best_A
    
    # Phase 3: Try some structured sets and local search from them
    # Arithmetic progression
    for start in range(1, 50):
        for step in range(1, 20):
            if time.time() >= deadline:
                break
            A = [start + i * step for i in range(20)]
            if _valid(A):
                sc = _score(A)
                if sc < best_score:
                    best_score = sc
                    best_A = list(A)
    
    if time.time() >= deadline:
        return best_A
    
    # Final local search if time remains
    if time.time() < deadline - 5:
        A, sc = _local_search(best_A, 20, deadline)
        if sc < best_score:
            best_score = sc
            best_A = A
    
    return best_A
# EVOLVE-BLOCK-END
