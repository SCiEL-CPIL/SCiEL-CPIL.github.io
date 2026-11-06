"""
Integer sum-product extremal sets: minimize max(|A+A|, |A*A|) for |A|=20, 1<=a<=10000.
Strategy: Use geometric progressions (powers of small integers) to create multiplicative
collisions via unique factorization, then refine with local search (swap/replace) targeting
product collisions and sum compactness.
"""
import time
import itertools
import random
import numpy as np

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
    """Compute max(|A+A|, |A*A|)."""
    return max(_sumset_size(A), _prodset_size(A))

def _is_valid(A, max_val=10000):
    """Check A has 20 distinct positive integers <= max_val."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > max_val for x in A):
        return False
    return True

def _make_geometric_set(base, start, count, max_val=10000):
    """Generate a geometric progression: start, start*base, start*base^2, ..."""
    vals = []
    v = start
    for _ in range(count):
        if v > max_val:
            break
        vals.append(v)
        v *= base
    return vals

def _random_geometric_candidate(base1, base2, n1, n2, max_val=10000):
    """
    Generate a candidate set as union of two geometric progressions.
    base1^e1 * s1 and base2^e2 * s2, picking n1 from first, n2 from second.
    """
    set1 = set()
    set2 = set()
    
    # Generate geometric progression for base1
    v = 1
    gp1 = []
    while v <= max_val:
        gp1.append(v)
        v *= base1
    
    # Generate geometric progression for base2
    v = 1
    gp2 = []
    while v <= max_val:
        gp2.append(v)
        v *= base2
    
    # Pick n1 elements from gp1 (consecutive powers)
    if len(gp1) >= n1:
        start_idx = random.randint(0, max(0, len(gp1) - n1))
        set1 = set(gp1[start_idx:start_idx + n1])
    
    # Pick n2 elements from gp2, avoiding overlap with set1
    remaining = [x for x in gp2 if x not in set1]
    if len(remaining) >= n2:
        start_idx = random.randint(0, max(0, len(remaining) - n2))
        set2 = set(remaining[start_idx:start_idx + n2])
    
    A = list(set1 | set2)
    if len(A) < 20:
        # Fill in with random small integers
        used = set(A)
        for i in range(1, max_val + 1):
            if len(A) >= 20:
                break
            if i not in used:
                A.append(i)
                used.add(i)
    
    A = A[:20]
    return A

def _local_search(A, max_val=10000, max_iter=5000, deadline=None):
    """
    Local search: try swapping one element with another to improve score.
    Neighbor: replace one element with a nearby integer or a multiple/divisor.
    """
    A = sorted(A)
    current_score = _score(A)
    best_score = current_score
    best_A = A[:]
    no_improve = 0
    iter_count = 0
    
    while no_improve < 200 and iter_count < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        iter_count += 1
        
        improved = False
        # Try replacing each element with a candidate
        for i in range(20):
            old_val = A[i]
            # Candidate replacements: nearby values, multiples, divisors
            candidates = set()
            
            # Nearby values
            for delta in range(-5, 6):
                nv = old_val + delta
                if 1 <= nv <= max_val and nv not in set(A) and nv != old_val:
                    candidates.add(nv)
            
            # Multiples and divisors
            if old_val > 1:
                for d in range(2, min(10, int(np.sqrt(old_val)) + 1)):
                    if old_val % d == 0:
                        candidates.add(d)
                        candidates.add(old_val // d)
                for m in range(2, 10):
                    nv = old_val * m
                    if nv <= max_val:
                        candidates.add(nv)
            
            # Small primes and powers
            for p in [2, 3, 5, 7, 11, 13]:
                candidates.add(p)
                for k in range(1, 6):
                    candidates.add(p ** k)
            
            for nv in candidates:
                if nv in set(A) or nv == old_val or nv < 1 or nv > max_val:
                    continue
                new_A = A[:]
                new_A[i] = nv
                new_A.sort()
                s = _score(new_A)
                if s < current_score:
                    current_score = s
                    A = new_A
                    improved = True
                    if s < best_score:
                        best_score = s
                        best_A = A[:]
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
    
    return best_A, best_score

def _multi_start_geometric_search(deadline):
    """
    Multi-start local search initialized with geometric progression candidates.
    """
    best_A = None
    best_score = float('inf')
    
    # Try various geometric bases
    bases = [(2, 3), (2, 5), (3, 5), (2, 7), (3, 7), (4, 5), (2, 11), (3, 11)]
    
    for b1, b2 in bases:
        # Multiple random starts for each base pair
        for trial in range(5):
            if time.time() > deadline:
                break
            n1 = random.randint(10, 15)
            n2 = 20 - n1
            A = _random_geometric_candidate(b1, b2, n1, n2)
            if not _is_valid(A):
                continue
            A_sorted = sorted(A)
            A_opt, sc = _local_search(A_sorted, deadline=deadline)
            if sc < best_score:
                best_score = sc
                best_A = A_opt
    
    return best_A, best_score

def _try_pure_geometric(deadline):
    """
    Try pure geometric progressions with various bases and starting points.
    """
    best_A = None
    best_score = float('inf')
    
    for base in [2, 3, 4, 5]:
        # Try different starting exponents
        for start_pow in range(0, 5):
            if time.time() > deadline:
                break
            A = []
            v = base ** start_pow
            while len(A) < 20 and v <= 10000:
                A.append(v)
                v *= base
            if len(A) == 20:
                sc = _score(A)
                if sc < best_score:
                    best_score = sc
                    best_A = A[:]
    
    return best_A, best_score

def construct():
    """
    Construct 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|).
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Phase 1: Try pure geometric progressions (fast)
    best_A, best_score = _try_pure_geometric(deadline)
    
    if best_A is None:
        # Fallback: consecutive integers
        best_A = list(range(1, 21))
        best_score = _score(best_A)
    
    # Phase 2: Multi-start geometric + local search
    if time.time() < deadline:
        A2, s2 = _multi_start_geometric_search(deadline)
        if A2 is not None and s2 < best_score:
            best_A = A2
            best_score = s2
    
    # Phase 3: Additional local search refinement on best candidate
    if time.time() < deadline - 5:
        A3, s3 = _local_search(best_A, deadline=deadline - 5)
        if s3 < best_score:
            best_A = A3
            best_score = s3
    
    # Final validation
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return best_A
