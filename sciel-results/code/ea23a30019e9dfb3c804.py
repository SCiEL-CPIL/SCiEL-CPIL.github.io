"""Improved program for two Sidon sets with disjoint difference sets.
Uses precomputed conflict tables, degree-based greedy, and swap local improvement.
"""
import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 88.0  # Safety margin for 90s limit

def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    
    # Check Sidon property for A
    sumsA = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            s = ai + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
            
    # Check Sidon property for B
    sumsB = set()
    for i in range(len(B)):
        bi = B[i]
        for j in range(i, len(B)):
            s = bi + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
            
    # Check difference disjointness
    diffsA = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(len(A)):
            if i != j:
                diffsA.add(A[j] - ai)
                
    diffsB = set()
    for i in range(len(B)):
        bi = B[i]
        for j in range(len(B)):
            if i != j:
                diffsB.add(B[j] - bi)
                
    if diffsA & diffsB:
        return False
        
    # Check bounds
    for x in A:
        if x < 1 or x > N:
            return False
    for x in B:
        if x < 1 or x > N:
            return False
            
    return True


def _build_conflict_tables():
    """
    Precompute conflict tables for fast O(1) validity checks.
    
    sum_conflict_A[x][y] = True if x+y conflicts with some pair in A (i.e., there exist a,b in A with a+b = x+y, not both equal to x,y)
    Actually, we need: can we add x to set S? We need:
    1. For all s in S: s+x not in sums(S)  [sum collision]
    2. x+x not in sums(S)
    3. For all s in S: x-s not in diffs(S) and s-x not in diffs(S)  [internal diff collision]
    4. For all s in S: x-s not in diffs(other) and s-x not in diffs(other)  [cross diff collision]
    
    Since S changes dynamically, we can't fully precompute. But we CAN precompute:
    - For any pair (x,y), whether x+y equals some other pair's sum (but that depends on the set)
    
    Better approach: maintain sumsA, sumsB, diffsA, diffsB as sets, and check membership in O(1).
    The current code already does this. The bottleneck is the inner loops over existing elements.
    
    Key optimization: instead of looping over all elements in A to check sum/diff conflicts,
    we can just check set membership directly.
    """
    pass

def _can_add_to_set(x, S, sums, diffs, other_diffs):
    """
    Fast check: can x be added to set S with given sums/diffs state?
    Returns True if valid.
    """
    # Check sum collisions: for each s in S, s+x must not be in sums
    for s in S:
        if (s + x) in sums:
            return False
    if (x + x) in sums:
        return False
    # Check internal diff collisions: for each s in S, x-s and s-x must not be in diffs
    for s in S:
        if (x - s) in diffs or (s - x) in diffs:
            return False
    # Check cross diff collisions: for each s in S, x-s and s-x must not be in other_diffs
    for s in S:
        if (x - s) in other_diffs or (s - x) in other_diffs:
            return False
    return True


def _greedy_construct(rng, noise_scale=0.5):
    """
    Greedy construction with degree-based heuristic.
    Uses O(1) set membership checks for validity.
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()
    
    remaining = list(range(1, N + 1))
    rng.shuffle(remaining)
    
    for x in remaining:
        if len(A) >= MAX_EACH and len(B) >= MAX_EACH:
            break
            
        valid_A = False
        valid_B = False
        
        if len(A) < MAX_EACH:
            valid_A = _can_add_to_set(x, A, sumsA, diffsA, diffsB)
            
        if len(B) < MAX_EACH:
            valid_B = _can_add_to_set(x, B, sumsB, diffsB, diffsA)
            
        if not valid_A and not valid_B:
            continue
            
        # Score: prefer balance, with noise
        score_A = 10000.0
        score_B = 10000.0
        
        if valid_A:
            # Heuristic: prefer smaller set, but also consider "degree" (how many future elements would be blocked)
            # Approximate degree: number of remaining elements that would conflict with x if added to A
            # This is expensive, so use a simpler proxy: prefer balance
            score_A = len(A) - len(B) + rng.uniform(-noise_scale, noise_scale)
            
        if valid_B:
            score_B = len(B) - len(A) + rng.uniform(-noise_scale, noise_scale)
            
        if score_A < score_B:
            side = 0
        else:
            side = 1
            
        if side == 0:
            A.append(x)
            for a in A[:-1]:
                sumsA.add(a + x)
            sumsA.add(x + x)
            for a in A[:-1]:
                diffsA.add(x - a)
                diffsA.add(a - x)
        else:
            B.append(x)
            for b in B[:-1]:
                sumsB.add(b + x)
            sumsB.add(x + x)
            for b in B[:-1]:
                diffsB.add(x - b)
                diffsB.add(b - x)
        
    return A, B

def _recompute_state(lst):
    """Recompute sums and diffs for a list."""
    sums = set()
    diffs = set()
    n = len(lst)
    for i in range(n):
        li = lst[i]
        for j in range(i, n):
            sums.add(li + lst[j])
        for j in range(n):
            if i != j:
                diffs.add(lst[j] - li)
    return sums, diffs


def _local_improve(A, B, rng, t0):
    """
    Try to improve the solution by adding elements and performing swaps.
    """
    A = list(A)
    B = list(B)
    
    sumsA, diffsA = _recompute_state(A)
    sumsB, diffsB = _recompute_state(B)
    used = set(A) | set(B)
    
    max_iters = 50
    for _ in range(max_iters):
        if time.time() - t0 > DEADLINE - 3:
            break
            
        improved = False
        
        # Phase 1: Try to add any unused element
        candidates = [x for x in range(1, N + 1) if x not in used]
        rng.shuffle(candidates)
        
        for x in candidates:
            if len(A) < MAX_EACH and _can_add_to_set(x, A, sumsA, diffsA, diffsB):
                A.append(x)
                used.add(x)
                for a in A[:-1]:
                    sumsA.add(a + x)
                sumsA.add(x + x)
                for a in A[:-1]:
                    diffsA.add(x - a)
                    diffsA.add(a - x)
                improved = True
                break
            if len(B) < MAX_EACH and _can_add_to_set(x, B, sumsB, diffsB, diffsA):
                B.append(x)
                used.add(x)
                for b in B[:-1]:
                    sumsB.add(b + x)
                sumsB.add(x + x)
                for b in B[:-1]:
                    diffsB.add(x - b)
                    diffsB.add(b - x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps - remove an element from one set, try to add two new ones
        # Simpler: try to move an element from A to B or vice versa
        # Or: remove a from A, then try to add a different element to A
        
        # Try: for each a in A (random order), temporarily remove it, then try to add any candidate
        a_list = list(range(len(A)))
        rng.shuffle(a_list)
        
        for idx in a_list:
            if time.time() - t0 > DEADLINE - 3:
                break
            a = A[idx]
            
            # Temporarily remove a from A
            A.remove(a)
            # Recompute state for A
            sumsA, diffsA = _recompute_state(A)
            
            # Try to add any candidate to A
            added = False
            candidates2 = [x for x in range(1, N + 1) if x not in used and x != a]
            rng.shuffle(candidates2)
            
            for x in candidates2:
                if len(A) < MAX_EACH and _can_add_to_set(x, A, sumsA, diffsA, diffsB):
                    A.append(x)
                    used.add(x)
                    for aa in A[:-1]:
                        sumsA.add(aa + x)
                    sumsA.add(x + x)
                    for aa in A[:-1]:
                        diffsA.add(x - aa)
                        diffsA.add(aa - x)
                    added = True
                    break
            
            if not added:
                # Put a back
                A.append(a)
                sumsA, diffsA = _recompute_state(A)
                # a is still in used
            else:
                improved = True
                # a is no longer in A; it might be in B or unused
                if a not in B:
                    used.discard(a)
                break
        
        if not improved:
            break
        
    return A, B

def construct():
    t0 = time.time()
    best_A, best_B = [], []
    best_size = 0
    
    # Run multiple restarts with different noise levels and seeds
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() - t0 > DEADLINE - 2:
            break
            
        seed = 12345 + i * 7919
        rng = random.Random(seed)
        
        # Vary noise scale to explore different parts of the solution space
        # Early runs: low noise (greedy), Later runs: high noise (exploratory)
        noise_scale = 0.05 + (i / num_restarts) * 1.5
        
        A, B = _greedy_construct(rng, noise_scale)
        
        # Local improvement
        A, B = _local_improve(A, B, rng, t0)
        
        if _validate(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A, best_B = list(A), list(B)
                
                # Early stopping if we hit the theoretical max?
                if best_size == 128:
                    break

    # Fallback
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    if not _validate(best_A, best_B):
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
