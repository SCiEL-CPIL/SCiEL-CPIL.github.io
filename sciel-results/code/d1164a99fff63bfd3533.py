"""Two Sidon sets with disjoint difference sets.
Uses optimized greedy construction with bitmasks for fast collision checking
and a strict time budget to ensure termination."""

import time
import random

def _check_sidon(els):
    """Check all pair sums a+b (a<=b) are unique."""
    n = len(els)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _nonzero_diffs(els):
    """Set of all nonzero differences (positive and negative)."""
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = els[j] - els[i]
                d.add(diff)
    return d

def _valid(A, B):
    """Check A,B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True

def _try_add_to_set(S, x, used_sums, other_diffs):
    """Try to add x to set S. Returns True if valid."""
    s2 = 2 * x
    if s2 in used_sums:
        return False
    for a in S:
        if (a + x) in used_sums:
            return False
    for a in S:
        d = x - a
        if d != 0 and d in other_diffs:
            return False
        d = a - x
        if d != 0 and d in other_diffs:
            return False
    return True

def _commit_add(S, x, used_sums):
    """Actually add x to S and update used_sums."""
    S.append(x)
    used_sums.add(2 * x)
    for a in S:
        if a != x:
            used_sums.add(a + x)

def _build_state(A, B):
    """Build state dictionaries for fast checking."""
    used_sums_A = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            used_sums_A.add(A[i] + A[j])
    
    used_sums_B = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            used_sums_B.add(B[i] + B[j])
            
    diffs_A = _nonzero_diffs(A)
    diffs_B = _nonzero_diffs(B)
    
    return used_sums_A, used_sums_B, diffs_A, diffs_B

def _fast_polish(A, B, time_limit=5.0):
    """Fast greedy polish to maximize size."""
    start = time.time()
    A_set = set(A)
    B_set = set(B)
    used_sums_A, used_sums_B, diffs_A, diffs_B = _build_state(A, B)
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Try to add to A
        if len(A) < 64:
            for x in range(1, 257):
                if x not in A_set:
                    if _try_add_to_set(A, x, used_sums_A, diffs_B):
                        _commit_add(A, x, used_sums_A)
                        A_set.add(x)
                        for a in A:
                            if a != x:
                                diffs_A.add(x - a)
                                diffs_A.add(a - x)
                        improved = True
                        break
        
        # Try to add to B
        if len(B) < 64:
            for x in range(1, 257):
                if x not in B_set:
                    if _try_add_to_set(B, x, used_sums_B, diffs_A):
                        _commit_add(B, x, used_sums_B)
                        B_set.add(x)
                        for b in B:
                            if b != x:
                                diffs_B.add(x - b)
                                diffs_B.add(b - x)
                        improved = True
                        break
                        
    return A, B

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 55  # Strict safety margin for 60s limit
    n = 256
    max_size = 64

    best_A = [1]
    best_B = [2]
    best_score = 2

    rng = random.Random(42)

    def _attempt_build(seed_A=None, seed_B=None):
        """Build A and B from optional seeds using optimized greedy strategy."""
        A = list(seed_A) if seed_A else []
        B = list(seed_B) if seed_B else []

        if not A:
            A.append(rng.randint(1, n))
        if not B:
            B.append(rng.randint(1, n))

        A = list(set(A))
        B = list(set(B))

        if not _valid(A, B):
            return None, None, 0

        used_sums_A, used_sums_B, diffs_A, diffs_B = _build_state(A, B)
        
        # Greedy addition phase
        # We alternate adding to A and B to balance growth
        while time.time() < deadline - 5:
            progress = False
            
            # Try to add to A
            if len(A) < max_size:
                # Randomize candidate order to escape local minima
                candidates = [x for x in range(1, n + 1) if x not in A]
                rng.shuffle(candidates)
                for x in candidates:
                    if _try_add_to_set(A, x, used_sums_A, diffs_B):
                        _commit_add(A, x, used_sums_A)
                        for a in A:
                            if a != x:
                                diffs_A.add(x - a)
                                diffs_A.add(a - x)
                        progress = True
                        break
                        
            # Try to add to B
            if not progress and len(B) < max_size:
                candidates = [x for x in range(1, n + 1) if x not in B]
                rng.shuffle(candidates)
                for x in candidates:
                    if _try_add_to_set(B, x, used_sums_B, diffs_A):
                        _commit_add(B, x, used_sums_B)
                        for b in B:
                            if b != x:
                                diffs_B.add(x - b)
                                diffs_B.add(b - x)
                        progress = True
                        break
                        
            if not progress:
                break

        if _valid(A, B):
            return A, B, len(A) + len(B)
        return None, None, 0

    # Main search loop
    for it in range(10000):
        if time.time() > deadline - 10:
            break

        # Strategy: Random starts with slight bias towards separation
        if it % 2 == 0:
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, 256)]
        else:
            A0 = [rng.randint(1, 256)]
            B0 = [rng.randint(1, 256)]
            
        A, B, score = _attempt_build(A0, B0)
        
        if A is not None and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)
            
            # Polish the best found so far
            if time.time() < deadline - 15:
                best_A, best_B = _fast_polish(best_A, best_B, time_limit=3.0)
                best_score = len(best_A) + len(best_B)

    # Final polish with remaining time
    remaining = deadline - time.time()
    if remaining > 2:
        best_A, best_B = _fast_polish(best_A, best_B, time_limit=max(1.0, remaining - 2))
        best_score = len(best_A) + len(best_B)

    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2

    return [best_A, best_B]
