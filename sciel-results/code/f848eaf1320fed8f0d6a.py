"""
Combined approach: Two-AP structural search + exhaustive 1-swap local search.

Strategy:
  - Generate candidate sets as unions of two disjoint arithmetic progressions.
  - For each candidate, run an exhaustive 1-swap local search using a precomputed
    pool of small and smooth numbers, accepting only strict improvements.
  - This exploits the additive structure of AP unions (bounded sumset growth)
    while the swap search refines elements to induce multiplicative collisions.
"""

import time
import random
import math

def _score(A):
    """Compute max(|A+A|, |A*A|) for a sorted list of 20 distinct positive ints."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _make_pool():
    """Build a candidate pool of replacement values: small ints + smooth numbers."""
    pool = set(range(1, 151))
    # 2,3-smooth numbers up to 10000
    for a in range(0, 14):
        va = 1
        for _ in range(a):
            va *= 2
        for b in range(0, 14):
            vb = va
            for _ in range(b):
                vb *= 3
            if vb <= 10000:
                pool.add(vb)
    # 2,3,5-smooth numbers
    for a in range(0, 14):
        va = 1
        for _ in range(a):
            va *= 2
        for b in range(0, 10):
            vb = va
            for _ in range(b):
                vb *= 3
            for c in range(0, 8):
                vc = vb
                for _ in range(c):
                    vc *= 5
                if vc <= 10000:
                    pool.add(vc)
    pool.discard(0)
    return sorted(pool)

def _local_search_1swap(A, pool, deadline):
    """
    Exhaustive 1-swap local search: repeatedly try replacing each element
    with every pool value (if distinct and in range), accepting strict improvements.
    Loops until no improvement is found or deadline is hit.
    """
    A = sorted(A)
    current_score = _score(A)
    best = A[:]
    best_score = current_score
    n = len(A)
    
    while time.time() < deadline:
        improved = False
        A_set = set(A)
        for i in range(n):
            if time.time() > deadline:
                return best, best_score
            old = A[i]
            for new_val in pool:
                if new_val == old or new_val in A_set:
                    continue
                # Build candidate
                cand = A[:]
                cand[i] = new_val
                cand.sort()
                s = _score(cand)
                if s < current_score:
                    current_score = s
                    A = cand
                    A_set = set(A)
                    improved = True
                    if s < best_score:
                        best_score = s
                        best = A[:]
                    break  # restart i-loop for this element
        if not improved:
            break
    return best, best_score

def _two_ap_candidate(a0, d1, m, b0, d2, k, max_val=10000):
    """
    Build a set from two APs: {a0 + i*d1 for i in range(m)} ∪ {b0 + j*d2 for j in range(k)}.
    Returns sorted list if valid (20 distinct positive ints ≤ max_val), else None.
    """
    if m + k != 20:
        return None
    ap1 = [a0 + i * d1 for i in range(m)]
    ap2 = [b0 + j * d2 for j in range(k)]
    A = ap1 + ap2
    if any(x < 1 or x > max_val for x in A):
        return None
    if len(set(A)) != 20:
        return None
    return sorted(A)

def _search_two_ap(deadline, pool):
    """
    Search over two-AP unions. Try various splits, starts, and steps.
    Run 1-swap local search on promising candidates.
    """
    best_set = None
    best_score = float('inf')
    
    # Precompute some good AP parameter ranges
    # Split m from 5 to 15 (k = 20 - m)
    splits = list(range(5, 16))
    
    for m in splits:
        k = 20 - m
        # d1 range: keep a0 + (m-1)*d1 <= 10000
        max_d1 = 10000 // max(m, 1)
        # d2 range
        max_d2 = 10000 // max(k, 1)
        
        # Try a grid of starts and steps
        # For efficiency, sample rather than exhaustive
        start_range1 = range(1, min(500, max_d1 + 1), max(1, max_d1 // 50))
        step_range1 = range(1, min(200, max_d1 + 1), max(1, max_d1 // 30))
        start_range2 = range(1, min(500, max_d2 + 1), max(1, max_d2 // 50))
        step_range2 = range(1, min(200, max_d2 + 1), max(1, max_d2 // 30))
        
        # Limit iterations
        count = 0
        for a0 in start_range1:
            if time.time() > deadline - 5:
                break
            for d1 in step_range1:
                if time.time() > deadline - 5:
                    break
                for b0 in start_range2:
                    if time.time() > deadline - 5:
                        break
                    for d2 in step_range2:
                        A = _two_ap_candidate(a0, d1, m, b0, d2, k)
                        if A is None:
                            continue
                        s = _score(A)
                        if s < best_score:
                            best_score = s
                            best_set = A[:]
                        count += 1
                        # Occasionally run local search on improving candidates
                        if s < best_score + 2 and time.time() < deadline - 10:
                            ls_set, ls_score = _local_search_1swap(A, pool, deadline - 5)
                            if ls_score < best_score:
                                best_score = ls_score
                                best_set = ls_set[:]
    
    return best_set, best_score

def _search_structured_simple(deadline, pool):
    """
    Quick structured search: APs, GP-like, smooth number sets.
    """
    best_set = None
    best_score = float('inf')
    
    # Pure APs
    for start in range(1, 30):
        for step in range(1, 500):
            if start + 19 * step > 10000:
                break
            A = [start + i * step for i in range(20)]
            s = _score(A)
            if s < best_score:
                best_score = s
                best_set = A[:]
    
    # 2,3-smooth sets (smallest 20)
    smooth = []
    for a in range(0, 14):
        va = 1
        for _ in range(a):
            va *= 2
        for b in range(0, 14):
            vb = va
            for _ in range(b):
                vb *= 3
            if vb <= 10000:
                smooth.append(vb)
    smooth = sorted(set(smooth))
    if len(smooth) >= 20:
        A = smooth[:20]
        s = _score(A)
        if s < best_score:
            best_score = s
            best_set = A[:]
    
    # Mixed: small AP + large sparse
    for split in range(8, 13):
        for s1 in range(1, 20):
            for d1 in range(1, 30):
                A1 = [s1 + i * d1 for i in range(split)]
                if max(A1) > 1000:
                    continue
                # Fill rest with large smooth-ish numbers
                remaining = 20 - split
                large = []
                for x in range(100, 10000):
                    if x not in A1 and len(large) < remaining:
                        # Prefer numbers with small prime factors
                        if x % 2 == 0 or x % 3 == 0 or x % 5 == 0:
                            large.append(x)
                if len(large) < remaining:
                    continue
                A = sorted(A1 + large)
                if len(set(A)) != 20 or max(A) > 10000:
                    continue
                s = _score(A)
                if s < best_score:
                    best_score = s
                    best_set = A[:]
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    pool = _make_pool()
    
    best_set = None
    best_score = float('inf')
    
    # Phase 1: Quick structured search
    if time.time() < deadline - 20:
        s_set, s_score = _search_structured_simple(deadline - 20, pool)
        if s_set is not None and s_score < best_score:
            best_score = s_score
            best_set = s_set[:]
    
    # Phase 2: Two-AP structural search with local search
    if time.time() < deadline - 10:
        p_set, p_score = _search_two_ap(deadline - 10, pool)
        if p_set is not None and p_score < best_score:
            best_score = p_score
            best_set = p_set[:]
    
    # Phase 3: Final local search refinement on best candidate
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = _score(best_set)
    
    if time.time() < deadline - 2:
        refined, refined_score = _local_search_1swap(best_set, pool, deadline - 1)
        if refined_score < best_score:
            best_set = refined
            best_score = refined_score
    
    return best_set
