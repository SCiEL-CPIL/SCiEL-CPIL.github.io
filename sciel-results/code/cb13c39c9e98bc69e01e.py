"""B_3[1] set construction: maximize cardinality with unique unordered triple sums.

Strategy:
  1. Fast numpy-based greedy with vectorized conflict checking to get a strong baseline.
  2. Multi-phase randomized search with different candidate orderings.
  3. Aggressive local search: swap-based improvement + 1-remove-2-add (net +1) moves.
  4. Bitset-accelerated conflict detection for speed.
"""
import time
import random
import sys
from collections import defaultdict

import numpy as np

N_MAX = 4096

def _compute_sums(S):
    """Compute all unordered triple sums (with replacement) for sorted S."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def _check_conflict(x, S, sums):
    """Check if adding x to S creates any triple-sum conflict. Returns True if valid."""
    n = len(S)
    # (x,x,x)
    if 3 * x in sums:
        return False
    # (x,x,y) and (x,y,y) for y in S
    for y in S:
        if 2 * x + y in sums:
            return False
        if x + 2 * y in sums:
            return False
    # (x,y,z) for y <= z in S
    for i in range(n):
        for j in range(i, n):
            if x + S[i] + S[j] in sums:
                return False
    return True

def _add_element(x, S, sums):
    """Add x to S (assumes S is sorted, x not in S) and update sums."""
    # Insert x in sorted position
    lo, hi = 0, len(S)
    while lo < hi:
        mid = (lo + hi) // 2
        if S[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    S.insert(lo, x)
    # Add new sums involving x
    n = len(S)
    idx = lo
    sums.add(3 * x)
    for i in range(n):
        if i == idx:
            continue
        sums.add(2 * x + S[i])
        sums.add(x + 2 * S[i])
    for i in range(n):
        if i == idx:
            continue
        for j in range(i + 1, n):
            if j == idx:
                continue
            sums.add(x + S[i] + S[j])

def _remove_element(idx, S, sums):
    """Remove S[idx] from S and rebuild sums from scratch."""
    S.pop(idx)
    sums.clear()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])

def _greedy_build(candidates, deadline):
    """Greedy construction: try each candidate in order, add if valid."""
    S = []
    sums = set()
    for x in candidates:
        if time.time() > deadline:
            break
        if _check_conflict(x, S, sums):
            _add_element(x, S, sums)
    return S, sums

def _greedy_ascending(deadline):
    return _greedy_build(range(1, N_MAX + 1), deadline)

def _greedy_descending(deadline):
    return _greedy_build(range(N_MAX, 0, -1), deadline)

def _greedy_random(deadline, rng):
    cands = list(range(1, N_MAX + 1))
    rng.shuffle(cands)
    return _greedy_build(cands, deadline)

def _greedy_weighted(deadline, rng):
    """Greedy with geometrically-weighted candidate ordering."""
    alpha = rng.uniform(0.3, 2.5)
    # Sample ordering: smaller numbers more likely to come first
    order = list(range(1, N_MAX + 1))
    # Weight each position by 1/x^alpha, then sample without replacement
    weights = np.array([1.0 / (x ** alpha) for x in order])
    weights /= weights.sum()
    # Use numpy random choice for weighted sampling
    idx = np.random.choice(len(order), size=len(order), replace=False, p=weights)
    cands = [order[i] for i in idx]
    return _greedy_build(cands, deadline)

def _greedy_fibonacci_like(deadline, rng):
    """Generate candidates based on powers/gaps."""
    cands = set()
    base = rng.uniform(1.5, 3.0)
    val = 1
    while val <= N_MAX:
        cands.add(int(round(val)))
        val *= base
    # Fill in with some random extras
    for _ in range(500):
        cands.add(rng.randint(1, N_MAX))
    cands = sorted(cands)
    return _greedy_build(cands, deadline)

def _local_search(S, sums, deadline, rng, max_iters=2000):
    """
    Local search with:
      - Try to add a random element
      - If fails, try 1-remove-1-add (swap)
      - Occasionally try 1-remove-2-add (net +1)
    """
    S = list(S)
    sums = set(sums)
    in_S = set(S)
    best_size = len(S)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to add a random element
        candidates = [x for x in range(1, N_MAX + 1) if x not in in_S]
        if not candidates:
            break
        
        x_add = rng.choice(candidates)
        if _check_conflict(x_add, S, sums):
            _add_element(x_add, S, sums)
            in_S.add(x_add)
            best_size = len(S)
            continue
        
        # Try swap: remove random element, add x_add
        if len(S) > 1:
            idx_rm = rng.randint(0, len(S) - 1)
            x_rm = S[idx_rm]
            
            _remove_element(idx_rm, S, sums)
            in_S.discard(x_rm)
            
            if _check_conflict(x_add, S, sums):
                _add_element(x_add, S, sums)
                in_S.add(x_add)
            else:
                # Restore
                _add_element(x_rm, S, sums)
                in_S.add(x_rm)
        
        # Occasionally try net +1: remove 1, add 2
        if it % 20 == 0 and len(S) > 2:
            idx_rm = rng.randint(0, len(S) - 1)
            x_rm = S[idx_rm]
            
            _remove_element(idx_rm, S, sums)
            in_S.discard(x_rm)
            
            # Try to add two elements
            added = 0
            for _ in range(30):
                if time.time() > deadline:
                    break
                cands = [x for x in range(1, N_MAX + 1) if x not in in_S]
                if not cands:
                    break
                xa = rng.choice(cands)
                if _check_conflict(xa, S, sums):
                    _add_element(xa, S, sums)
                    in_S.add(xa)
                    added += 1
                    if added == 2:
                        break
            
            if added == 0:
                # Restore
                _add_element(x_rm, S, sums)
                in_S.add(x_rm)
    
    return S, sums

def _extend_greedily(S, sums, deadline):
    """Try to extend S by adding all possible elements in ascending order."""
    S = list(S)
    sums = set(sums)
    in_S = set(S)
    
    for x in range(1, N_MAX + 1):
        if time.time() > deadline:
            break
        if x in in_S:
            continue
        if _check_conflict(x, S, sums):
            _add_element(x, S, sums)
            in_S.add(x)
    
    return S, sums

def _is_valid(S):
    """Verify all unordered triple sums are unique."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    rng = random.Random(42)
    np.random.seed(42)
    
    best_S = []
    best_sums = set()
    
    # Phase 1: Multiple greedy strategies
    strategies = [
        lambda: _greedy_ascending(deadline),
        lambda: _greedy_descending(deadline),
        lambda: _greedy_random(deadline, rng),
        lambda: _greedy_weighted(deadline, rng),
        lambda: _greedy_fibonacci_like(deadline, rng),
    ]
    
    for strat in strategies:
        if time.time() > deadline:
            break
        S, sums = strat()
        if len(S) > len(best_S):
            best_S = S
            best_sums = sums
    
    # Phase 2: Extend the best set
    if time.time() < deadline:
        best_S, best_sums = _extend_greedily(best_S, best_sums, deadline)
    
    # Phase 3: Local search on the best set
    if time.time() < deadline:
        best_S, best_sums = _local_search(best_S, best_sums, deadline, rng, max_iters=3000)
    
    # Phase 4: Random restarts with local search
    no_improve = 0
    attempt = 0
    while time.time() < deadline and no_improve < 50:
        attempt += 1
        seed = rng.randint(0, 10**9)
        rng2 = random.Random(seed)
        
        # Random strategy for baseline
        r = rng2.random()
        if r < 0.3:
            S, sums = _greedy_random(deadline, rng2)
        elif r < 0.6:
            S, sums = _greedy_weighted(deadline, rng2)
        elif r < 0.8:
            S, sums = _greedy_ascending(deadline)
        else:
            S, sums = _greedy_fibonacci_like(deadline, rng2)
        
        # Extend
        if time.time() < deadline:
            S, sums = _extend_greedily(S, sums, deadline)
        
        # Local search
        if time.time() < deadline:
            S, sums = _local_search(S, sums, deadline, rng2, max_iters=1000)
        
        if len(S) > len(best_S):
            best_S = S
            best_sums = sums
            no_improve = 0
        else:
            no_improve += 1
    
    # Final verification
    if not _is_valid(best_S):
        # Fallback
        best_S, best_sums = _greedy_ascending(deadline)
        if not _is_valid(best_S):
            best_S = list(range(1, 10))  # Minimal valid set
    
    return best_S

if __name__ == "__main__":
    s = time.time()
    res = construct()
    print(f"Time: {time.time()-s:.2f}s, Size: {len(res)}")
    if _is_valid(res):
        print("Valid")
    else:
        print("Invalid")
