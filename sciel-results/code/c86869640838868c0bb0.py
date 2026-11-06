import time
import random

def _check_sidon(S):
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def _build_sidon(pool_set, max_n=256, max_size=64, start=1, mod=0):
    """
    Build a Sidon set whose differences are all in pool_set.
    Uses the recurrence x_{k+1} = 2*x_k - x_{k-1} (arithmetic progression)
    but with a twist: we pick step s such that all multiples of s up to max_n
    are in pool_set. The set is {start, start+s, start+2s, ...} within [1,max_n].
    Differences are multiples of s: s, 2s, 3s, ...
    """
    if not pool_set:
        return []
    
    # Find the largest step s such that all multiples of s up to max_n are in pool_set
    # and the resulting set size is maximized
    best_set = []
    best_start = 1
    
    # Try each possible step
    for s in range(1, max_n + 1):
        # Check that all multiples of s (that could appear as differences) are in pool_set
        # The differences will be s, 2s, 3s, ... up to (k-1)*s where k is the set size
        # We need (k-1)*s <= max_n - start, and all j*s for j=1..k-1 must be in pool_set
        # Actually, differences are |a_i - a_j| = |i-j|*s for i,j in the set
        # So we need all multiples of s up to (k-1)*s to be in pool_set
        
        # Find max k such that (k-1)*s <= max_n - start and all j*s in pool_set for j=1..k-1
        # Start with start=1 for simplicity, but we can also try other starts
        
        for st in range(1, min(max_n - s * 2 + 1, 50) + 1):  # Try a few starts
            # Max size with this start and step
            max_k = (max_n - st) // s + 1
            if max_k < 2:
                continue
            
            # Check that all differences j*s for j=1..max_k-1 are in pool_set
            ok = True
            for j in range(1, max_k):
                if j * s not in pool_set:
                    ok = False
                    break
            
            if ok and max_k > len(best_set):
                # Build the set
                S = [st + i * s for i in range(max_k)]
                if len(S) <= max_size:
                    best_set = S
                    best_start = st
                else:
                    # Truncate to max_size
                    S = [st + i * s for i in range(max_size)]
                    if len(S) > len(best_set):
                        best_set = S
                        best_start = st
    
    return best_set

def _build_sidon_smart(pool_set, max_n=256, max_size=64):
    """
    Smarter builder: try to find a Sidon set (not just arithmetic progression)
    whose differences are all in pool_set.
    Uses greedy: pick elements one by one, ensuring all new differences are in pool_set
    and Sidon property holds.
    """
    if not pool_set:
        return []
    
    S = []
    sums = set()
    
    for x in range(1, max_n + 1):
        if len(S) >= max_size:
            break
        
        # Check Sidon
        ok = True
        for a in S:
            if (x + a) in sums:
                ok = False
                break
        if ok and (x + x) in sums:
            ok = False
        
        # Check differences
        if ok:
            for a in S:
                d = abs(x - a)
                if d > 0 and d not in pool_set:
                    ok = False
                    break
        
        if ok:
            # Add x
            for a in S:
                sums.add(x + a)
            sums.add(x + x)
            S.append(x)
    
    return S

def _build_sidon_greedy(pool_set, max_n=256, max_size=64, order='asc'):
    """
    Greedy Sidon builder with difference constraint.
    order: 'asc' or 'desc'
    """
    if not pool_set:
        return []
    
    S = []
    sums = set()
    diffs_used = set()
    
    candidates = list(range(1, max_n + 1))
    if order == 'desc':
        candidates.reverse()
    
    for x in candidates:
        if len(S) >= max_size:
            break
        
        # Check Sidon
        ok = True
        for a in S:
            if (x + a) in sums:
                ok = False
                break
        if ok and (x + x) in sums:
            ok = False
        
        # Check differences are in pool_set
        if ok:
            for a in S:
                d = abs(x - a)
                if d > 0 and d not in pool_set:
                    ok = False
                    break
        
        if ok:
            for a in S:
                sums.add(x + a)
            sums.add(x + x)
            S.append(x)
    
    return S

def _partition_and_build(pool_A, pool_B, max_n=256, max_size=64):
    """Build Sidon sets from the two difference pools."""
    A = _build_sidon_greedy(pool_A, max_n, max_size, 'asc')
    B = _build_sidon_greedy(pool_B, max_n, max_size, 'asc')
    
    # Also try descending
    A2 = _build_sidon_greedy(pool_A, max_n, max_size, 'desc')
    B2 = _build_sidon_greedy(pool_B, max_n, max_size, 'desc')
    
    # Pick the better
    if len(A2) > len(A):
        A = A2
    if len(B2) > len(B):
        B = B2
    
    return A, B

def _partition_greedy(max_n=256, max_size=64, rng=None):
    """
    Greedy partitioning: assign differences to pools to maximize combined Sidon size.
    Strategy: for each difference d, assign to the pool where it's "most useful".
    """
    if rng is None:
        rng = random.Random(42)
    
    pool_A = set()
    pool_B = set()
    
    # Assign differences in a smart order: small differences first (they're more constraining)
    diffs = list(range(1, max_n))
    rng.shuffle(diffs)
    
    for d in diffs:
        # Try adding to A, see if it helps
        # Heuristic: assign to the pool that currently has fewer differences
        # but also consider which pool benefits more
        if len(pool_A) <= len(pool_B):
            pool_A.add(d)
        else:
            pool_B.add(d)
    
    return pool_A, pool_B

def _partition_balanced(max_n=256, max_size=64, rng=None):
    """
    Balanced partitioning: split differences roughly evenly, but with structure.
    Assign multiples of certain steps to each pool.
    """
    if rng is None:
        rng = random.Random(123)
    
    pool_A = set()
    pool_B = set()
    
    # Strategy: assign each difference to A or B based on a random but structured rule
    # Try several rules and pick the best
    best_score = 0
    best_pools = (set(), set())
    
    for trial in range(20):
        pA = set()
        pB = set()
        
        # Rule: assign d to A if (d mod p) < q for random p, q
        p = rng.randint(2, 10)
        q = rng.randint(1, p - 1)
        
        for d in range(1, max_n):
            if d % p < q:
                pA.add(d)
            else:
                pB.add(d)
        
        A, B = _partition_and_build(pA, pB, max_n, max_size)
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_pools = (pA, pB)
    
    return best_pools

def _local_search_partition(pool_A, pool_B, A, B, max_n=256, max_size=64, 
                            time_budget=30.0, rng=None):
    """
    Local search: move differences between pools, rebuild sets, keep if improved.
    """
    if rng is None:
        rng = random.Random(0)
    
    start_time = time.time()
    pool_A = set(pool_A)
    pool_B = set(pool_B)
    best_A = list(A)
    best_B = list(B)
    best_score = len(A) + len(B)
    
    all_diffs = set(range(1, max_n))
    
    while time.time() - start_time < time_budget:
        # Try moving a random difference from A to B or B to A
        moved = False
        
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            
            d = rng.choice(list(all_diffs))
            
            if d in pool_A:
                pool_A.remove(d)
                pool_B.add(d)
                new_A, new_B = _partition_and_build(pool_A, pool_B, max_n, max_size)
                score = len(new_A) + len(new_B)
                if score > best_score:
                    best_score = score
                    best_A = list(new_A)
                    best_B = list(new_B)
                    moved = True
                else:
                    # Revert
                    pool_B.remove(d)
                    pool_A.add(d)
            elif d in pool_B:
                pool_B.remove(d)
                pool_A.add(d)
                new_A, new_B = _partition_and_build(pool_A, pool_B, max_n, max_size)
                score = len(new_A) + len(new_B)
                if score > best_score:
                    best_score = score
                    best_A = list(new_A)
                    best_B = list(new_B)
                    moved = True
                else:
                    # Revert
                    pool_A.remove(d)
                    pool_B.add(d)
            else:
                # d is unused, assign to A or B
                if rng.random() < 0.5:
                    pool_A.add(d)
                    new_A, new_B = _partition_and_build(pool_A, pool_B, max_n, max_size)
                    score = len(new_A) + len(new_B)
                    if score > best_score:
                        best_score = score
                        best_A = list(new_A)
                        best_B = list(new_B)
                        moved = True
                    else:
                        pool_A.remove(d)
                else:
                    pool_B.add(d)
                    new_A, new_B = _partition_and_build(pool_A, pool_B, max_n, max_size)
                    score = len(new_A) + len(new_B)
                    if score > best_score:
                        best_score = score
                        best_A = list(new_A)
                        best_B = list(new_B)
                        moved = True
                    else:
                        pool_B.remove(d)
        
        if not moved:
            # Try a batch move
            if time.time() - start_time > time_budget:
                break
            for _ in range(10):
                d1 = rng.choice(list(all_diffs))
                d2 = rng.choice(list(all_diffs))
                if d1 == d2:
                    continue
                if d1 in pool_A and d2 in pool_B:
                    pool_A.remove(d1)
                    pool_B.add(d1)
                    pool_B.remove(d2)
                    pool_A.add(d2)
                    new_A, new_B = _partition_and_build(pool_A, pool_B, max_n, max_size)
                    score = len(new_A) + len(new_B)
                    if score > best_score:
                        best_score = score
                        best_A = list(new_A)
                        best_B = list(new_B)
                    else:
                        # Revert
                        pool_A.add(d1)
                        pool_B.remove(d1)
                        pool_B.add(d2)
                        pool_A.remove(d2)
    
    return best_A, best_B, best_score

def _greedy_fallback(max_n=256, max_size=64):
    """Simple greedy fallback."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in range(1, max_n + 1):
        # Try A
        if len(A) < max_size:
            ok = True
            for a in A:
                if (x + a) in sums_A:
                    ok = False
                    break
            if ok and (x + x) in sums_A:
                ok = False
            if ok:
                for a in A:
                    d = abs(x - a)
                    if d > 0 and d in diffs_B:
                        ok = False
                        break
            if ok:
                for a in A:
                    sums_A.add(x + a)
                sums_A.add(x + x)
                for a in A:
                    d = abs(x - a)
                    if d > 0:
                        diffs_A.add(d)
                A.append(x)
                continue
        
        # Try B
        if len(B) < max_size:
            ok = True
            for b in B:
                if (x + b) in sums_B:
                    ok = False
                    break
            if ok and (x + x) in sums_B:
                ok = False
            if ok:
                for b in B:
                    d = abs(x - b)
                    if d > 0 and d in diffs_A:
                        ok = False
                        break
            if ok:
                for b in B:
                    sums_B.add(x + b)
                sums_B.add(x + x)
                for b in B:
                    d = abs(x - b)
                    if d > 0:
                        diffs_B.add(d)
                B.append(x)
    
    return A, B

def construct():
    """
    Main: difference-pool partitioning + structured Sidon completion + local search.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Phase 1: Try several partition strategies
    best_A = [1]
    best_B = [3]
    best_score = 2
    
    # Strategy 1: Balanced partition with random modular rules
    if time.time() < deadline - 40.0:
        pA, pB = _partition_balanced(256, 64, random.Random(42))
        A, B = _partition_and_build(pA, pB, 256, 64)
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Strategy 2: Greedy partition
    if time.time() < deadline - 30.0:
        pA, pB = _partition_greedy(256, 64, random.Random(99))
        A, B = _partition_and_build(pA, pB, 256, 64)
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Strategy 3: Direct greedy (fallback)
    if time.time() < deadline - 20.0:
        A, B = _greedy_fallback(256, 64)
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Phase 2: Local search on partition
    if time.time() < deadline - 5.0:
        # Reconstruct pools from best solution
        dA = _diffs(best_A)
        dB = _diffs(best_B)
        pA = set(dA)
        pB = set(dB)
        
        ls_budget = min(35.0, deadline - 5.0 - time.time())
        if ls_budget > 2.0:
            A_ls, B_ls, score_ls = _local_search_partition(
                pA, pB, best_A, best_B, 256, 64, ls_budget, random.Random(777)
            )
            valid_ls, score_ls_val = _validate(A_ls, B_ls)
            if valid_ls and score_ls_val > best_score:
                best_score = score_ls_val
                best_A, best_B = list(A_ls), list(B_ls)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
