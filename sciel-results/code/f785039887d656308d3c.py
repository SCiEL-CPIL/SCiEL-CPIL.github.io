"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys

DEADLINE = 55.0  # safety margin before 60s

def _triple_sums_involving(x, S):
    """Compute all triple sums that include x, given existing sorted set S."""
    n = len(S)
    sums = [3 * x]
    for s in S:
        sums.append(2 * x + s)
        sums.append(x + 2 * s)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + S[i] + S[j])
    return sums

def _check_conflict(x, S, sum_set):
    """Check if adding x to S (with sum_set) causes any triple sum collision."""
    n = len(S)
    # 3x
    if 3 * x in sum_set:
        return True
    # 2x + s and x + 2s
    for s in S:
        if 2 * x + s in sum_set or x + 2 * s in sum_set:
            return True
    # x + s1 + s2
    for i in range(n):
        for j in range(i + 1, n):
            if x + S[i] + S[j] in sum_set:
                return True
    return False

def _greedy_build(candidates, max_size=80):
    """Greedy: try each candidate in order, add if no collision."""
    S = []
    sum_set = set()
    for x in candidates:
        if _check_conflict(x, S, sum_set):
            continue
        S.append(x)
        n = len(S)
        for i in range(n - 1):
            for j in range(i, n):
                sum_set.add(x + S[i] + S[j])
        if len(S) >= max_size:
            break
    return S

def _greedy_linear(start=1, end=4097):
    """Greedy scanning linearly from start to end."""
    return _greedy_build(range(start, end))

def _greedy_exponential(base=3):
    """Greedy with exponential spacing to pack more."""
    candidates = []
    x = 1
    while x <= 4096:
        candidates.append(x)
        x = x * base + 1
    return _greedy_build(candidates)

def _random_greedy(seed_val=42):
    """Random candidate order greedy."""
    rng = random.Random(seed_val)
    candidates = list(range(1, 4097))
    rng.shuffle(candidates)
    S = []
    sum_set = set()
    for x in candidates:
        if _check_conflict(x, S, sum_set):
            continue
        S.append(x)
        n = len(S)
        for i in range(n - 1):
            for j in range(i, n):
                sum_set.add(x + S[i] + S[j])
    return S

def _local_search(S, max_iters=100, deadline=None):
    """Try to improve S by removing one element and adding a different one."""
    if not S:
        return S
    S = list(S)
    best_size = len(S)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        rng = random.Random(hash((it, len(S))))
        
        # Try removing each element and replacing with a random one
        for idx in range(len(S)):
            old = S[idx]
            S.pop(idx)
            sum_set = set()
            # Recompute sum_set for current S
            n = len(S)
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        sum_set.add(S[i] + S[j] + S[k])
            
            # Try to find a replacement
            found = False
            for _ in range(100):  # Try 100 random replacements
                x = rng.randint(1, 4096)
                if x in S:
                    continue
                if _check_conflict(x, S, sum_set):
                    continue
                S.append(x)
                found = True
                improved = True
                break
            
            if not found:
                S.insert(idx, old)
        
        if not improved:
            break
    
    return S

def _verify(S):
    """Verify all unordered triples have unique sums."""
    from itertools import combinations_with_replacement
    sums = [sum(t) for t in combinations_with_replacement(S, 3)]
    return len(sums) == len(set(sums))

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best = []
    
    # Strategy 1: Linear greedy from 1
    S1 = _greedy_linear(1, 4097)
    if len(S1) > len(best):
        best = S1
    if time.time() > deadline - 5:
        return best
    
    # Strategy 2: Exponential greedy
    for base in [2, 3, 4, 5, 7, 10]:
        S2 = _greedy_exponential(base)
        if len(S2) > len(best):
            best = S2
        if time.time() > deadline - 5:
            return best
    
    # Strategy 3: Random greedy with different seeds - main improvement
    seeds = [1, 7, 42, 123, 999, 2024, 31415, 271828, 10001, 10002, 10003, 10004,
             10005, 10006, 10007, 10008, 10009, 10010, 20001, 20002, 20003, 20004,
             30001, 30002, 30003, 30004, 40001, 40002, 40003, 40004]
    for seed_val in seeds:
        if time.time() > deadline - 15:
            break
        S3 = _random_greedy(seed_val)
        if len(S3) > len(best):
            best = S3
    
    # Strategy 4: Local search on best result
    if len(best) > 0 and time.time() < deadline - 10:
        S4 = _local_search(best, max_iters=50, deadline=deadline - 5)
        if len(S4) > len(best):
            best = S4
    
    # Strategy 5: More random restarts + local search if time permits
    if time.time() < deadline - 20:
        for seed_val in [50001, 50002, 50003, 50004, 50005, 50006, 50007, 50008]:
            if time.time() > deadline - 15:
                break
            S5 = _random_greedy(seed_val)
            if len(S5) > len(best):
                best = S5
        if len(best) > 0 and time.time() < deadline - 10:
            S6 = _local_search(best, max_iters=30, deadline=deadline - 5)
            if len(S6) > len(best):
                best = S6
    
    # Final verification of best
    if best and not _verify(best):
        S_fallback = _greedy_linear(1, 4097)
        if _verify(S_fallback):
            best = S_fallback
    
    return best
