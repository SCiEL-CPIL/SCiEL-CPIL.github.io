import time
from math import gcd
from functools import reduce

_START_TIME = None
_TIME_LIMIT = 85.0

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _lcm(a, b):
    return a * b // gcd(a, b)

_LCM = reduce(_lcm, range(1, 25))
_V = {i: _LCM // i for i in range(1, 25)}

def _subset_sums(items):
    sums = {0}
    for v in items:
        sums = sums | {s + v for s in sums}
    return sums

def _mim_subset_sum(target, items):
    """Check if target is a subset sum of items using meet-in-the-middle."""
    if target < 0:
        return False
    if target == 0:
        return True
    n = len(items)
    if n == 0:
        return False
    if n <= 12:
        # Brute force
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += items[i]
            if s == target:
                return True
        return False
    half = n // 2
    left = items[:half]
    right = items[half:]
    left_sums = {0}
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
    right_sums = {0}
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
    for l in left_sums:
        if target - l in right_sums:
            return True
    return False

def _is_valid(S):
    if not S:
        return True
    vals = [_V[x] for x in S]
    for i in range(len(S)):
        others = vals[:i] + vals[i+1:]
        if _mim_subset_sum(vals[i], others):
            return False
    return True

def _greedy(order):
    S = []
    full_sums = {0}
    for x in order:
        if _time_exceeded():
            break
        if _V[x] in full_sums:
            continue
        valid = True
        for y in S:
            target = _V[y] - _V[x]
            if target < 0:
                continue
            others = [_V[z] for z in S if z != y]
            if _mim_subset_sum(target, others):
                valid = False
                break
        if not valid:
            continue
        S.append(x)
        full_sums = full_sums | {s + _V[x] for s in full_sums}
    return S

def _local_search(S):
    S = list(S)
    improved = True
    while improved and not _time_exceeded():
        improved = False
        for i in range(len(S)):
            for j in range(1, 25):
                if j in S:
                    continue
                new_S = S[:i] + S[i+1:] + [j]
                new_S.sort()
                if _is_valid(new_S):
                    S = new_S
                    improved = True
                    break
            if improved:
                break
        if not improved:
            for i in range(len(S)):
                for k in range(i+1, len(S)):
                    removed = {S[i], S[k]}
                    base = [x for x in S if x not in removed]
                    for j in range(1, 25):
                        if j in removed or j in base:
                            continue
                        for l in range(j+1, 25):
                            if l in removed or l in base:
                                continue
                            new_S = base + [j, l]
                            new_S.sort()
                            if _is_valid(new_S):
                                S = new_S
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    return S

def _branch_and_bound(order, best_size_ref):
    """B&B with MIM checks on the fly."""
    best = []
    best_sz = best_size_ref[0]
    
    S = []
    full_sums = {0}
    
    def search(idx):
        nonlocal best, best_sz
        if _time_exceeded():
            return
        if idx >= len(order):
            if len(S) > best_sz:
                best_sz = len(S)
                best = S[:]
                best_size_ref[0] = best_sz
            return
        x = order[idx]
        
        remaining = len(order) - idx
        if len(S) + remaining <= best_sz:
            return
        
        # Exclude x
        search(idx + 1)
        
        if _time_exceeded():
            return
        
        # Include x
        if _V[x] in full_sums:
            return
        valid = True
        for y in S:
            target = _V[y] - _V[x]
            if target < 0:
                continue
            others = [_V[z] for z in S if z != y]
            if _mim_subset_sum(target, others):
                valid = False
                break
        if not valid:
            return
        
        S.append(x)
        old_full = full_sums
        full_sums = full_sums | {s + _V[x] for s in full_sums}
        
        search(idx + 1)
        
        S.pop()
        full_sums = old_full
    
    search(0)
    return best

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        list(range(12, 0, -1)) + list(range(24, 12, -1)),
        [x for x in range(1, 25) if x % 2 == 1] + [x for x in range(1, 25) if x % 2 == 0],
        [x for x in range(1, 25) if x % 3 != 0] + [x for x in range(1, 25) if x % 3 == 0],
        sorted(range(1, 25), key=lambda x: -_V[x]),
        sorted(range(1, 25), key=lambda x: _V[x]),
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    ]
    
    best_set = []
    for order in orders:
        if _time_exceeded():
            break
        S = _greedy(order)
        if len(S) > len(best_set):
            best_set = S
    
    if not best_set:
        return [1]
    
    best_set = _local_search(best_set)
    best_size = len(best_set)
    
    # B&B from empty
    best_size_ref = [best_size]
    for bnb_order in [list(range(1, 25)), list(range(24, 0, -1))]:
        if _time_exceeded():
            break
        result = _branch_and_bound(bnb_order, best_size_ref)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    best_set = _local_search(best_set)
    
    if not best_set:
        return [1]
    return best_set
