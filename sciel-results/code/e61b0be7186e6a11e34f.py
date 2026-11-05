from fractions import Fraction
import time
import itertools
from math import gcd
import random

_DEADLINE = None
_BEST = None

def _lcm(a, b):
    return a * b // gcd(a, b)

def _global_lcm():
    l = 1
    for i in range(1, 25):
        l = _lcm(l, i)
    return l

_LCM_24 = _global_lcm()

def _can_form_subset_sum(target_int, elements_int):
    if not elements_int:
        return False
    
    n = len(elements_int)
    total = sum(elements_int)
    if total < target_int:
        return False
    
    mid = n // 2
    left = elements_int[:mid]
    right = elements_int[mid:]
    
    left_sums = set()
    for mask in range(1 << len(left)):
        s = 0
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
        left_sums.add(s)
        
    right_sums = set()
    for mask in range(1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        right_sums.add(s)
        
    for L in left_sums:
        R_needed = target_int - L
        if R_needed in right_sums:
            if not (L == 0 and R_needed == 0):
                return True
    return False

def _init():
    global _DEADLINE, _BEST
    _DEADLINE = time.time() + 80.0
    _BEST = None

def _is_valid_set(S):
    S_list = sorted(S, reverse=True)
    for x in S_list:
        target = _LCM_24 // x
        others = [y for y in S_list if y != x]
        elements_int = [_LCM_24 // y for y in others]
        
        if _can_form_subset_sum(target, elements_int):
            return False
    return True

def _greedy_decreasing():
    selected = []
    for x in range(24, 0, -1):
        candidate = selected + [x]
        if _is_valid_set(candidate):
            selected = candidate
    return selected

def _greedy_increasing():
    selected = []
    for x in range(1, 25):
        candidate = selected + [x]
        if _is_valid_set(candidate):
            selected = candidate
    return selected

def _local_search(S):
    S = list(S)
    all_elements = set(range(1, 25))
    
    while time.time() < _DEADLINE:
        improved = False
        
        # 1. Try to add
        for x in sorted(all_elements - set(S)):
            candidate = S + [x]
            if _is_valid_set(candidate):
                S = candidate
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try to remove one, add two (net +1)
        for i in range(len(S)):
            u = S[i]
            S_minus_u = [x for x in S if x != u]
            excluded = list(all_elements - set(S_minus_u))
            for v in excluded:
                for w in excluded:
                    if v == w: continue
                    new_S = S_minus_u + [v, w]
                    if _is_valid_set(new_S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
        if improved:
            continue
            
        # 3. Stuck. Try a random 1-for-1 swap to escape local optimum.
        # Pick random u in S, random v not in S.
        if S:
            u = random.choice(S)
            excluded = list(all_elements - set(S))
            if excluded:
                v = random.choice(excluded)
                new_S = [x for x in S if x != u] + [v]
                if _is_valid_set(new_S):
                    S = new_S
                    # Don't mark as improved to allow further search
                    # But we changed state, so loop continues
        else:
            break
            
    return S

def construct():
    _init()
    
    candidates = []
    
    # Try different starts
    candidates.append(_greedy_decreasing())
    candidates.append(_greedy_increasing())
    
    # Try a few random starts
    for _ in range(5):
        random_set = random.sample(range(1, 25), 10)
        candidates.append(random_set)
        
    best = []
    for c in candidates:
        if not c:
            continue
        # Ensure start is valid? Greedy ensures it. Random might not.
        if not _is_valid_set(c):
            # Filter to make it valid? Just skip or repair?
            # Let's just skip invalid random starts or repair by removing elements.
            # Simpler: skip.
            continue
            
        searched = _local_search(c)
        if len(searched) > len(best):
            best = searched
            
    if not best:
        best = [24]
        
    _BEST = best
    return list(_BEST)
