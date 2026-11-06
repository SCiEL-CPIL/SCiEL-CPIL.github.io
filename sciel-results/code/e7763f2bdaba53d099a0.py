"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _is_valid_set(S, recips):
    """Check if set S is reciprocal-subsum-free."""
    S_list = sorted(S)
    for a in S_list:
        target = recips[a]
        others = [x for x in S_list if x != a]
        if not others:
            continue
        # Generate all subset sums with pruning
        achievable = {Fraction(0)}
        for x in others:
            r = recips[x]
            new_achievable = set()
            for s in achievable:
                ns = s + r
                if ns <= target:
                    new_achievable.add(ns)
            achievable = achievable | new_achievable
        if target in achievable:
            return False
    return True

def _greedy_construct(order, deadline, recips):
    """Greedy: try to add elements in given order."""
    S = []
    for x in order:
        if _time.time() > deadline:
            break
        candidate = S + [x]
        if _is_valid_set(candidate, recips):
            S = candidate
    return S

def _local_search(S, deadline, recips):
    """Try to improve S by adding or swapping elements."""
    S = list(S)
    improved = True
    while improved and _time.time() < deadline:
        improved = False
        S_set = set(S)
        
        # Try adding each element not in S
        for x in range(1, 25):
            if _time.time() > deadline:
                return S
            if x in S_set:
                continue
            candidate = S + [x]
            if _is_valid_set(candidate, recips):
                S = candidate
                S_set.add(x)
                improved = True
                break
        
        # Try swapping
        if not improved:
            for i, x in enumerate(S):
                if _time.time() > deadline:
                    return S
                for y in range(1, 25):
                    if y in S_set:
                        continue
                    candidate = S[:i] + S[i+1:] + [y]
                    if _is_valid_set(candidate, recips):
                        S = candidate
                        S_set.discard(x)
                        S_set.add(y)
                        improved = True
                        break
                if improved:
                    break
    return S

def construct():
    start_time = _time.time()
    deadline = start_time + 85
    
    recips = {x: Fraction(1, x) for x in range(1, 25)}
    
    # Start with seed
    best = list(range(13, 25))
    
    # Try greedy with various orderings
    orderings = [
        list(range(24, 0, -1)),  # descending
        list(range(1, 25)),       # ascending
        list(range(24, 8, -1)) + list(range(8, 0, -1)),  # large first
    ]
    
    for order in orderings:
        if _time.time() > deadline:
            break
        S = _greedy_construct(order, deadline, recips)
        if len(S) > len(best):
            best = S
    
    # Local search from best
    if _time.time() < deadline:
        best = _local_search(best, deadline, recips)
    
    # Try random orderings
    if _time.time() < deadline - 5:
        random.seed(12345)
        for _ in range(20):
            if _time.time() > deadline:
                break
            order = list(range(1, 25))
            random.shuffle(order)
            S = _greedy_construct(order, deadline, recips)
            if len(S) > len(best):
                best = S
                if _time.time() < deadline:
                    best = _local_search(best, deadline, recips)
    
    if not best:
        best = [13]
    
    return best
# EVOLVE-BLOCK-END
