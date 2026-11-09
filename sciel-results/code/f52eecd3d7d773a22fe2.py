import time
import random
from math import lcm
from functools import reduce
import itertools

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)

def is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    if not S:
        return True
    S_list = list(S)
    n = len(S_list)
    if n == 0:
        return True
    
    weights = [L // x for x in S_list]
    
    for i, x in enumerate(S_list):
        target = weights[i]
        others = weights[:i] + weights[i+1:]
        
        # Quick bounds check
        if sum(others) < target:
            continue
        if not others:
            continue
        
        # Meet-in-the-middle subset sum
        m = len(others)
        mid = m // 2
        left = others[:mid]
        right = others[mid:]
        
        # Generate all subset sums of left
        left_sums = {0}
        for w in left:
            left_sums = left_sums | {s + w for s in left_sums}
        
        # Check if target - right_sum is in left_sums for any right subset
        found = False
        right_sums = {0}
        for w in right:
            new_sums = {s + w for s in right_sums}
            for s in new_sums:
                if target - s in left_sums:
                    found = True
                    break
            if found:
                break
            right_sums |= new_sums
        
        if found:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Try to find a set of size >= 20 via exhaustive search
    # We go from largest to smallest to maximize size
    for k in range(24, 19, -1):
        if time.time() > deadline - 10: # Keep 10s for heuristic fallback
            break
        for comb in itertools.combinations(UNIVERSE, k):
            if time.time() > deadline - 10:
                break
            if is_valid(comb):
                return list(comb)
        
    # Fallback to heuristic search for best size (likely 19)
    best = []
    best_size = 0
    
    # Fallback
    fallback = list(range(13, 25))
    if is_valid(fallback):
        best = fallback
        best_size = len(fallback)
    
    random.seed(12345)
    
    # Multi-start greedy with various orderings
    orderings = [
        sorted(UNIVERSE, reverse=True),  # Large first
        sorted(UNIVERSE),                # Small first
        list(reversed(UNIVERSE)),
    ]
    for _ in range(15):
        perm = UNIVERSE[:]
        random.shuffle(perm)
        orderings.append(perm)
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = S + [x]
            if is_valid(candidate):
                S = candidate
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local search: try adding any missing element
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for x in UNIVERSE:
            if time.time() > deadline:
                break
            if x not in best:
                candidate = best + [x]
                if is_valid(candidate):
                    best = candidate
                    best_size = len(best)
                    improved = True
    
    # Swap moves: remove one, try adding two different ones
    for _ in range(30):
        if time.time() > deadline - 1:
            break
        S = best[:]
        for i in range(len(S)):
            if time.time() > deadline - 1:
                break
            S_no_i = S[:i] + S[i+1:]
            # Try adding pairs of elements not in S_no_i
            missing = [x for x in UNIVERSE if x not in S_no_i]
            for a_idx in range(len(missing)):
                if time.time() > deadline - 1:
                    break
                for b_idx in range(a_idx + 1, len(missing)):
                    if time.time() > deadline - 1:
                        break
                    candidate = S_no_i + [missing[a_idx], missing[b_idx]]
                    if len(candidate) > best_size and is_valid(candidate):
                        best = candidate
                        best_size = len(best)
                        S = best[:]
                        break
                if len(best) > len(S_no_i):
                    break
            if len(best) > len(S_no_i) + 1:
                break
    
    return best
