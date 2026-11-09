import time
import random
from math import lcm
from functools import reduce
from itertools import combinations

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
WEIGHTS = {x: L // x for x in UNIVERSE}

def subset_sum_check(weights_list, target):
    """Check if target can be formed as a sum of a subset of weights_list.
    Uses meet-in-the-middle for efficiency."""
    n = len(weights_list)
    if n == 0:
        return False
    if target == 0:
        return False  # We need non-empty subset, and target > 0 always
    
    # Quick bounds
    total = sum(weights_list)
    if total < target:
        return False
    
    # If any single element equals target
    for w in weights_list:
        if w == target:
            return True
    
    if n <= 1:
        return False
    
    mid = n // 2
    left = weights_list[:mid]
    right = weights_list[mid:]
    
    # Generate all subset sums of left
    left_sums = set()
    for mask in range(1 << len(left)):
        s = 0
        for j in range(len(left)):
            if mask & (1 << j):
                s += left[j]
        left_sums.add(s)
    
    # Check if any right subset sum plus a left subset sum equals target
    for mask in range(1 << len(right)):
        s = 0
        for j in range(len(right)):
            if mask & (1 << j):
                s += right[j]
        if target - s in left_sums:
            return True
    
    return False

def is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    if len(S) <= 1:
        return True
    
    for x in S:
        target = WEIGHTS[x]
        others = [WEIGHTS[y] for y in S if y != x]
        if subset_sum_check(others, target):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin
    
    best = set()
    
    # Precompute all valid subsets up to certain size for fast lookup?
    # Instead, let's do systematic search.
    
    # Strategy 1: Multiple greedy passes with different orderings
    orders = []
    # Descending
    orders.append(list(reversed(UNIVERSE)))
    # Ascending
    orders.append(list(UNIVERSE))
    # Random orders
    for _ in range(30):
        perm = list(UNIVERSE)
        random.shuffle(perm)
        orders.append(perm)
    # Some structured orders: even then odd, multiples of 3, etc.
    orders.append([x for x in UNIVERSE if x % 2 == 0] + [x for x in UNIVERSE if x % 2 == 1])
    orders.append([x for x in UNIVERSE if x % 3 == 0] + [x for x in UNIVERSE if x % 3 != 0])
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        for x in order:
            if time.time() > deadline:
                break
            cand = current + [x]
            if is_valid(cand):
                current = cand
        if len(current) > len(best):
            best = set(current)
    
    # Strategy 2: Local search - try adding elements, swapping
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        
        # Try adding any single missing element
        for x in UNIVERSE:
            if time.time() > deadline:
                break
            if x not in best:
                cand = list(best) + [x]
                if is_valid(cand):
                    best.add(x)
                    improved = True
                    break
        
        # Try swapping: remove one, add one
        for x in list(best):
            if time.time() > deadline:
                break
            for y in UNIVERSE:
                if y in best or y == x:
                    continue
                cand = [z for z in best if z != x] + [y]
                if is_valid(cand):
                    best.remove(x)
                    best.add(y)
                    improved = True
                    break
            if improved:
                break
    
    # Strategy 3: Try to add two elements at once (remove one, add two)
    if time.time() < deadline - 2:
        improved2 = True
        while improved2 and time.time() < deadline - 2:
            improved2 = False
            for x in list(best):
                if time.time() > deadline:
                    break
                remaining = [z for z in best if z != x]
                for y in UNIVERSE:
                    if y in best or y == x:
                        continue
                    for z in UNIVERSE:
                        if z in best or z == x or z == y:
                            continue
                        if y > z:
                            continue
                        cand = remaining + [y, z]
                        if is_valid(cand):
                            best.remove(x)
                            best.add(y)
                            best.add(z)
                            improved2 = True
                            break
                    if improved2:
                        break
                if improved2:
                    break
    
    return list(best)
