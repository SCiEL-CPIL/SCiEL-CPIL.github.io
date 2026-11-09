import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)

def is_valid(S):
    """Check if set S is reciprocal-subsum-free using meet-in-the-middle."""
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
        if not others:
            continue
        if sum(others) < target:
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
    deadline = time.time() + 85  # 85s to leave safety margin
    
    best = list(range(13, 25))  # Fallback: size 12
    if not is_valid(best):
        best = [24]
    
    random.seed(42)
    
    # Strategy 1: Greedy construction with various orderings
    orders = [
        sorted(UNIVERSE, reverse=True),  # Large first
        sorted(UNIVERSE),                # Small first
    ]
    # Add random permutations
    for _ in range(10):
        perm = UNIVERSE[:]
        random.shuffle(perm)
        orders.append(perm)
    
    for order in orders:
        if time.time() > deadline:
            break
        S = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = S + [x]
            if is_valid(candidate):
                S = candidate
        if len(S) > len(best):
            best = S
    
    # Strategy 2: Local search - try to add elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in UNIVERSE:
            if time.time() > deadline:
                break
            if x not in best:
                candidate = best + [x]
                if is_valid(candidate):
                    best = candidate
                    improved = True
    
    # Strategy 3: Swap moves - remove one, try adding a different one
    for _ in range(50):
        if time.time() > deadline:
            break
        made_swap = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            S_no_i = best[:i] + best[i+1:]
            for x in UNIVERSE:
                if time.time() > deadline:
                    break
                if x not in S_no_i:
                    candidate = S_no_i + [x]
                    if len(candidate) > len(best) and is_valid(candidate):
                        best = candidate
                        made_swap = True
                        break
            if made_swap:
                break
        if not made_swap:
            break
    
    # Strategy 4: Remove one and add two different ones
    for _ in range(30):
        if time.time() > deadline:
            break
        S = best[:]
        random.shuffle(S)
        for i in range(len(S)):
            if time.time() > deadline:
                break
            S_no_i = S[:i] + S[i+1:]
            # Try adding pairs of elements not in S_no_i
            missing = [x for x in UNIVERSE if x not in S_no_i]
            for a in missing:
                if time.time() > deadline:
                    break
                for b in missing:
                    if time.time() > deadline:
                        break
                    if a == b:
                        continue
                    candidate = S_no_i + [a, b]
                    if len(candidate) > len(best) and is_valid(candidate):
                        best = candidate
                        S = best[:]
                        break
                if len(best) > len(S_no_i):
                    break
            if len(best) > len(S_no_i):
                break
    
    return best
