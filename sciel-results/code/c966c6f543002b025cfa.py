"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from fractions import Fraction

def _check_conflict(x, others):
    """Check if 1/x equals sum of reciprocals of some nonempty subset of others."""
    target = Fraction(1, x)
    others_list = list(others)
    n = len(others_list)
    if n == 0:
        return False
    
    # For small n, brute force
    if n <= 15:
        for r in range(1, n + 1):
            for subset in combinations(others_list, r):
                s = sum(Fraction(1, y) for y in subset)
                if s == target:
                    return True
                if s > target:
                    # Since all terms are positive, no need to continue with this r
                    break
    else:
        # Meet in the middle
        mid = n // 2
        left = others_list[:mid]
        right = others_list[mid:]
        
        # Generate all subset sums of left
        left_sums = set()
        nl = len(left)
        for mask in range(1, 1 << nl):
            s = Fraction(0)
            for i in range(nl):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
                    if s > target:
                        break
            if s <= target:
                left_sums.add(s)
        
        # Check if target is in left_sums
        if target in left_sums:
            return True
        
        # Generate all subset sums of right and check
        nr = len(right)
        for mask in range(1, 1 << nr):
            s = Fraction(0)
            for i in range(nr):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
                    if s > target:
                        break
            if s <= target:
                if (target - s) in left_sums:
                    return True
    
    return False

def _is_valid(S):
    """Check if set S is valid (no reciprocal subsum)."""
    S_set = set(S)
    for x in S:
        others = S_set - {x}
        if _check_conflict(x, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best = list(range(13, 25))  # Initial: 13 elements
    best_size = len(best)
    
    random.seed(42)
    
    # Phase 1: Greedy construction from different starting points
    for attempt in range(30):
        if time.time() > deadline - 15:
            break
        
        if attempt == 0:
            current = set(range(10, 25))
        elif attempt == 1:
            current = set(range(1, 25))
        elif attempt == 2:
            current = set(range(1, 25, 2))
        elif attempt == 3:
            current = set(range(2, 25, 2))
        elif attempt == 4:
            current = set([2, 3, 5, 7, 11, 13, 17, 19, 23])
        elif attempt == 5:
            current = set(range(9, 25))
        elif attempt == 6:
            current = set(range(6, 25))
        elif attempt == 7:
            current = set(range(4, 25))
        elif attempt == 8:
            current = set(range(2, 25))
        elif attempt == 9:
            current = set(range(3, 25))
        elif attempt == 10:
            current = set(range(5, 25))
        elif attempt == 11:
            current = set(range(7, 25))
        elif attempt == 12:
            current = set(range(8, 25))
        elif attempt == 13:
            current = set(range(11, 25))
        elif attempt == 14:
            current = set(range(12, 25))
        elif attempt == 15:
            current = set(range(14, 25))
        elif attempt == 16:
            current = set(range(15, 25))
        elif attempt == 17:
            current = set(range(16, 25))
        elif attempt == 18:
            current = set(range(17, 25))
        elif attempt == 19:
            current = set(range(18, 25))
        elif attempt == 20:
            current = set(range(19, 25))
        elif attempt == 21:
            current = set(range(20, 25))
        elif attempt == 22:
            current = set(range(21, 25))
        elif attempt == 23:
            current = set(range(22, 25))
        elif attempt == 24:
            current = set(range(23, 25))
        elif attempt == 25:
            current = set(range(24, 25))
        else:
            current = set(random.sample(range(1, 25), random.randint(10, 20)))
        
        # Remove violations iteratively
        changed = True
        while changed:
            changed = False
            for x in list(current):
                others = current - {x}
                if _check_conflict(x, others):
                    current.remove(x)
                    changed = True
                    break
        
        if len(current) > best_size:
            best = list(current)
            best_size = len(current)
        
        # Try to add elements greedily
        for _ in range(200):
            if time.time() > deadline - 15:
                break
            candidates = [x for x in range(1, 25) if x not in current]
            if not candidates:
                break
            random.shuffle(candidates)
            added = False
            for x in candidates:
                new_set = current | {x}
                if _is_valid(new_set):
                    current = new_set
                    added = True
                    break
            if not added:
                break
        
        if len(current) > best_size:
            best = list(current)
            best_size = len(current)
    
    # Phase 2: Local search on best found
    if time.time() < deadline - 10:
        current = set(best)
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            improved = False
            # Try remove-1-add-1
            for x in list(current):
                if time.time() > deadline - 10:
                    break
                new_set = current - {x}
                for y in range(1, 25):
                    if y not in new_set:
                        test_set = new_set | {y}
                        if _is_valid(test_set):
                            current = test_set
                            improved = True
                            break
                if improved:
                    break
            if len(current) > best_size:
                best = list(current)
                best_size = len(current)
            if not improved:
                break
    
    # Final check
    if not _is_valid(best):
        for x in best[:]:
            if _check_conflict(x, set(best) - {x}):
                best.remove(x)
    
    return best
# EVOLVE-BLOCK-END
