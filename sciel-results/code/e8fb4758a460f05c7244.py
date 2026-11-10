"""Rewritten program: Branch-and-bound DFS targeting size 17."""
import time
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L
    L = _lcm_all(universe)
    num = [L // e for e in universe]

    # Order: descending value (largest weights first)
    order = list(range(N - 1, -1, -1))
    vals = [universe[i] for i in order]
    nums = [num[i] for i in order]

    best = []
    best_len = 0
    target_size = 17  # Try to find a set of size 17

    def _time_ok():
        return (time.time() - start_time) < deadline

    import sys
    sys.setrecursionlimit(10000)

    def dfs(pos, cur_list, cur_sums):
        nonlocal best, best_len

        # Prune: cannot reach target size
        remaining = N - pos
        if len(cur_list) + remaining < target_size:
            return
        
        # Prune: time check
        if not _time_ok():
            return

        if pos == N:
            if len(cur_list) > best_len:
                best_len = len(cur_list)
                best = cur_list[:]
                # If we hit target, we can stop early? 
                # But maybe there's a larger one? No, max is 22.
                # If we find 17, we can try to find 18?
                # But 17 is a big jump from 16.
                # Let's just return the first 17 found.
                if len(cur_list) >= target_size:
                    return
            return

        # Option 1: Include element at pos
        n = nums[pos]
        # Check collision
        # Optimization: use a set for cur_sums
        collision = False
        # We can iterate over a list copy if cur_sums is modified? 
        # No, we don't modify cur_sums in place.
        for s in cur_sums:
            if (s + n) in cur_sums:
                collision = True
                break
        
        if not collision:
            new_sums = cur_sums | {s + n for s in cur_sums}
            cur_list.append(vals[pos])
            dfs(pos + 1, cur_list, new_sums)
            cur_list.pop()
            
            # If we found a valid set of target size, we can stop searching this branch?
            # Actually, if we found one, best_len is updated.
            # We can continue to see if we find a larger one?
            # But target_size is 17. If we find 17, we are happy.
            # To save time, if we find 17, we can set a flag to stop?
            # But we might find 18? Unlikely.
            # Let's just let it run until time out or better found.

        if not _time_ok():
            return

        # Option 2: Exclude element at pos
        dfs(pos + 1, cur_list, cur_sums)

    dfs(0, [], {0})

    if best:
        return sorted(best)

    # Fallback
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
