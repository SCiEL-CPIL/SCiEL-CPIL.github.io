"""Branch-and-bound over {1..22} for max dissociated reciprocal set."""
import time
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 88  # safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L  where L = lcm(1..22)
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # integer numerator for each reciprocal

    # Order candidates: Descending value (22, 21, ..., 1)
    # This corresponds to Ascending Numerator (L/22, ..., L)
    # Starting with small reciprocals (light elements) allows packing more elements
    order = list(range(N - 1, -1, -1))  # indices for elements 22, 21, ..., 1
    vals = [universe[i] for i in order]
    nums = [num[i] for i in order]

    best = []
    best_len = 0
    path = []

    def _dfs(pos, cur_sums):
        nonlocal best, best_len, path

        # Check time limit
        if (time.time() - start_time) > deadline:
            return

        # Prune: even if we add all remaining, can't beat best
        if len(path) + (N - pos) <= best_len:
            return

        if pos == N:
            if len(path) > best_len:
                best_len = len(path)
                best = path[:]
            return

        n = nums[pos]
        v = vals[pos]

        # Check collision for including n
        # Collision if (s + n) in cur_sums for any s in cur_sums
        if cur_sums.isdisjoint(s + n for s in cur_sums):
            # No collision, we can include
            # Optimized set update: copy and add
            new_sums = cur_sums.copy()
            for s in cur_sums:
                new_sums.add(s + n)
            
            path.append(v)
            _dfs(pos + 1, new_sums)
            path.pop()

        # Try exclude
        _dfs(pos + 1, cur_sums)

    _dfs(0, {0})

    # Convert best (values) to sorted list
    if best:
        return sorted(best)

    # Fallback: a known dissociated set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
