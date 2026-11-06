import time
from math import lcm
from functools import reduce

def construct():
    start = time.time()
    deadline = start + 85.0

    L = reduce(lcm, range(1, 23))
    recip = {i: L // i for i in range(1, 23)}

    # order: try larger i first (smaller reciprocal) to allow bigger sets
    order = list(range(22, 0, -1))

    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_len = len(best)

    def can_add(sums, r):
        # collision if exists s1,s2 in sums with s1 - s2 == r  (i.e., s2 + r in sums)
        for s in sums:
            if s + r in sums:
                return False
        return True

    def dfs(idx, sums, chosen):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        # prune: even adding all remaining can't beat best
        remaining = len(order) - idx
        if len(chosen) + remaining <= best_len:
            return
        if idx == len(order):
            if len(chosen) > best_len:
                best = list(chosen)
                best_len = len(chosen)
            return
        i = order[idx]
        r = recip[i]
        # try include
        if can_add(sums, r):
            new_sums = sums | {s + r for s in sums}
            chosen.append(i)
            dfs(idx + 1, new_sums, chosen)
            chosen.pop()
        # try exclude
        dfs(idx + 1, sums, chosen)

    dfs(0, {0}, [])
    return best
