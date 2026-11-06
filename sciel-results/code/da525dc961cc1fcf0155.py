import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0

    N = 256

    # Build conflict sets
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        s2 = 2 * i
        r = int(math.isqrt(s2))
        if r * r == s2:
            conflicts[i].add(i)
        for j in range(i + 1, N + 1):
            s = i + j
            r = int(math.isqrt(s))
            if r * r == s:
                conflicts[i].add(j)
                conflicts[j].add(i)

    def greedy_construct(order):
        S = set()
        for v in order:
            if not (conflicts[v] & S):
                S.add(v)
        return S

    best_S = set()
    best_size = 0

    # Try natural order first
    S = greedy_construct(range(1, N + 1))
    best_S = S.copy()
    best_size = len(S)

    # Try many random orderings
    time_limit = deadline - 2.0
    while time.time() < time_limit:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()

    result = sorted(best_S)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
