import time
import random
from collections import deque

N = 12
FULL = (1 << N) - 1


def _reset_length_bitmask(a, b):
    """Exact shortest reset word length via BFS on subset automaton (bitmasks).
    Returns -1 if not synchronizing."""
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]

    # Precompute mapping for all 2^N subsets for both letters
    # map_a[mask] = subset after applying 'a'
    # map_b[mask] = subset after applying 'b'
    map_a = [0] * (1 << N)
    map_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res_a |= dest_a[idx]
            res_b |= dest_b[idx]
            m ^= lsb
        map_a[mask] = res_a
        map_b[mask] = res_b

    visited = [False] * (1 << N)
    visited[FULL] = True
    queue = deque([(FULL, 0)])
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        nm = map_a[mask]
        if not visited[nm]:
            visited[nm] = True
            queue.append((nm, dist + 1))
        nm = map_b[mask]
        if not visited[nm]:
            visited[nm] = True
            queue.append((nm, dist + 1))
    return -1


def construct():
    deadline = time.time() + 85.0

    def cycle_a():
        return list(range(1, N)) + [0]

    def rank2_b(x):
        b = [(i - 1) % N for i in range(N)]
        b[x] = b[x - 1]  # state x merges with x-1, creating a rank-2 subset under 'a'
        return b

    def offset_a(x, off):
        a = cycle_a()
        return [(v + off) % N for v in a]

    def eval_ab(a, b):
        if time.time() > deadline:
            return -1
        return _reset_length_bitmask(a, b)

    # Model: family F(x, offset) = (offset_a(x,offset), rank2_b(x)). Fit (x, offset).
    best_a, best_b, best_len = None, None, -1

    for x in range(N):
        for off in range(N):
            if time.time() > deadline:
                break
            a = offset_a(x, off)
            b = rank2_b(x)
            L = eval_ab(a, b)
            if L > best_len:
                best_len, best_a, best_b = L, a, b

    if best_a is None:
        best_a, best_b, best_len = cycle_a(), rank2_b(0), eval_ab(cycle_a(), rank2_b(0))

    # Light local search: improve the found model via hill-climbing in spare time.
    def improve(a, b, budget):
        nonlocal best_a, best_b, best_len
        rng = random.Random(12345)
        cur_len = best_len
        while time.time() < budget:
            improved = False
            for _ in range(40):
                if time.time() > budget:
                    break
                r = rng.random()
                if r < 0.5:
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    nb = list(best_b); nb[i], nb[j] = nb[j], nb[i]
                    L = eval_ab(best_a, nb)
                    if L > cur_len:
                        best_b, cur_len, improved = nb, L, True
                elif r < 0.9:
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    na = list(best_a); na[i], na[j] = na[j], na[i]
                    L = eval_ab(na, best_b)
                    if L > cur_len:
                        best_a, cur_len, improved = na, L, True
                else:
                    na, nb = list(best_b), list(best_a)  # swap letters
                    L = eval_ab(na, nb)
                    if L > cur_len:
                        best_a, best_b, cur_len, improved = na, nb, L, True
            if not improved:
                break
        best_len = cur_len

    improve(best_a, best_b, deadline)

    return [best_a, best_b]
