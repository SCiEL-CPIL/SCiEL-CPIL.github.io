import time
import random
from collections import deque

N = 12
FULL = (1 << N) - 1
TIME_LIMIT = 90.0


def _reset_length_bitmask(a, b):
    """Exact shortest reset word length via BFS on subset automaton (bitmasks).
    Returns -1 if not synchronizing."""
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]

    # Precompute mapping for all 2^N subsets for both letters
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
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    def cycle_a():
        return list(range(1, N)) + [0]

    def rank2_b(x):
        b = [(i - 1) % N for i in range(N)]
        b[x] = b[x - 1]  # state x merges with x-1
        return b

    def offset_a(x, off):
        a = cycle_a()
        return [(v + off) % N for v in a]

    def eval_ab(a, b):
        if time.time() > deadline:
            return -1
        return _reset_length_bitmask(a, b)

    # Phase 1: Grid search over the model family F(x, offset)
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
        best_a, best_b = cycle_a(), rank2_b(0)
        best_len = eval_ab(best_a, best_b)

    # Phase 2: Local search / hill-climbing to improve
    def improve(a, b, budget):
        nonlocal best_a, best_b, best_len
        rng = random.Random(42)
        cur_len = best_len
        cur_a, cur_b = list(a), list(b)

        while time.time() < budget:
            improved = False
            # Try multiple random perturbations per iteration
            for _ in range(60):
                if time.time() > budget:
                    break
                r = rng.random()
                if r < 0.4:
                    # Swap two destinations in b
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    nb = list(cur_b)
                    nb[i], nb[j] = nb[j], nb[i]
                    L = eval_ab(cur_a, nb)
                    if L > cur_len:
                        cur_b, cur_len, improved = nb, L, True
                elif r < 0.8:
                    # Swap two destinations in a
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    na = list(cur_a)
                    na[i], na[j] = na[j], na[i]
                    L = eval_ab(na, cur_b)
                    if L > cur_len:
                        cur_a, cur_len, improved = na, L, True
                else:
                    # Swap the two letters
                    na, nb = list(cur_b), list(cur_a)
                    L = eval_ab(na, nb)
                    if L > cur_len:
                        cur_a, cur_b, cur_len, improved = na, nb, L, True

            if not improved:
                # No improvement in this round; try a few more with different seeds
                for _ in range(10):
                    if time.time() > budget:
                        break
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    nb = list(cur_b)
                    nb[i], nb[j] = nb[j], nb[i]
                    L = eval_ab(cur_a, nb)
                    if L > cur_len:
                        cur_b, cur_len, improved = nb, L, True
                if not improved:
                    break

        best_a, best_b, best_len = cur_a, cur_b, cur_len

    improve(best_a, best_b, deadline)

    return [best_a, best_b]
