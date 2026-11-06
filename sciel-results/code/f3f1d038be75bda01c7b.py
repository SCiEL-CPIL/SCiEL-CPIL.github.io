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
    two_cache = {}

    def map2(mask, da):
        key = (mask, id(da))
        v = two_cache.get(key)
        if v is not None:
            return v
        res = 0
        m = mask
        while m:
            lsb = m & -m
            res |= da[lsb.bit_length() - 1]
            m ^= lsb
        two_cache[key] = res
        return res

    visited = {FULL}
    queue = deque([(FULL, 0)])
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        nm = map2(mask, dest_a)
        if nm not in visited:
            visited.add(nm)
            queue.append((nm, dist + 1))
        nm = map2(mask, dest_b)
        if nm not in visited:
            visited.add(nm)
            queue.append((nm, dist + 1))
    return -1


def construct():
    deadline = time.time() + 84.0

    def cycle_a():
        return list(range(1, N)) + [0]

    def rank2_b(x):
        b = [(i - 1) % N for i in range(N)]
        b[x] = b[x - 1]  # state x merges with x-1, creating a rank-2 subset under 'a'
        return b

    def offset_a(x, off):
        a = cycle_a()
        return [(v + off) % N for v in a]

    def pinwheel_a(k):
        """Pinwheel-style: state i goes to (i+1)%N, except state k goes to 0"""
        a = list(range(1, N)) + [0]
        a[k] = 0
        return a

    def pinwheel_b(k):
        """Pinwheel-style: state i goes to (i-1)%N, except state k goes to 0"""
        b = [(i - 1) % N for i in range(N)]
        b[k] = 0
        return b

    def path_a():
        """Path structure: 0->1->2->...->N-1->0 with a twist"""
        a = list(range(1, N)) + [0]
        return a

    def path_b_with_merge(x, y):
        """Path b with states x and y both going to some state"""
        b = [(i - 1) % N for i in range(N)]
        b[x] = y
        return b

    def eval_ab(a, b):
        if time.time() > deadline:
            return -1
        return _reset_length_bitmask(a, b)

    # Phase 1: Systematic enumeration of structural families
    best_a, best_b, best_len = None, None, -1

    families = []

    # Family 1: offset_a(x, off) + rank2_b(x)
    for x in range(N):
        for off in range(N):
            if time.time() > deadline:
                break
            a = offset_a(x, off)
            b = rank2_b(x)
            families.append((a, b))

    # Family 2: pinwheel_a(k) + pinwheel_b(j)
    for k in range(N):
        for j in range(N):
            if time.time() > deadline:
                break
            a = pinwheel_a(k)
            b = pinwheel_b(j)
            families.append((a, b))

    # Family 3: cycle_a + path_b_with_merge(x, y)
    for x in range(N):
        for y in range(N):
            if time.time() > deadline:
                break
            a = cycle_a()
            b = path_b_with_merge(x, y)
            families.append((a, b))

    # Family 4: pinwheel_a(k) + rank2_b(x)
    for k in range(N):
        for x in range(N):
            if time.time() > deadline:
                break
            a = pinwheel_a(k)
            b = rank2_b(x)
            families.append((a, b))

    # Family 5: offset_a(x, off) + pinwheel_b(j)
    for x in range(N):
        for off in range(N):
            for j in range(N):
                if time.time() > deadline:
                    break
                a = offset_a(x, off)
                b = pinwheel_b(j)
                families.append((a, b))

    # Evaluate all families
    for a, b in families:
        if time.time() > deadline:
            break
        L = eval_ab(a, b)
        if L > best_len:
            best_len, best_a, best_b = L, a, b

    if best_a is None:
        best_a, best_b, best_len = cycle_a(), rank2_b(0), eval_ab(cycle_a(), rank2_b(0))

    # Phase 2: Targeted local search
    def improve(a, b, budget):
        nonlocal best_a, best_b, best_len
        rng = random.Random(12345)
        cur_len = best_len

        # Mutation strategies
        def swap_in_list(lst, i, j):
            nl = list(lst)
            nl[i], nl[j] = nl[j], nl[i]
            return nl

        def single_change(lst, i, val):
            nl = list(lst)
            nl[i] = val
            return nl

        while time.time() < budget:
            improved = False
            for _ in range(60):
                if time.time() > budget:
                    break
                r = rng.random()
                if r < 0.3:
                    # Swap two destinations in b
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    nb = swap_in_list(best_b, i, j)
                    L = eval_ab(best_a, nb)
                    if L > cur_len:
                        best_b, cur_len, improved = nb, L, True
                elif r < 0.5:
                    # Swap two destinations in a
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    na = swap_in_list(best_a, i, j)
                    L = eval_ab(na, best_b)
                    if L > cur_len:
                        best_a, cur_len, improved = na, L, True
                elif r < 0.7:
                    # Change one destination in b
                    i = rng.randrange(N)
                    val = rng.randrange(N)
                    if best_b[i] == val:
                        continue
                    nb = single_change(best_b, i, val)
                    L = eval_ab(best_a, nb)
                    if L > cur_len:
                        best_b, cur_len, improved = nb, L, True
                elif r < 0.9:
                    # Change one destination in a
                    i = rng.randrange(N)
                    val = rng.randrange(N)
                    if best_a[i] == val:
                        continue
                    na = single_change(best_a, i, val)
                    L = eval_ab(na, best_b)
                    if L > cur_len:
                        best_a, cur_len, improved = na, L, True
                else:
                    # Swap letters
                    na, nb = list(best_b), list(best_a)
                    L = eval_ab(na, nb)
                    if L > cur_len:
                        best_a, best_b, cur_len, improved = na, nb, L, True
            if not improved:
                # Try a different seed
                rng = random.Random(rng.randint(0, 1000000))
                if time.time() > budget - 5:
                    break
        best_len = cur_len

    improve(best_a, best_b, deadline)

    return [best_a, best_b]
