import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 82  # wall-clock limit is 90s; keep a safety margin

    N = 128

    # 1. Precompute GCD matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # 2. Precompute conflict structure.
    #    For a bad triple {a,b,c} (all pairwise gcds equal), each element's
    #    "partner set" records the other two. x is addable iff no pair of its
    #    partners is simultaneously present in the current set.
    partners = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gab == gcd_mat[a][c] and gab == gcd_mat[b][c]:
                    partners[a].add(b)
                    partners[a].add(c)
                    partners[b].add(a)
                    partners[b].add(c)
                    partners[c].add(a)
                    partners[c].add(b)

    # degree[x] = number of bad triples containing x (fewest = least constrained)
    degree = [0] * (N + 1)
    for x in range(1, N + 1):
        degree[x] = len(partners[x])

    def can_add(x, in_set):
        """O(n) with early break: x is addable iff no partner-pair is present."""
        for a in partners[x]:
            if in_set[a]:
                for b in partners[x]:
                    if b != a and in_set[b]:
                        return False
        return True

    def try_add_all(in_set, mask):
        """Add every addable element (one pass is sufficient). Returns (mask, in_set, added)."""
        added = 0
        for x in range(1, N + 1):
            if not in_set[x] and can_add(x, in_set):
                in_set[x] = True
                mask |= (1 << x)
                added += 1
        return mask, in_set, added

    def greedy_order(order, in_set, mask):
        for x in order:
            if not in_set[x] and can_add(x, in_set):
                in_set[x] = True
                mask |= (1 << x)
        return mask, in_set

    def local_search(mask, in_set):
        best_mask = mask
        best_size = bin(mask).count('1')

        # initial polish: add everything addable
        mask, in_set, _ = try_add_all(in_set, mask)
        best_mask = mask
        best_size = bin(mask).count('1')

        while time.time() < deadline - 1:
            # 1. try to add any element
            new_mask, new_in_set, added = try_add_all(in_set[:], mask)
            if added > 0:
                in_set = new_in_set
                mask = new_mask
                s = bin(mask).count('1')
                if s > best_size:
                    best_size = s
                    best_mask = mask
                continue

            current = [i for i in range(1, N + 1) if in_set[i]]
            if not current:
                break

            found = False

            # 2. 1-element kicks
            for _ in range(30):
                if time.time() > deadline - 1:
                    break
                u = random.choice(current)
                old_in_set = in_set[:]
                old_mask = mask
                in_set[u] = False
                mask &= ~(1 << u)
                new_mask, new_in_set, _ = try_add_all(in_set, mask)
                ns = bin(new_mask).count('1')
                if ns > best_size:
                    best_size = ns
                    best_mask = new_mask
                    in_set = new_in_set
                    mask = new_mask
                    found = True
                    break
                else:
                    in_set = old_in_set
                    mask = old_mask
            if found:
                continue

            # 3. 2-element kicks
            if len(current) >= 2:
                for _ in range(20):
                    if time.time() > deadline - 1:
                        break
                    u, v = random.sample(current, 2)
                    old_in_set = in_set[:]
                    old_mask = mask
                    in_set[u] = False
                    in_set[v] = False
                    mask &= ~(1 << u)
                    mask &= ~(1 << v)
                    new_mask, new_in_set, _ = try_add_all(in_set, mask)
                    ns = bin(new_mask).count('1')
                    if ns > best_size:
                        best_size = ns
                        best_mask = new_mask
                        in_set = new_in_set
                        mask = new_mask
                        found = True
                        break
                    else:
                        in_set = old_in_set
                        mask = old_mask
            if found:
                continue

            break  # stuck at a local optimum for this seed

        return best_mask

    best_mask = 0
    best_size = 0

    # Strategy A: minimum-degree (least-constrained) greedy order
    in_set = [False] * (N + 1)
    mask = 0
    order = sorted(range(1, N + 1), key=lambda x: (degree[x], x))
    mask, in_set = greedy_order(order, in_set, mask)
    m = local_search(mask, in_set)
    if bin(m).count('1') > best_size:
        best_mask = m
        best_size = bin(m).count('1')

    # Strategy B: reverse minimum-degree
    if time.time() < deadline - 2:
        in_set = [False] * (N + 1)
        mask = 0
        order = sorted(range(1, N + 1), key=lambda x: (degree[x], -x))
        mask, in_set = greedy_order(order, in_set, mask)
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')

    # Strategy C: natural order
    if time.time() < deadline - 2:
        in_set = [False] * (N + 1)
        mask = 0
        mask, in_set = greedy_order(range(1, N + 1), in_set, mask)
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')

    # Strategy D: randomized restarts (degree-sorted + pure random)
    while time.time() < deadline - 2:
        in_set = [False] * (N + 1)
        mask = 0
        if random.random() < 0.5:
            order = sorted(range(1, N + 1), key=lambda x: (degree[x], random.random()))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
        mask, in_set = greedy_order(order, in_set, mask)
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')

    # Final high-precision polish: add anything addable to the best set
    if best_size > 0 and time.time() < deadline - 0.5:
        in_set = [False] * (N + 1)
        for i in range(1, N + 1):
            if best_mask & (1 << i):
                in_set[i] = True
        pm, pi, _ = try_add_all(in_set, best_mask)
        if bin(pm).count('1') > best_size:
            best_mask = pm
            best_size = bin(pm).count('1')

    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    if not result:
        result = [1]
    return result
