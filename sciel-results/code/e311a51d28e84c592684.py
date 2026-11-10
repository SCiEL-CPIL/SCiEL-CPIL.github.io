def construct():
    import time
    from math import gcd
    import random

    start_time = time.time()
    DEADLINE = 85.0

    n = 128
    g = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]

    # Build conflict lists: conflicts[x] = list of (a,b) pairs that with x form a bad triple
    conflicts = [[] for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            gab = g[a][b]
            for c in range(b+1, n+1):
                if g[a][c] == gab and g[b][c] == gab:
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))

    def can_add(s, x):
        for (a, b) in conflicts[x]:
            if a in s and b in s:
                return False
        return True

    def greedy(order):
        s = set()
        for x in order:
            if can_add(s, x):
                s.add(x)
        return s

    def is_valid(s):
        for x in s:
            for (a, b) in conflicts[x]:
                if a in s and b in s:
                    return False
        return True

    best_set = set()
    best_size = 0
    rng = random.Random(42)

    # Multiple greedy starts with different orderings
    orderings = [
        list(range(1, n+1)),
        list(range(n, 0, -1)),
        sorted(range(1, n+1), key=lambda x: len(conflicts[x])),
        sorted(range(1, n+1), key=lambda x: -len(conflicts[x])),
    ]

    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Random orderings
    for _ in range(10):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Local search / hill climbing
    current = set(best_set)
    cur_size = len(current)

    for it in range(100000):
        if time.time() - start_time > DEADLINE:
            break

        improved = False
        excluded = [x for x in range(1, n+1) if x not in current]
        rng.shuffle(excluded)

        # Try additions
        for x in excluded:
            if can_add(current, x):
                current.add(x)
                cur_size += 1
                improved = True
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = set(current)

        if not improved:
            # Try 1-for-1 swaps
            swap_improved = False
            for x_in in excluded:
                if time.time() - start_time > DEADLINE:
                    break
                blockers = set()
                for (a, b) in conflicts[x_in]:
                    if a in current and b in current:
                        blockers.add(a)
                        blockers.add(b)

                for x_out in blockers:
                    if x_out == x_in:
                        continue
                    # Verify removing x_out clears all conflicts
                    ok = True
                    for (a, b) in conflicts[x_in]:
                        if a in current and b in current and a != x_out and b != x_out:
                            ok = False
                            break
                    if ok:
                        current.discard(x_out)
                        current.add(x_in)
                        improved = True
                        swap_improved = True
                        break
                if swap_improved:
                    break

        if not improved:
            break

    # Final validation
    if not is_valid(best_set):
        best_set = greedy(list(range(1, n+1)))

    return list(best_set)
