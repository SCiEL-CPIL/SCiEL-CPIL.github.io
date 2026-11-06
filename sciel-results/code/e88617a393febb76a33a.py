import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin before the 90s kill

    N = 192

    # ---- Precompute the pairwise conflict structure ONCE ----
    # conflict[a][x] = set of y such that (x + y) % a == 0
    # Then for a set S, 'x conflicts with a'  <=>  (S \ {a}) intersects conflict[a][x]
    conflict = [None] * (N + 1)
    for a in range(1, N + 1):
        c = [set() for _ in range(N + 1)]
        for x in range(1, N + 1):
            # y in 1..N with (x + y) % a == 0
            rem = (-x) % a
            if rem == 0:
                base = a
            else:
                base = rem
            if base <= N:
                for y in range(base, N + 1, a):
                    c[x].add(y)
        conflict[a] = c

    # ---- Fast membership-aware validity test ----
    def can_add(S, Sset, x):
        n = len(S)
        # (1) no triple of S sums to a multiple of x
        if n >= 3:
            for i in range(n):
                a = S[i]
                for j in range(i + 1, n):
                    b = S[j]
                    base = a + b
                    for k in range(j + 1, n):
                        if (base + S[k]) % x == 0:
                            return False
        # (2) for each a in S, x must not complete a forbidden pair
        for a in S:
            if conflict[a][x] & Sset:
                return False
        return True

    # ---- Greedy over a given ordering ----
    def greedy(order, hard_deadline):
        S = []
        Sset = set()
        for x in order:
            if len(S) >= 48:
                break
            if time.time() > hard_deadline:
                break
            if can_add(S, Sset, x):
                S.append(x)
                Sset.add(x)
        return S

    best = []

    # Phase 1: deterministic descending
    S = greedy(list(range(N, 0, -1)), deadline - 20.0)
    best = S
    blen = len(S)

    # Phase 2: many randomized orderings (large numbers loosely prioritized)
    for trial in range(400):
        if blen >= 48:
            break
        if time.time() > deadline - 25.0:
            break
        random.seed(100000 + trial)
        order = list(range(N, 0, -1))
        # perturbed priority: keep big numbers near the front but shuffle the rest
        for i in range(N):
            j = random.randrange(i, N)
            order[i], order[j] = order[j], order[i]
        S = greedy(order, deadline - 25.0)
        if len(S) > blen:
            best = S
            blen = len(S)

    # Phase 3: deep 1-out-2-in local search
    if blen < 48 and time.time() < deadline - 10.0:
        cur = best[:]
        curset = set(cur)
        clen = blen
        improved = True
        while improved and clen < 48 and time.time() < deadline - 10.0:
            improved = False
            for i in range(len(cur)):
                if time.time() > deadline - 10.0:
                    break
                a = cur[i]
                without_a = [v for j, v in enumerate(cur) if j != i]
                wa_set = curset - {a}
                for b in range(N, 0, -1):
                    if b in wa_set:
                        continue
                    if not can_add(without_a, wa_set, b):
                        continue
                    with_b = without_a + [b]
                    wb_set = wa_set | {b}
                    for c in range(N, 0, -1):
                        if c in wb_set:
                            continue
                        if can_add(with_b, wb_set, c):
                            cur = with_b + [c]
                            curset = wb_set | {c}
                            clen = len(cur)
                            if clen > blen:
                                best = cur[:]
                                blen = clen
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Phase 4: final polish — add any leftover element
    if blen < 48 and time.time() < deadline - 3.0:
        cur = best[:]
        curset = set(cur)
        for x in range(N, 0, -1):
            if time.time() > deadline - 3.0:
                break
            if len(cur) >= 48:
                break
            if x in curset:
                continue
            if can_add(cur, curset, x):
                cur.append(x)
                curset.add(x)
        if len(cur) > blen:
            best = cur
            blen = len(cur)

    return best[:48] if len(best) >= 48 else best

# EVOLVE-BLOCK-END
