"""Bitmask local search with incremental 3-AP deltas, greedy fill, 2-swap, and SA."""
import time
import random
import math

N = 256
DEADLINE = 78.0  # safety margin under 90s kill


def construct():
    start_time = time.time()
    random.seed(12345)

    # ---------- Precompute APs (1-indexed) ----------
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))

    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each element, the other pairs/triples it completes
    # 3-AP: adding x forms a 3-AP iff the other two are present.
    elem_3pairs = [[] for _ in range(N + 1)]
    for a, b, c in three_aps:
        elem_3pairs[a].append((b, c))
        elem_3pairs[b].append((a, c))
        elem_3pairs[c].append((a, b))

    # 4-AP: adding x forms a 4-AP iff the other three are present.
    elem_4triples = [[] for _ in range(N + 1)]
    for a, b, c, d in four_aps:
        elem_4triples[a].append((b, c, d))
        elem_4triples[b].append((a, c, d))
        elem_4triples[c].append((a, b, d))
        elem_4triples[d].append((a, b, c))

    def present(S, x):
        return (S >> (x - 1)) & 1

    def count_3aps(S):
        c = 0
        for a, b, cc in three_aps:
            if present(S, a) and present(S, b) and present(S, cc):
                c += 1
        return c

    def safe_add(S, x):
        for u, v, w in elem_4triples[x]:
            if present(S, u) and present(S, v) and present(S, w):
                return False
        return True

    def gain(S, x):
        g = 0
        for u, v in elem_3pairs[x]:
            if present(S, u) and present(S, v):
                g += 1
        return g

    def loss(S, x):
        # x currently present
        l = 0
        for u, v in elem_3pairs[x]:
            if present(S, u) and present(S, v):
                l += 1
        return l

    def members(S):
        return [i + 1 for i in range(N) if (S >> i) & 1]

    # ---------- Initial: weighted greedy ----------
    def initial_greedy():
        S = 0
        # weight = number of 3-APs element participates in
        w = [0] * (N + 1)
        for x in range(1, N + 1):
            w[x] = len(elem_3pairs[x])
        order = list(range(1, N + 1))
        random.shuffle(order)
        # sort by weight desc, tie random
        order.sort(key=lambda x: -w[x])
        for x in order:
            if safe_add(S, x):
                S |= (1 << (x - 1))
        return S

    # ---------- Local search ----------
    def local_search(S, deadline):
        S = S
        score = count_3aps(S)
        best_S = S
        best_score = score
        T = 6.0
        alpha = 0.997
        min_T = 0.5

        while time.time() < deadline:
            mem = members(S)
            if not mem:
                break
            move = random.random()
            new_S = S
            new_score = score

            if move < 0.5:
                # remove one element, then greedily fill
                x = random.choice(mem)
                new_S = S & ~(1 << (x - 1))
                new_score = score - loss(S, x)
                # greedy fill: try all missing, add safe ones by gain
                cand = [i for i in range(1, N + 1) if not present(new_S, i)]
                cand.sort(key=lambda y: -gain(new_S, y))
                for y in cand:
                    if safe_add(new_S, y):
                        new_S |= (1 << (y - 1))
                        new_score += gain(new_S, y)
            elif move < 0.8:
                # 2-swap: remove x, add y
                x = random.choice(mem)
                y = random.choice([i for i in range(1, N + 1) if not present(S, i)])
                new_S = S & ~(1 << (x - 1))
                new_score = score - loss(S, x)
                if safe_add(new_S, y):
                    new_S |= (1 << (y - 1))
                    new_score += gain(new_S, y)
                else:
                    continue
            else:
                # single add (if safe)
                y = random.choice([i for i in range(1, N + 1) if not present(S, i)])
                if safe_add(S, y):
                    new_S = S | (1 << (y - 1))
                    new_score = score + gain(S, y)
                else:
                    continue

            delta = new_score - score
            if new_score > best_score:
                best_S = new_S
                best_score = new_score
            if delta >= 0:
                S = new_S
                score = new_score
            else:
                if random.random() < math.exp(delta / T):
                    S = new_S
                    score = new_score
            T = max(min_T, T * alpha)

        return best_S, best_score

    # ---------- Run restarts ----------
    best_S = 0
    best_score = -1

    # first: greedy start + long search
    S0 = initial_greedy()
    s0 = count_3aps(S0)
    if s0 > best_score:
        best_S, best_score = S0, s0

    # Phase 1: a few medium searches
    n1 = 6
    for i in range(n1):
        if time.time() > DEADLINE - 12:
            break
        S = initial_greedy()
        slice_end = start_time + (i + 1) * 8.0
        if slice_end > DEADLINE - 12:
            slice_end = DEADLINE - 12
        S2, sc = local_search(S, slice_end)
        if sc > best_score:
            best_S, best_score = S2, sc

    # Phase 2: long search from best
    if time.time() < DEADLINE - 2:
        S2, sc = local_search(best_S, DEADLINE - 2)
        if sc > best_score:
            best_S, best_score = S2, sc

    result = members(best_S)

    # ---------- Safety: verify no 4-AP ----------
    valid = True
    for a, b, c, d in four_aps:
        if present(best_S, a) and present(best_S, b) and present(best_S, c) and present(best_S, d):
            valid = False
            break
    if not valid or not result:
        S = 0
        for x in range(1, N + 1):
            if safe_add(S, x):
                S |= (1 << (x - 1))
        result = members(S)

    return sorted(result)
