"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

def construct():
    _t0 = _time.monotonic()
    _DEADLINE = _t0 + 120.0

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def build_pairs(arr):
        n = len(arr)
        pairs = [[] for _ in range(n)]
        seen = set()
        for i in range(n):
            ai = arr[i]
            for j in range(i, n):
                s = ai + arr[j]
                if s in seen:
                    continue
                k = s + 10000
                if 0 <= k < n:
                    pairs[i].append((j, k))
        for i in range(n):
            pairs[i].sort(key=lambda t: t[1])
        return pairs

    def greedy(arr, pairs, rng):
        n = len(arr)
        in_set = bytearray(n)
        rem = [0] * n
        for i in range(n):
            rem[i] = len(pairs[i])
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if rem[i] == 0:
                in_set[i] = 1
        for i in range(n):
            if in_set[i]:
                continue
            for (j, k) in pairs[i]:
                if in_set[j]:
                    rem[i] -= 1
                    if rem[i] == 0:
                        in_set[i] = 1
                        break
        return in_set

    def local_search(arr, pairs, in_set, rng, deadline):
        n = len(arr)
        in_set = bytearray(in_set)
        rem = [0] * n
        for i in range(n):
            if in_set[i]:
                continue
            c = 0
            for (j, k) in pairs[i]:
                if in_set[j] or in_set[k]:
                    c += 1
            rem[i] = c
        while _time.monotonic() < deadline:
            improved = False
            for i in range(n):
                if in_set[i] or rem[i] != 0:
                    continue
                in_set[i] = 1
                for (j, k) in pairs[i]:
                    if in_set[j]:
                        rem[j] += 1
                    if in_set[k]:
                        rem[k] += 1
                improved = True
            if not improved:
                break
        # randomized add/remove hill-climbing
        for _ in range(200000):
            if _time.monotonic() >= deadline:
                break
            # try to add a random non-member
            if rng.random() < 0.5:
                i = rng.randrange(n)
                if not in_set[i] and rem[i] == 0:
                    in_set[i] = 1
                    for (j, k) in pairs[i]:
                        if in_set[j]:
                            rem[j] += 1
                        if in_set[k]:
                            rem[k] += 1
                    continue
            # try removing a random member that frees a non-member
            i = rng.randrange(n)
            if not in_set[i]:
                continue
            freed = -1
            for (j, k) in pairs[i]:
                if in_set[j] or in_set[k]:
                    freed = j if in_set[j] else k
                    break
            if freed >= 0:
                in_set[i] = 0
                for (j, k) in pairs[i]:
                    if in_set[j]:
                        rem[j] -= 1
                    if in_set[k]:
                        rem[k] -= 1
                if rem[freed] == 0:
                    in_set[freed] = 1
                    for (j, k) in pairs[freed]:
                        if in_set[j]:
                            rem[j] += 1
                        if in_set[k]:
                            rem[k] += 1
        # final deterministic sweep
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not in_set[i] and rem[i] == 0:
                    in_set[i] = 1
                    for (j, k) in pairs[i]:
                        if in_set[j]:
                            rem[j] += 1
                        if in_set[k]:
                            rem[k] += 1
                    changed = True
        return in_set

    def solve(arr, seed, deadline):
        n = len(arr)
        pairs = build_pairs(arr)
        best = bytearray(n)
        best_size = 0
        for r in range(200):
            if _time.monotonic() >= deadline:
                break
            rng = Random(seed * 1000003 + 7 + r)
            g = greedy(arr, pairs, rng)
            s = local_search(arr, pairs, g, rng, deadline)
            sz = s.count(1)
            if sz > best_size:
                best_size = sz
                best = s
            if best_size == n:
                break
        return [arr[i] for i in range(n) if best[i]]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        per = (_DEADLINE - _time.monotonic()) / (3 - len(results))
        dl = _time.monotonic() + max(0.5, per)
        results.append(solve(arr, seed, dl))
    return results
# EVOLVE-BLOCK-END
