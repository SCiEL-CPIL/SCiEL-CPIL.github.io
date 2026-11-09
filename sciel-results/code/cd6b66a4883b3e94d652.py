def construct():
    import time
    t0 = time.time()
    DEADLINE = 140.0
    REQUIRED = frozenset(range(1, 31))

    def time_left():
        return DEADLINE - time.time()

    def try_search(D, max_n=32, node_budget=4_000_000):
        used = set()
        covered = set()
        cur = [0]
        best = []
        nodes = [0]

        def dfs():
            if best:
                return True
            if time_left() < 2.0:
                return False
            nodes[0] += 1
            if nodes[0] > node_budget:
                return False
            if covered == REQUIRED:
                best.extend(cur)
                return True
            if len(cur) >= max_n:
                return False
            # pruning: can we cover remaining?
            rem = REQUIRED - covered
            if not rem:
                best.extend(cur)
                return True
            # max additional differences if we fill to max_n
            k = max_n - len(cur)
            add = 0
            c = len(cur)
            for i in range(k):
                add += c + i
            avail = D - len(used)
            if len(rem) > min(add, avail):
                return False
            last = cur[-1]
            # build candidate list with coverage score
            cands = []
            for x in range(last + 1, D + 1):
                newd = []
                ok = True
                for a in cur:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                    newd.append(d)
                if not ok:
                    continue
                score = sum(1 for d in newd if d in rem)
                cands.append((score, x, newd))
            cands.sort(key=lambda t: (-t[0], t[1]))
            for score, x, newd in cands:
                used.update(newd)
                covered.update(newd)
                cur.append(x)
                if dfs():
                    return True
                cur.pop()
                covered.difference_update(newd)
                used.difference_update(newd)
            return False

        dfs()
        return list(best) if best else None

    for D in range(36, 400):
        if time_left() < 3.0:
            break
        res = try_search(D)
        if res:
            return res

    # fallback greedy (valid but possibly large)
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        placed = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
