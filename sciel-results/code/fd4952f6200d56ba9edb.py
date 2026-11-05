import time

_START = None

def _seed_greedy():
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

def _diffs_of(marks):
    return [marks[j] - marks[i] for i in range(len(marks)) for j in range(i + 1, len(marks))]

def _search(D, node_budget):
    full = tuple(range(1, 31))
    fullset = set(full)
    visited = set()
    nodes = [0]
    best = [None]
    start = time.time()

    def dfs(marks):
        if best[0] is not None:
            return
        nodes[0] += 1
        if nodes[0] > node_budget:
            return
        if time.time() - start > 95:
            return
        used = set(_diffs_of(marks))
        missing = [d for d in full if d not in used]
        if not missing:
            best[0] = list(marks)
            return
        m = missing[0]
        for v in marks:  # marks sorted, ascending -> x ascending
            x = v + m
            if x > D:
                break
            if x in marks:
                continue
            ok = True
            for w in marks:
                diff = x - w if x > w else w - x
                if diff in used or diff > D:
                    ok = False
                    break
            if not ok:
                continue
            nm = list(marks)
            nm.append(x)
            nm.sort()
            key = tuple(nm)
            if key in visited:
                continue
            visited.add(key)
            dfs(nm)
            if best[0] is not None:
                return

    dfs((0,))
    return best[0]

def construct():
    global _START
    if _START is None:
        _START = time.time()
    for D in range(30, 160):
        if time.time() - _START > 100:
            break
        res = _search(D, 200000)
        if res is not None:
            return res
    # fallback
    return _seed_greedy()
