import time

def construct():
    t0 = time.time()
    BUDGET = 55.0
    n = 128

    as_last = [[] for _ in range(n)]
    as_third = [[] for _ in range(n)]
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
            as_last[p3].append((p0, p1, p2))
            as_third[p2].append((p0, p1, p3))

    def greedy(k):
        colors = [-1] * n
        for p in range(n):
            placed = False
            for c in range(k):
                ok = True
                for (a, b, cc) in as_last[p]:
                    ca, cb, cd = colors[a], colors[b], colors[cc]
                    # distinct count
                    d1 = 1
                    if cb != ca: d1 += 1
                    if c != ca and c != cb: d1 += 1
                    if cd != ca and cd != cb and cd != c: d1 += 1
                    if d1 < 3:
                        ok = False; break
                if not ok: continue
                for (a, b, dd) in as_third[p]:
                    ca, cb = colors[a], colors[b]
                    if ca == cb == c:
                        ok = False; break
                if ok:
                    colors[p] = c; placed = True; break
            if not placed:
                return None
        return colors

    def dfs(k, deadline):
        colors = [-1] * n
        nodes = 0
        def rec(p):
            nonlocal nodes
            if time.time() > deadline:
                return False
            if p == n:
                return True
            for c in range(k):
                nodes += 1
                ok = True
                for (a, b, cc) in as_last[p]:
                    ca, cb, cd = colors[a], colors[b], colors[cc]
                    d1 = 1
                    if cb != ca: d1 += 1
                    if c != ca and c != cb: d1 += 1
                    if cd != ca and cd != cb and cd != c: d1 += 1
                    if d1 < 3:
                        ok = False; break
                if ok:
                    for (a, b, dd) in as_third[p]:
                        ca, cb = colors[a], colors[b]
                        if ca == cb == c:
                            ok = False; break
                if ok:
                    colors[p] = c
                    if rec(p + 1):
                        return True
                    colors[p] = -1
            return False
        return rec(0), nodes
    ...
