"""Rewritten: fast 3-color search via DSU-with-rollback + annealing, robust 4-color fallback."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
NEIGH = defaultdict(list)
for _u, _v in EDGES:
    NEIGH[_u].append(_v)
    NEIGH[_v].append(_u)
NEIGH = {i: NEIGH[i] for i in range(N)}
ORDER = sorted(range(N), key=lambda x: len(NEIGH[x]), reverse=True)


def _full_count(colors, k):
    """Exact number of bicolored-cycle (i<j) pairs. O(k^2 * E)."""
    cnt = 0
    for i in range(k):
        for j in range(i + 1, k):
            parent = list(range(N))
            rank = [0] * N
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            cyc = False
            for u, v in EDGES:
                cu, cv = colors[u], colors[v]
                if (cu == i and cv == j) or (cu == j and cv == i):
                    ru, rv = find(u), find(v)
                    if ru == rv:
                        cyc = True
                        break
                    if rank[ru] < rank[rv]:
                        ru, rv = rv, ru
                    parent[rv] = ru
                    if rank[ru] == rank[rv]:
                        rank[ru] += 1
            if cyc:
                cnt += 1
    return cnt


def _anneal(colors, k, deadline, T0=0.7, T1=0.03, steps=140000):
    """Simulated annealing on the bicolored-cycle count. Returns best (colors,count)."""
    c = list(colors)
    cnt = _full_count(c, k)
    best = c[:]
    best_cnt = cnt
    for s in range(steps):
        if time.time() > deadline:
            break
        if best_cnt == 0:
            return best, 0
        v = random.randrange(N)
        oc = c[v]
        forb = set(c[u] for u in NEIGH[v])
        opts = [x for x in range(k) if x not in forb and x != oc]
        if not opts:
            continue
        nc = random.choice(opts)
        c[v] = nc
        ncnt = _full_count(c, k)
        T = T0 * (T1 / T0) ** (s / max(1, steps - 1))
        d = ncnt - cnt
        if d <= 0 or random.random() < math.exp(-d / max(1e-6, T)):
            cnt = ncnt
            if cnt < best_cnt:
                best_cnt = cnt
                best = c[:]
        else:
            c[v] = oc
    return best, best_cnt


def _greedy4():
    c = [0] * N
    for v in ORDER:
        forb = set(c[u] for u in NEIGH[v])
        for x in range(4):
            if x not in forb:
                c[v] = x
                break
        else:
            c[v] = 0
    return c


def _anneal4(deadline):
    best = None
    best_cnt = 10**9
    for trial in range(6):
        if time.time() > deadline:
            break
        c = _greedy4()
        for _ in range(40):  # randomize a few vertices, keep proper
            v = random.randrange(N)
            forb = set(c[u] for u in NEIGH[v])
            opts = [x for x in range(4) if x not in forb]
            if opts:
                c[v] = random.choice(opts)
        c, cnt = _anneal(c, 4, deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best = c
        if best_cnt == 0:
            break
    return best, best_cnt


def _search3(deadline):
    """DSU-with-rollback backtracking + annealing to find a valid 3-coloring."""
    parent = list(range(N))
    size = [1] * N
    stack = []
    comp = {i: i for i in range(N)}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]
        stack.append((ru, rv))
        return True

    def roll(n):
        for _ in range(n):
            ru, rv = stack.pop()
            parent[rv] = rv
            size[ru] -= size[rv]

    def assign(v, c):
        cnt = 0
        for u in NEIGH[v]:
            cu = comp[u]
            if cu == c:
                return -1
            if cu != -1:
                if not union(u, v):
                    return -1
                cnt += 1
        comp[v] = c
        return cnt

    def unassign(v, c):
        roll(c)
        comp[v] = -1

    comp = [-1] * N
    sol = None

    def bt(idx):
        nonlocal sol
        if sol is not None:
            return
        if time.time() > deadline:
            return
        if idx == N:
            sol = [comp[v] for v in range(N)]
            return
        v = ORDER[idx]
        forb = set(comp[u] for u in NEIGH[v] if comp[u] != -1)
        opts = [x for x in range(3) if x not in forb]
        random.shuffle(opts)
        for c in opts:
            r = assign(v, c)
            if r >= 0:
                bt(idx + 1)
                unassign(v, c)
                if sol is not None:
                    return

    bt(0)
    if sol is not None and _full_count(sol, 3) == 0:
        return sol

    # Anneal to find a valid 3-coloring
    best = None
    best_cnt = 10**9
    for trial in range(30):
        if time.time() > deadline:
            break
        c = list(range(3)) + [random.randrange(3) for _ in range(N - 3)]
        for _ in range(60):
            v = random.randrange(N)
            forb = set(c[u] for u in NEIGH[v])
            opts = [x for x in range(3) if x not in forb]
            if opts:
                c[v] = random.choice(opts)
        c, cnt = _anneal(c, 3, deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best = c
        if best_cnt == 0:
            break
    if best is not None and best_cnt == 0:
        return best
    return None


def construct():
    start = time.time()
    deadline = start + 85.0
    random.seed(12345)

    # 1) Try to find a valid 3-coloring (score 16.67).
    t3 = min(deadline - 0.5, start + 32.0)
    sol3 = _search3(t3)
    if sol3 is not None and _full_count(sol3, 3) == 0:
        return [int(c) for c in sol3]

    # 2) Robust 4-color fallback (score 12.5, always valid).
    t4 = deadline - 0.5
    c4, cnt4 = _anneal4(t4)
    if c4 is not None:
        # ensure proper + minimal cycle count
        for v in range(N):
            forb = set(c4[u] for u in NEIGH[v])
            if c4[v] in forb:
                for x in range(4):
                    if x not in forb:
                        c4[v] = x
                        break
        if _full_count(c4, 4) == 0:
            return [int(c) for c in c4]

    # 3) Absolute fallback: greedy 4-coloring (proper, may have cycles but valid proper).
    c = _greedy4()
    for v in range(N):
        forb = set(c[u] for u in NEIGH[v])
        if c[v] in forb:
            for x in range(4):
                if x not in forb:
                    c[v] = x
                    break
    return [int(c) for c in c]

# EVOLVE-BLOCK-END
