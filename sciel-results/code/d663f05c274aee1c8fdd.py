"""Maximum independent set on the conflict graph, bitset branch-and-bound."""
import time

def _build_conflicts(n=256):
    # conflict[i][j] True iff (a+b) divides ab, with a=i+1, b=j+1
    N = n
    conf = [bytearray(N) for _ in range(N)]
    for i in range(N):
        a = i + 1
        row = conf[i]
        for j in range(i + 1, N):
            b = j + 1
            s = a + b
            if (a * b) % s == 0:
                row[j] = 1
                conf[j][i] = 1
    return conf

def _neighbors(conf, i):
    return [j for j in range(len(conf)) if conf[i][j]]

def _greedy_color_bound(cand, conf):
    # Returns (colors, color_class_size_bound) where colors[i] is color index of vertex i
    # and bound is number of colors (an upper bound on MIS of induced subgraph on cand)
    n = len(cand)
    if n == 0:
        return 0, []
    # Order by degree in candidate subgraph, descending
    deg = [0] * n
    pos = {cand[k]: k for k in range(n)}
    for k in range(n):
        vi = cand[k]
        # count neighbors within cand
        c = 0
        confrow = conf[vi]
        for t in range(n):
            if t != k and confrow[cand[t]]:
                c += 1
        deg[k] = c
    order = sorted(range(n), key=lambda k: -deg[k])
    color = [-1] * n
    color_class = []  # list of sets of vertex indices (within cand)
    for k in order:
        vi = cand[k]
        used = set()
        for cc in color_class:
            conflict = False
            for u in cc:
                if conf[vi][cand[u]]:
                    conflict = True
                    break
            if conflict:
                used.add(len(color_class) - 1)
        # assign smallest unused color
        c = 0
        while c in used:
            c += 1
        if c == len(color_class):
            color_class.append([k])
        else:
            color_class[c].append(k)
        color[k] = c
    return len(color_class), color_class

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin under 90s
    N = 256
    conf = _build_conflicts(N)

    # Precompute neighbor lists
    neigh = [set() for _ in range(N)]
    for i in range(N):
        row = conf[i]
        for j in range(N):
            if row[j]:
                neigh[i].add(j)

    best = [0]
    best_set = [set()]

    # Initial greedy solution (greedy by min-degree)
    def greedy_init():
        cand = set(range(N))
        sol = set()
        while cand:
            # pick min degree in cand
            v = min(cand, key=lambda x: len(neigh[x] & cand))
            sol.add(v)
            cand.discard(v)
            cand -= neigh[v]
        return sol

    g0 = greedy_init()
    best[0] = len(g0)
    best_set[0] = set(g0)

    # Bitset representation of candidate set
    # We'll do branch and bound on a candidate set.
    # Use a list of vertices as candidate set.

    def expand(cand, current):
        # cand: list of vertices (indices), current: set of chosen
        if time.time() > deadline:
            return
        if not cand:
            if len(current) > best[0]:
                best[0] = len(current)
                best_set[0] = set(current)
            return
        # Upper bound via greedy coloring
        cb, _ = _greedy_color_bound(cand, conf)
        if len(current) + cb <= best[0]:
            return
        # Pick pivot vertex: max degree in cand
        cand_set = set(cand)
        pivot = max(cand, key=lambda x: len(neigh[x] & cand_set))
        # Branch: include pivot
        current.add(pivot)
        new_cand = [v for v in cand if v != pivot and not conf[pivot][v]]
        expand(new_cand, current)
        current.discard(pivot)
        # Branch: exclude pivot
        new_cand = [v for v in cand if v != pivot]
        expand(new_cand, current)

    # Start with all vertices
    expand(list(range(N)), set())

    if best[0] == 0:
        best_set[0] = g0
        best[0] = len(g0)

    return sorted(v + 1 for v in best_set[0])
