"""Solver for few-color graph coloring with no bicolored cycles.

Structural approach: the graph is the circulant Cayley graph C(50; 1, 7) on Z_50.
We exploit this by:
  1. Exhaustively checking all affine colorings f(v) = (a*v + b) mod k for small k,
     and a few other regular patterns. For Cayley graphs these are the most likely
     to be valid because the bicolored subgraphs inherit the group structure.
  2. If that fails, run a fast local-search / repair procedure that starts from a
     valid k-coloring (or a near-valid one) and tries to merge colors.
  3. Fall back to a guaranteed-valid high-coloring.
"""
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_valid(coloring):
    """Check adjacent differ and no bicolored cycles. Returns True if valid."""
    n = len(coloring)
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Collect color pairs present
    color_pairs = set()
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        color_pairs.add((min(cu, cv), max(cu, cv)))

    for c1, c2 in color_pairs:
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
    return True


def _num_colors(coloring):
    return len(set(coloring))


def _generate_candidates(n):
    """Yield candidate colorings in order of increasing expected color count."""
    # 1. Affine colorings f(v) = (a*v + b) mod k for small k.
    #    For a Cayley graph these are the natural low-color candidates.
    for k in range(2, 15):
        for a in range(1, k):  # a=0 gives constant coloring (invalid for k>1)
            for b in range(k):
                yield [(a * v + b) % k for v in range(n)]

    # 2. v mod k
    for k in range(2, 15):
        yield [v % k for v in range(n)]

    # 3. (v // 10 + v % 10) % k  (block-ish)
    for k in range(2, 10):
        yield [(v // 10 + v % 10) % k for v in range(n)]

    # 4. (v // 5) % k
    for k in range(2, 10):
        yield [(v // 5) % k for v in range(n)]

    # 5. (v // 10) * 2 + (v % 2)
    yield [(v // 10) * 2 + (v % 2) for v in range(n)]

    # 6. (v % 4) + 4 * ((v // 10) % 2)
    yield [(v % 4) + 4 * ((v // 10) % 2) for v in range(n)]

    # 7. (v % 5) + 5 * ((v // 10) % 2)
    yield [(v % 5) + 5 * ((v // 10) % 2) for v in range(n)]


def _try_candidates(n, deadline):
    """Try candidate colorings, return the best valid one found."""
    best = None
    best_count = n + 1
    for coloring in _generate_candidates(n):
        if time.time() > deadline:
            break
        # Quick reject: if any edge has equal colors, skip
        ok = True
        for u, v in P['edges']:
            if coloring[u] == coloring[v]:
                ok = False
                break
        if not ok:
            continue
        if _check_valid(coloring):
            cnt = _num_colors(coloring)
            if cnt < best_count:
                best_count = cnt
                best = [int(x) for x in coloring]
            # If we hit a very low count, we can stop early
            if best_count <= 3:
                break
    return best, best_count


def _local_repair(start_coloring, adj, max_iter, deadline):
    """Try to reduce colors by local moves. Returns improved coloring or None."""
    coloring = [int(x) for x in start_coloring]
    n = len(coloring)
    best = [int(x) for x in coloring]
    best_count = _num_colors(coloring)
    rng = random.Random(12345)

    def try_move():
        nonlocal best, best_count
        # Pick a random vertex and try to change its color to a color not used by neighbors
        v = rng.randrange(n)
        used = set()
        for nb in adj[v]:
            used.add(coloring[nb])
        avail = [c for c in range(best_count) if c not in used]
        if not avail:
            return
        c_new = rng.choice(avail)
        if c_new == coloring[v]:
            return
        old = coloring[v]
        coloring[v] = c_new
        if _check_valid(coloring):
            cnt = _num_colors(coloring)
            if cnt < best_count:
                best_count = cnt
                best = [int(x) for x in coloring]
        else:
            coloring[v] = old

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        try_move()
    return best if best_count < _num_colors(start_coloring) else None


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    adj = _build_adj()
    n = P['n']

    # Phase 1: systematic candidate sweep (affine + regular patterns)
    best, best_count = _try_candidates(n, deadline - 5)
    if best is not None and best_count <= 4:
        return best

    # Phase 2: local repair starting from the best candidate (or a fresh one)
    if best is None:
        best = [v % 5 for v in range(n)]
    improved = _local_repair(best, adj, max_iter=200000, deadline=deadline - 2)
    if improved is not None:
        return improved

    # Phase 3: if we have a valid result, return it; else guaranteed fallback
    if best is not None and _check_valid(best):
        return best

    # Guaranteed valid: all distinct colors
    return list(range(n))


if __name__ == "__main__":
    result = construct()
    print("colors used:", _num_colors(result))
    print("valid:", _check_valid(result))
