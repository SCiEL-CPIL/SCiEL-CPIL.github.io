import time
from itertools import combinations, permutations

DEADLINE = 132.0  # safety margin before 150s wall-clock kill
_START = None

def _now():
    return time.time()

def _remaining():
    return DEADLINE - (_now() - _START)


def _has_c6_cayley(D, n):
    """True iff the Cayley graph on Z_n with undirected difference set D contains a C6.
    D is a set of canonical differences (d in {1..n//2}); undirected means +-d are edges.
    """
    if len(D) < 3:
        return False
    D = set(D)
    # precompute forward moves for each vertex: (v+d)%n and (v-d)%n for d in D
    moves = {}
    for v in range(n):
        lst = []
        for d in D:
            a = (v + d) % n
            b = (v - d) % n
            lst.append(a)
            if b != a:
                lst.append(b)
        moves[v] = lst
    seen = [[False] * n for _ in range(n)]
    for v1 in D:  # v0 = 0
        m1 = moves[0]
        for v2 in m1:
            if v2 == v1:
                continue
            if seen[v1][v2]:
                continue
            for v3 in moves[v2]:
                if v3 == 0 or v3 == v1 or v3 == v2:
                    continue
                if seen[v1][v3]:
                    continue
                for v4 in moves[v3]:
                    if v4 in (0, v1, v2, v3):
                        continue
                    if seen[v1][v4]:
                        continue
                    for v5 in moves[v4]:
                        if v5 in (0, v1, v2, v3, v4):
                            continue
                        if seen[v1][v5]:
                            continue
                        # closing edge v5 - v0=0 must be in D
                        dd = v5 if v5 <= n // 2 else n - v5
                        if dd in D:
                            return True
                        seen[v1][v5] = True
    return False


def _has_c6_adj(adj, n):
    """True iff simple graph on vertices 0..n-1 (adjacency matrix) contains a C6."""
    nbrs = [None] * n
    for i in range(n):
        row = adj[i]
        nbrs[i] = [j for j in range(n) if j != i and row[j]]
    for v1 in range(n):
        nbrs1 = nbrs[v1]
        if len(nbrs1) < 2:
            continue
        mark = [-1] * n
        for v2 in nbrs1:
            mark[v2] = v1
            for v3 in nbrs[v2]:
                if v3 == v1 or v3 == v2:
                    continue
                if mark[v3] == v1:
                    continue
                mark[v3] = v1
                for v4 in nbrs[v3]:
                    if v4 in (v1, v2, v3):
                        continue
                    if mark[v4] == v1:
                        continue
                    mark[v4] = v1
                    for v5 in nbrs[v4]:
                        if v5 in (v1, v2, v3, v4):
                            continue
                        if mark[v5] == v1:
                            continue
                        if v5 in nbrs1:  # closing edge v5 - v1
                            return True
                        mark[v5] = v1
    return False


def _validate_cyclic(assignment, n):
    """assignment: list length n-1, assignment[k-1] = color of difference k (1..n-1).
    Returns (ok, colors_list) for the cyclic coloring."""
    # verify each color class has no C6
    for c in range(4):
        D = {k for k in range(1, n) if assignment[k - 1] == c}
        # canonicalize to undirected difference set
        canon = set()
        for d in D:
            canon.add(d if d <= n // 2 else n - d)
        if _has_c6_cayley(canon, n):
            return False, None
    # build colors list in lexicographic order
    color_map = {}
    for k in range(1, n):
        color_map[k] = assignment[k - 1]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(color_map[(j - i) % n])
    return True, out


def _validate_generic(colors_list, n):
    """Validate a full coloring list for K_n: no color class contains a C6."""
    for c in range(4):
        adj = [[False] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors_list[idx] == c:
                    adj[i][j] = adj[j][i] = True
                idx += 1
        if _has_c6_adj(adj, n):
            return False
    return True


def _try_cyclic(n, budget):
    """Search for a cyclic 4-coloring of Z_n with no monochromatic C6."""
    all_diffs = list(range(1, n))
    m = len(all_diffs)
    assignment = [None] * m
    colors = [set() for _ in range(4)]
    # order differences so we fill; use a heuristic order: put "small" differences first
    order = list(range(m))

    # Precompute for pruning: when a color class reaches a size, we can test C6 early.
    def backtrack(pos):
        if _remaining() < 0.5:
            return False
        if pos == m:
            for c in range(4):
                if _has_c6_cayley(colors[c], n):
                    return False
            return True
        k = all_diffs[order[pos]]
        for c in range(4):
            if len(colors[c]) >= (m + 3) // 4:  # balanced upper bound
                continue
            colors[c].add(k)
            # early prune: if this color class already has a C6, stop
            if _has_c6_cayley(colors[c], n):
                colors[c].remove(k)
                continue
            assignment[k - 1] = c
            if backtrack(pos + 1):
                return True
            assignment[k - 1] = None
            colors[c].remove(k)
        return False

    if backtrack(0):
        ok, out = _validate_cyclic(assignment, n)
        if ok:
            return out
    return None


def _try_cyclic_n17():
    """Dedicated, fast search for the known n=17 cyclic coloring."""
    all_diffs = list(range(1, 17))
    m = 16
    assignment = [None] * m
    colors = [set() for _ in range(4)]

    def backtrack(pos):
        if _remaining() < 0.5:
            return False
        if pos == m:
            for c in range(4):
                if _has_c6_cayley(colors[c], 17):
                    return False
            return True
        k = all_diffs[pos]
        for c in range(4):
            if len(colors[c]) >= 4:
                continue
            colors[c].add(k)
            if _has_c6_cayley(colors[c], 17):
                colors[c].remove(k)
                continue
            assignment[k - 1] = c
            if backtrack(pos + 1):
                return True
            assignment[k - 1] = None
            colors[c].remove(k)
        return False

    if backtrack(0):
        ok, out = _validate_cyclic(assignment, 17)
        if ok:
            return out
    return None


def _try_n18():
    """Attempt a cyclic coloring of Z_18 (4 colors, 4.5 avg -> sizes 4,4,5,5)."""
    all_diffs = list(range(1, 18))
    m = 17
    assignment = [None] * m
    colors = [set() for _ in range(4)]
    cap = [5, 5, 4, 4]

    def backtrack(pos):
        if _remaining() < 0.5:
            return False
        if pos == m:
            for c in range(4):
                if _has_c6_cayley(colors[c], 18):
                    return False
            return True
        k = all_diffs[pos]
        for c in range(4):
            if len(colors[c]) >= cap[c]:
                continue
            colors[c].add(k)
            if _has_c6_cayley(colors[c], 18):
                colors[c].remove(k)
                continue
            assignment[k - 1] = c
            if backtrack(pos + 1):
                return True
            assignment[k - 1] = None
            colors[c].remove(k)
        return False

    if backtrack(0):
        ok, out = _validate_cyclic(assignment, 18)
        if ok:
            return out
    return None


def _fallback_n9():
    n = 9
    # simple construction: color by min distance; verify it's C6-free (it is small).
    # Use a known-safe pattern: split by distance into 4 buckets.
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            out.append(d % 4)
    # verify
    if _validate_generic(out, n):
        return {'n': n, 'colors': out}
    # ultimate trivial fallback: all edges color 0 is NOT valid (K9 has C6).
    # Provide a guaranteed valid small coloring: n=5 has no C6 at all.
    n = 5
    out = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': out}


def construct():
    global _START
    _START = time.time()

    # First, secure a valid n=17 (the known best) quickly.
    best = None

    # Dedicated fast n=17 cyclic search (this is the known-good family).
    res17 = _try_cyclic_n17()
    if res17 is not None:
        best = {'n': 17, 'colors': res17}

    # Try to beat it: n=18 cyclic search (time-boxed).
    if _remaining() > 8:
        res18 = _try_n18()
        if res18 is not None:
            best = {'n': 18, 'colors': res18}

    # If we still have a lot of time, try a generic n=18 search via randomized
    # local improvement (best-effort, time-boxed).
    if _remaining() > 20:
        res18b = _try_n18_random()
        if res18b is not None:
            best = {'n': 18, 'colors': res18b}

    if best is not None:
        return best

    # Fallback: try a generic n=17 search (non-cyclic) in case cyclic failed.
    if _remaining() > 10:
        res17g = _try_generic_n17()
        if res17g is not None:
            return {'n': 17, 'colors': res17g}

    return _fallback_n9()


def _try_n18_random():
    """Randomized hill-climb for n=18: start from a cyclic seed, fix C6 violations."""
    import random
    n = 18
    m = n * (n - 1) // 2
    rng = random.Random(12345)

    # seed: random 4-coloring
    colors = [rng.randrange(4) for _ in range(m)]

    def edge_index(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def count_c6():
        total = 0
        for c in range(4):
            adj = [[False] * n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i][j] = adj[j][i] = True
                    idx += 1
            if _has_c6_adj(adj, n):
                total += 1
        return total

    cur = count_c6()
    if cur == 0:
        return colors[:]

    it = 0
    while _remaining() > 5 and cur > 0:
        it += 1
        e = rng.randrange(m)
        old = colors[e]
        new = (old + 1 + rng.randrange(3)) % 4
        colors[e] = new
        nxt = count_c6()
        if nxt < cur:
            cur = nxt
        else:
            colors[e] = old
    if cur == 0:
        return colors[:]
    return None


def _try_generic_n17():
    """Generic (non-cyclic) randomized search for n=17, time-boxed."""
    import random
    n = 17
    m = n * (n - 1) // 2
    rng = random.Random(777)
    colors = [rng.randrange(4) for _ in range(m)]

    def count_c6():
        total = 0
        for c in range(4):
            adj = [[False] * n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i][j] = adj[j][i] = True
                    idx += 1
            if _has_c6_adj(adj, n):
                total += 1
        return total

    cur = count_c6()
    while _remaining() > 5 and cur > 0:
        e = rng.randrange(m)
        old = colors[e]
        new = (old + 1 + rng.randrange(3)) % 4
        colors[e] = new
        nxt = count_c6()
        if nxt < cur:
            cur = nxt
        else:
            colors[e] = old
    if cur == 0:
        return colors[:]
    return None
