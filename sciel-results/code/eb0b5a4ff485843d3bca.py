"""Combined: decompose into two 25-cycles, solve constrained circular coloring."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Precompute the two independent cycles (even and odd indexed vertices)
EVEN_CYCLE = [2*i for i in range(25)]
ODD_CYCLE = [2*i + 1 for i in range(25)]

# Build adjacency for proper-coloring checks
_ADJ = [[] for _ in range(N)]
for _u, _v in EDGES:
    _ADJ[_u].append(_v)
    _ADJ[_v].append(_u)


def _cycle_edges(cycle):
    """Return edges of a cycle given as a list of vertex ids in order."""
    m = len(cycle)
    return [(cycle[i], cycle[(i + 1) % m]) for i in range(m)]


def _has_bicolored_cycle(cycle, coloring, c1, c2):
    """Check if subgraph of cycle induced by colors c1,c2 has a cycle (union-find)."""
    parent = list(range(len(cycle)))
    rank = [0] * len(cycle)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return True
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return False

    m = len(cycle)
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        cu, cv = coloring[u], coloring[v]
        if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
            # map vertex to its index in the cycle
            if union(i, (i + 1) % m):
                return True
    return False


def _is_valid_cycle(cycle, coloring, num_colors):
    """Proper + no bicolored cycle within this single cycle."""
    m = len(cycle)
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        if coloring[u] == coloring[v]:
            return False
    for c1 in range(num_colors):
        for c2 in range(c1 + 1, num_colors):
            if _has_bicolored_cycle(cycle, coloring, c1, c2):
                return False
    return True


def _is_valid_global(coloring, num_colors):
    """Full verification on the actual graph."""
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
    for c1 in range(num_colors):
        for c2 in range(c1 + 1, num_colors):
            if _has_bicolored_cycle(EVEN_CYCLE, coloring, c1, c2):
                return False
            if _has_bicolored_cycle(ODD_CYCLE, coloring, c1, c2):
                return False
    return True


def _modular_search():
    """Try modular colorings c_i = (a*i+b) % k; verify globally."""
    for k in range(3, 7):
        for a in range(1, k):
            for b in range(k):
                coloring = [(a * i + b) % k for i in range(N)]
                if _is_valid_global(coloring, k):
                    return [int(c) for c in coloring]
    return None


def _cycle_backtrack(cycle, K, deadline):
    """Backtrack a proper + acyclic coloring of one cycle with K colors.
    Returns a coloring dict for the cycle vertices, or None."""
    m = len(cycle)
    # precompute per-vertex color availability (proper) via adjacency within cycle
    # cycle neighbor indices
    col = {}
    used = [0] * K

    def ok_bicolors(c, v_idx):
        # tentative: assign color c to v_idx; check all bicolored pairs involving c
        # We check pairs (c, other) for all other colors that appear in the cycle
        # Simpler: after assignment, check all pairs among used colors.
        return True

    def check_bicolors():
        for c1 in range(K):
            if used[c1] == 0:
                continue
            for c2 in range(c1 + 1, K):
                if used[c2] == 0:
                    continue
                if _has_bicolored_cycle(cycle, col, c1, c2):
                    return False
        return True

    def bt(i):
        if time.time() > deadline:
            return False
        if i == m:
            return check_bicolors()
        v = cycle[i]
        # proper: cannot equal color of cycle neighbors
        prev = cycle[(i - 1) % m]
        nxt = cycle[(i + 1) % m]
        for c in range(K):
            if i > 0 and col.get(prev) == c:
                continue
            if i + 1 < m and col.get(nxt) == c:
                continue
            col[v] = c
            used[c] += 1
            if bt(i + 1):
                return True
            used[c] -= 1
            del col[v]
        return False

    if bt(0):
        return col
    return None


def _local_search(K, time_limit, deadline):
    """Local search minimizing (proper_violations, bicolored_violations)."""
    start = time.time()
    best = None
    best_cost = float('inf')

    for restart in range(4000):
        if time.time() > deadline or time.time() - start > time_limit:
            break
        colors = [random.randint(0, K - 1) for _ in range(N)]
        cur = _cost(colors, K)
        if cur == 0:
            return colors
        no_improve = 0
        while time.time() < deadline and time.time() - start < time_limit:
            if cur == 0:
                return colors
            v = random.randint(0, N - 1)
            old = colors[v]
            best_c = old
            best_mc = cur
            for c in range(K):
                if c == old:
                    continue
                colors[v] = c
                mc = _cost(colors, K)
                if mc < best_mc:
                    best_mc = mc
                    best_c = c
                colors[v] = old
            colors[v] = best_c
            if best_mc < cur:
                cur = best_mc
                no_improve = 0
            else:
                no_improve += 1
            if cur < best_cost:
                best_cost = cur
                best = list(colors)
            if no_improve > 150:
                break
    if best_cost == 0 and best is not None:
        return best
    return None


def _cost(colors, K):
    proper = 0
    for u, v in EDGES:
        if colors[u] == colors[v]:
            proper += 1
    if proper > 0:
        return proper * 1000
    used = set(colors)
    acyclic = 0
    cl = list(used)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a, b = cl[i], cl[j]
            if _has_bicolored_cycle(EVEN_CYCLE, colors, a, b):
                acyclic += 1
                break
            if _has_bicolored_cycle(ODD_CYCLE, colors, a, b):
                acyclic += 1
                break
    return proper + acyclic


def construct():
    deadline = time.time() + 80.0

    # 1) Deterministic modular search (fast, often finds 3)
    res = _modular_search()
    if res is not None:
        return res

    # 2) Cycle-decomposed backtracking: solve one 25-cycle, reuse for both
    for K in range(3, 7):
        if time.time() > deadline - 5:
            break
        cyc_col = _cycle_backtrack(EVEN_CYCLE, K, deadline - 5)
        if cyc_col is not None:
            full = [0] * N
            for v, c in cyc_col.items():
                full[v] = c
            # assign odd cycle with same pattern (shift by 1 in cycle index)
            for i in range(25):
                full[ODD_CYCLE[i]] = cyc_col[EVEN_CYCLE[i]]
            if _is_valid_global(full, K):
                return [int(c) for c in full]

    # 3) Local search fallback
    for K in range(3, 8):
        remaining = deadline - time.time()
        if remaining < 3:
            break
        res = _local_search(K, min(15.0, remaining - 3.0), deadline)
        if res is not None:
            return [int(c) for c in res]

    # Absolute fallback: proper greedy (valid, many colors)
    coloring = [0] * N
    for v in range(N):
        nc = set(coloring[nb] for nb in _ADJ[v])
        c = 0
        while c in nc:
            c += 1
        coloring[v] = c
    return [int(c) for c in coloring]

# EVOLVE-BLOCK-END
