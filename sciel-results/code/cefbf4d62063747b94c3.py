"""Rewritten solver: minimize distinct colors with bicolored-forest constraint."""
import random
import time

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

DEADLINE = time.time() + 75  # safety margin before 90s kill

def _check_valid(coloring):
    """Return True if coloring is proper and every bicolored subgraph is a forest."""
    # Proper coloring check
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
    # Bicolored forest check via union-find per color-pair
    # For each pair of colors (a,b), consider the subgraph of edges whose endpoints
    # have colors a and b. This must be a forest.
    # Efficient: for each edge (u,v), the pair is (coloring[u], coloring[v]).
    # We need: for each pair, no cycle. Use union-find keyed by pair.
    parent = {}
    def find(x):
        if x not in parent:
            parent[x] = x
            return x
        # path compression
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False  # cycle
        parent[ra] = rb
        return True
    for u, v in EDGES:
        cu, cv = coloring[u], coloring[v]
        if cu == cv:
            return False
        # Key: (pair, vertex)
        if cu > cv:
            cu, cv = cv, cu
        # Two endpoints: (pair, u) and (pair, v)
        key_u = (cu, cv, u)
        key_v = (cu, cv, v)
        if not union(key_u, key_v):
            return False
    return True

def _distinct_count(coloring):
    return len(set(coloring))

def _random_proper_coloring(max_colors, rng):
    """Try to build a proper coloring using at most max_colors colors via greedy+backtrack-lite."""
    # Order vertices by degree descending for greedy
    order = sorted(range(N), key=lambda v: -len(ADJ[v]))
    coloring = [-1] * N
    # Greedy with random tie-breaking
    for v in order:
        used = set()
        for nb in ADJ[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        avail = [c for c in range(max_colors) if c not in used]
        if not avail:
            return None
        coloring[v] = rng.choice(avail)
    if _check_valid(coloring):
        return coloring
    return None

def _local_search(coloring, max_colors, rng, time_budget):
    """Local search: try to reduce distinct colors and fix violations."""
    coloring = list(coloring)
    best = list(coloring)
    best_score = _distinct_count(coloring)
    start = time.time()
    # First, ensure validity
    if not _check_valid(coloring):
        return None
    # Try to reduce number of distinct colors by recoloring vertices
    # Strategy: pick a color that is used least, try to eliminate it by moving those vertices to other colors
    while time.time() - start < time_budget:
        # Count color usage
        from collections import Counter
        cnt = Counter(coloring)
        # Try to eliminate the rarest color
        rare_color = min(cnt, key=lambda c: cnt[c])
        if cnt[rare_color] == 0:
            break
        # Vertices with this color
        verts = [v for v in range(N) if coloring[v] == rare_color]
        rng.shuffle(verts)
        improved = False
        for v in verts:
            # Try to recolor v to another color
            used = set()
            for nb in ADJ[v]:
                used.add(coloring[nb])
            avail = [c for c in range(max_colors) if c not in used and c != rare_color]
            if not avail:
                continue
            # Try each available color, pick one that keeps validity
            rng.shuffle(avail)
            for new_c in avail:
                old = coloring[v]
                coloring[v] = new_c
                if _check_valid(coloring):
                    improved = True
                    break
                coloring[v] = old
            if improved:
                break
        if improved:
            new_score = _distinct_count(coloring)
            if new_score < best_score:
                best = list(coloring)
                best_score = new_score
        else:
            # No more eliminations possible; try swapping colors to reduce distinct count
            # Try to merge two colors
            colors_used = sorted(set(coloring))
            if len(colors_used) <= 1:
                break
            # Try to recolor all vertices of one color to another
            for c_from in colors_used:
                verts_from = [v for v in range(N) if coloring[v] == c_from]
                if len(verts_from) > 10:
                    continue
                for c_to in colors_used:
                    if c_to == c_from:
                        continue
                    # Check if we can recolor all verts_from to c_to
                    ok = True
                    for v in verts_from:
                        for nb in ADJ[v]:
                            if coloring[nb] == c_to:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    # Try it
                    backup = list(coloring)
                    for v in verts_from:
                        coloring[v] = c_to
                    if _check_valid(coloring):
                        new_score = _distinct_count(coloring)
                        if new_score < best_score:
                            best = list(coloring)
                            best_score = new_score
                        break
                    else:
                        coloring = backup
    return best

def _solve():
    rng = random.Random(42)
    best_solution = None
    best_score = N + 1
    start = time.time()
    
    # Try increasing max_colors from 3 upward
    for max_colors in range(3, N + 1):
        if time.time() - start > DEADLINE - time.time() - start + 10:
            break
        if time.time() > DEADLINE - 5:
            break
        # Try multiple random initial colorings
        for attempt in range(50):
            if time.time() > DEADLINE - 5:
                break
            coloring = _random_proper_coloring(max_colors, rng)
            if coloring is None:
                continue
            # Local search to reduce distinct colors
            ls_result = _local_search(coloring, max_colors, rng, time_budget=2.0)
            if ls_result is not None:
                score = _distinct_count(ls_result)
                if score < best_score:
                    best_score = score
                    best_solution = ls_result
            # Also check the initial coloring
            score = _distinct_count(coloring)
            if score < best_score:
                best_score = score
                best_solution = coloring
        # If we found a solution with few colors, try to improve further
        if best_solution is not None and best_score <= max_colors:
            # Try to reduce further
            pass
    return best_solution

def construct():
    sol = _solve()
    if sol is None:
        # Fallback: identity coloring (always valid, 50 colors)
        return list(range(N))
    # Ensure types are Python int and it's a list
    return [int(c) for c in sol]
# EVOLVE-BLOCK-END
