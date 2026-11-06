import time
import random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin under 150 s kill

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges):
        """Return list of Python ints; bit j set in conflict[i] iff edge i conflicts with edge j."""
        m = len(edges)
        # Map vertex -> list of edge indices incident to it
        v2e = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v2e[u].append(i)
            v2e[v].append(i)

        # Map edge tuple -> index for quick lookup
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        # Original graph adjacency (vertex -> set of vertices) for "connected by edge" check
        orig_adj = defaultdict(set)
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)

        conflict = [0] * m

        # 1) Edges sharing a vertex conflict
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]] |= 1 << lst[b]
                    conflict[lst[b]] |= 1 << lst[a]

        # 2) Disjoint edges connected by an edge conflict
        for i in range(m):
            u, v = edges[i]
            # For each other edge j that doesn't share a vertex with i
            # We need to check if u or v is adjacent to x or y where (x,y) = edges[j]
            # Efficient: for each neighbor w of u, find edges incident to w not in v2e[u]
            # Actually let's just iterate: for each vertex w adjacent to u,
            # for each edge j in v2e[w] that is not incident to u or v...
            # Simpler: iterate all j, but skip those sharing vertex (already handled)
            # For m up to ~240, O(m^2) is fine (~57600 checks)
            for j in range(m):
                if i == j:
                    continue
                if (conflict[i] >> j) & 1:
                    continue  # already conflict (share vertex)
                x, y = edges[j]
                # Check if any endpoint of edge i is adjacent to any endpoint of edge j
                if (u in orig_adj[x] or u in orig_adj[y] or
                        v in orig_adj[x] or v in orig_adj[y]):
                    conflict[i] |= 1 << j
                    conflict[j] |= 1 << i

        return conflict

    def greedy_color(conflict, m):
        """Fast greedy: order by descending degree, assign smallest available color."""
        deg = [bin(conflict[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda i: -deg[i])
        colors = [-1] * m
        for v in order:
            # Find colors used by neighbors
            used = set()
            mask = conflict[v]
            for u in range(m):
                if (mask >> u) & 1 and colors[u] != -1:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return colors

    def local_search_reduce(colors, conflict, m, deadline):
        """Iteratively try to reduce color count by recoloring vertices to smaller colors."""
        if m == 0:
            return colors, 0
        max_c = max(colors)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Process vertices in order of decreasing current color
            for v in sorted(range(m), key=lambda x: -colors[x]):
                cur = colors[v]
                if cur == 0:
                    continue
                # Try each color 0..cur-1
                for c in range(cur):
                    # Check if color c is available: no neighbor has color c
                    ok = True
                    mask = conflict[v]
                    for u in range(m):
                        if (mask >> u) & 1 and colors[u] == c:
                            ok = False
                            break
                    if ok:
                        colors[v] = c
                        improved = True
                        break
        max_c = max(colors) if m > 0 else 0
        return colors, max_c + 1

    def branch_and_bound(conflict, m, deadline, initial_bound):
        """
        Iterative deepening branch-and-bound for exact minimum coloring.
        Uses DSATUR-style vertex ordering with bitmasks for speed.
        Returns best coloring found (possibly not optimal if timed out).
        """
        if m == 0:
            return [], 0

        deg = [bin(conflict[i]).count('1') for i in range(m)]
        colors = [-1] * m
        sat = [0] * m  # bitmask of colors seen among neighbors
        best_colors = [None] * (m + 1)  # best_colors[k] = coloring using k colors
        best_k = initial_bound

        # Precompute for each vertex, the list of neighbors sorted by degree desc
        # Actually for bitmask we just use conflict[i] directly

        def feasible(k):
            """Try to color with at most k colors. Returns coloring or None."""
            colors = [-1] * m
            sat = [0] * m

            # Order vertices by a static heuristic: descending degree
            order = sorted(range(m), key=lambda i: -deg[i])

            def dfs(idx):
                if time.time() > deadline:
                    return False
                if idx == m:
                    return True

                v = order[idx]
                # Try colors 0..k-1
                for c in range(k):
                    bit = 1 << c
                    if sat[v] & bit:
                        continue
                    # Assign
                    colors[v] = c
                    # Update sat for neighbors
                    changed = []
                    ok = True
                    mask = conflict[v]
                    for u in range(m):
                        if (mask >> u) & 1:
                            if not (sat[u] & bit):
                                sat[u] |= bit
                                changed.append(u)
                    if dfs(idx + 1):
                        return True
                    # Undo
                    for u in changed:
                        sat[u] &= ~bit
                    colors[v] = -1
                return False

            return dfs(0)

        # Start from initial_bound and go down
        for k in range(initial_bound, 0, -1):
            if time.time() > deadline:
                break
            result = feasible(k)
            if result:
                best_k = k
                # Reconstruct coloring
                # feasible() modified local colors; we need to re-run to get it
                # Let's refactor: have feasible return the coloring
                pass

        # The above approach is problematic because feasible() uses local state.
        # Let me redo with a cleaner approach.
        return None  # fallback

    def exact_coloring(conflict, m, deadline, greedy_k):
        """
        Iterative deepening: try k = greedy_k, greedy_k-1, ..., 1
        For each k, run a DFS with DSATUR-style ordering.
        Return the best coloring found.
        """
        if m == 0:
            return [], 0

        deg = [bin(conflict[i]).count('1') for i in range(m)]

        def try_color(k):
            """Attempt to color with exactly k colors (colors 0..k-1). Return coloring or None."""
            colors = [-1] * m
            sat = [0] * m  # bitmask of forbidden colors per vertex
            uncolored = set(range(m))

            def select_vertex():
                # Max saturation, then max degree
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    s = bin(sat[v]).count('1')
                    d = deg[v]
                    if s > best_sat or (s == best_sat and d > best_deg):
                        best_sat = s
                        best_deg = d
                        best_v = v
                return best_v

            def dfs():
                if time.time() > deadline:
                    return 'timeout'
                if not uncolored:
                    return True

                v = select_vertex()
                if v == -1:
                    return True

                # Try colors 0..k-1
                for c in range(k):
                    bit = 1 << c
                    if sat[v] & bit:
                        continue
                    # Assign color c to v
                    colors[v] = c
                    uncolored.discard(v)

                    # Update saturation of neighbors
                    changed = []
                    mask = conflict[v]
                    for u in range(m):
                        if (mask >> u) & 1 and u in uncolored:
                            if not (sat[u] & bit):
                                sat[u] |= bit
                                changed.append(u)

                    result = dfs()
                    if result == 'timeout':
                        return 'timeout'
                    if result:
                        return True

                    # Undo
                    for u in changed:
                        sat[u] &= ~bit
                    uncolored.add(v)
                    colors[v] = -1

                return False

            result = dfs()
            if result == 'timeout':
                return None
            if result:
                return colors
            return None

        # Start from greedy_k and decrease
        best_coloring = None
        for k in range(greedy_k, 0, -1):
            if time.time() > deadline:
                break
            coloring = try_color(k)
            if coloring is not None:
                best_coloring = coloring
                break  # Found minimum, stop
        if best_coloring is None:
            # Timeout during search; use greedy
            best_coloring = greedy_color(conflict, m)
            best_coloring, _ = local_search_reduce(best_coloring, conflict, m, deadline)
        return best_coloring, (max(best_coloring) + 1 if m > 0 else 0)

    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    for n, d, seed in specs:
        if time.time() > DEADLINE - 5:
            # Not enough time; use very fast fallback
            edges = gen_instance(n, d, seed)
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            conflict = build_conflict_bitmasks(edges)
            colors = greedy_color(conflict, m)
            results.append(colors)
            continue

        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            results.append([])
            continue

        conflict = build_conflict_bitmasks(edges)

        # Phase 1: Greedy + local search to get an upper bound
        colors = greedy_color(conflict, m)
        colors, k_upper = local_search_reduce(colors, conflict, m, DEADLINE - 10)

        # Phase 2: Exact search from k_upper downward
        final_colors, k_final = exact_coloring(conflict, m, DEADLINE - 5, k_upper)
        results.append(final_colors)

    return results
