"""Combined: local search + backtracking with incremental bicolored-cycle detection."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency and edge lists by vertex
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Pre-compute edges incident to each vertex
    incident = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append((v, idx))
        incident[v].append((u, idx))

    def full_validate(coloring):
        """Full validation: proper coloring + no bicolored cycles."""
        # Proper coloring check
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Bicolored forest check
        pair_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if cu != cv:
                key = (min(cu, cv), max(cu, cv))
                pair_edges[key].append((u, v))
        for (c1, c2), elist in pair_edges.items():
            parent = list(range(n))
            rank = [0] * n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in elist:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
        return True

    # Backtracking with incremental cycle detection
    def backtrack(k, deadline_bt):
        """Backtracking search for k-coloring with bicolored-forest constraint."""
        # Vertex ordering: degree-descending, then index
        order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
        pos_in_order = [0] * n
        for i, v in enumerate(order):
            pos_in_order[v] = i

        colors = [-1] * n
        # For each color pair, track edges that connect vertices of those two colors
        # and maintain a union-find per pair incrementally
        # We'll track: for each pair (c1,c2), a set of edges between vertices colored c1/c2
        # and use a DSU that we rebuild when needed for the specific pair being checked

        # Simpler approach: track for each pair, the list of edges, and do a quick DSU check
        # when we assign a vertex. Since k is small (3-6), the number of pairs is small.

        # For efficiency, track per-pair edge lists and do incremental DSU
        # pair_dsus: dict from (c1,c2) to (parent_dict, rank_dict)
        # But rebuilding DSU each time is expensive. Instead, track edges per pair
        # and only rebuild DSU for pairs affected by the current assignment.

        # Let's use a simpler but effective approach:
        # Maintain pair_edges: dict (c1,c2) -> list of (u,v)
        # When assigning vertex v color c, for each neighbor u with color cu != c,
        # the pair (min(c,cu), max(c,cu)) gets edge (v,u).
        # Check if adding this edge creates a cycle in that pair's subgraph.

        # For cycle detection, we need to know if u and v are already connected
        # in the subgraph induced by that pair. We'll maintain a DSU per pair.

        # Since k <= 6, there are at most 15 pairs. We'll maintain DSUs for active pairs.
        # Actually, let's just rebuild the affected pair's DSU when needed.
        # For a graph with 50 vertices and ~100 edges, this should be fast enough.

        # Strategy: maintain pair_edges dict. When checking validity of a new assignment,
        # only check the pairs that involve the new color and existing neighbor colors.

        pair_edges = defaultdict(list)  # (c1,c2) -> list of (u,v)
        # Track which pairs have been "checked" and are cycle-free
        # We'll use a simple approach: when we assign a color, check all affected pairs

        def check_pair_acyclic(c1, c2):
            """Check if the subgraph of edges between colors c1 and c2 is a forest."""
            elist = pair_edges[(min(c1,c2), max(c1,c2))]
            if not elist:
                return True
            # DSU on the vertices involved
            vertices_involved = set()
            for u, v in elist:
                vertices_involved.add(u)
                vertices_involved.add(v)
            if len(vertices_involved) <= 1:
                return True
            parent = {v: v for v in vertices_involved}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in elist:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False
                parent[ru] = rv
            return True

        def is_valid_assignment():
            """Check if current partial assignment has no bicolored cycles."""
            for key in list(pair_edges.keys()):
                if not check_pair_acyclic(key[0], key[1]):
                    return False
            return True

        def try_assign(v, c, depth):
            """Try to assign color c to vertex v at depth depth."""
            if time.time() > deadline_bt:
                return False

            # Check proper coloring: no neighbor has color c
            for u, _ in incident[v]:
                if colors[u] == c:
                    return False

            # Assign
            colors[v] = c
            # Update pair_edges: for each neighbor u with a different color, add edge
            affected_pairs = set()
            for u, _ in incident[v]:
                cu = colors[u]
                if cu != c and cu != -1:
                    key = (min(c, cu), max(c, cu))
                    pair_edges[key].append((v, u))
                    affected_pairs.add(key)

            # Check if any affected pair now has a cycle
            valid = True
            for key in affected_pairs:
                if not check_pair_acyclic(key[0], key[1]):
                    valid = False
                    break

            if valid:
                if depth == n - 1:
                    return True
                # Try next vertex in order
                next_v = order[depth + 1]
                # Try colors in a smart order
                used_colors = set(colors[order[i]] for i in range(depth + 1) if colors[order[i]] != -1)
                color_order = list(used_colors) + [c for c in range(k) if c not in used_colors]
                random.shuffle(color_order)
                for nc in color_order:
                    if try_assign(next_v, nc, depth + 1):
                        return True

            # Undo
            colors[v] = -1
            for key in affected_pairs:
                elist = pair_edges[key]
                # Remove the edge we added (it's the last one)
                if elist:
                    elist.pop()
                if not elist:
                    del pair_edges[key]

            return False

        # Start backtracking
        for c0 in range(k):
            if try_assign(order[0], c0, 0):
                return colors[:]
        return None

    # Local search (from current program, refined)
    def local_search(k, deadline_ls):
        """Local search for k-coloring."""
        best_coloring = None
        best_score = float('inf')

        for restart in range(50):
            if time.time() > deadline_ls:
                break
            coloring = [random.randint(0, k - 1) for _ in range(n)]

            # Quick full check
            if full_validate(coloring):
                return coloring

            max_steps = 20000
            no_improve = 0
            max_no_improve = 1500

            for step in range(max_steps):
                if time.time() > deadline_ls:
                    break

                # Count violations
                proper_violations = 0
                for u, v in edges:
                    if coloring[u] == coloring[v]:
                        proper_violations += 1

                # Bicolored cycle check
                pair_edges = defaultdict(list)
                for u, v in edges:
                    cu, cv = coloring[u], coloring[v]
                    if cu != cv:
                        key = (min(cu, cv), max(cu, cv))
                        pair_edges[key].append((u, v))

                cycle_count = 0
                cycle_vertices = set()
                for (c1, c2), elist in pair_edges.items():
                    if not elist:
                        continue
                    parent = list(range(n))
                    rank = [0] * n
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    for u, v in elist:
                        ru, rv = find(u), find(v)
                        if ru == rv:
                            cycle_count += 1
                            cycle_vertices.add(u)
                            cycle_vertices.add(v)
                        else:
                            if rank[ru] < rank[rv]:
                                ru, rv = rv, ru
                            parent[rv] = ru
                            if rank[ru] == rank[rv]:
                                rank[ru] += 1

                score = proper_violations * 10 + cycle_count

                if score < best_score:
                    best_score = score
                    best_coloring = coloring[:]
                    no_improve = 0
                else:
                    no_improve += 1

                if score == 0:
                    return coloring

                if no_improve > max_no_improve:
                    coloring = [random.randint(0, k - 1) for _ in range(n)]
                    no_improve = 0
                    max_no_improve = max(500, max_no_improve // 2)
                    continue

                # Choose vertex to recolor
                candidates = set(cycle_vertices)
                if proper_violations > 0:
                    for u, v in edges:
                        if coloring[u] == coloring[v]:
                            candidates.add(u)
                            candidates.add(v)
                if not candidates:
                    candidates = set(range(n))

                v = random.choice(list(candidates))
                neighbor_colors = set(coloring[u] for u in adj[v])
                available = [c for c in range(k) if c not in neighbor_colors]
                if available:
                    new_color = random.choice(available)
                else:
                    new_color = random.randint(0, k - 1)
                coloring[v] = new_color

        if best_coloring is not None and full_validate(best_coloring):
            return best_coloring
        return None

    # Try backtracking first (more likely to find optimal), then local search
    for k in range(3, n + 1):
        if time.time() > deadline:
            break

        # Backtracking with time limit
        bt_deadline = min(deadline, time.time() + 15)
        result = backtrack(k, bt_deadline)
        if result is not None:
            return result

        # Local search
        ls_deadline = min(deadline, time.time() + 15)
        result = local_search(k, ls_deadline)
        if result is not None:
            return result

    # Fallback: all distinct colors
    return list(range(n))
