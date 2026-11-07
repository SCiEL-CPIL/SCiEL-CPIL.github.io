"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    start = time.time()
    deadline = start + 85.0
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Build adjacency lists as sorted lists for fast iteration
    adj_list = [sorted(adj[i]) for i in range(n)]

    def verify(colors, k):
        """Verify proper coloring and no bicolored cycles."""
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Check no bicolored cycles using union-find per color pair
        for i in range(k):
            for j in range(i + 1, k):
                verts = [x for x in range(n) if colors[x] in (i, j)]
                if len(verts) < 3:
                    continue
                parent = list(range(n))
                rank = [0] * n

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True

                vs = set(verts)
                for u in verts:
                    for v in adj_list[u]:
                        if v in vs and u < v:
                            if not union(u, v):
                                return False
        return True

    def solve_3color():
        """Solve 3-coloring with no bicolored cycles using backtracking with constraint propagation."""
        colors = [-1] * n
        # For each color pair (i,j), maintain DSU to detect bicolored cycles
        # DSU represented as parent array indexed by vertex
        uf_parent = [list(range(n)) for _ in range(3)]  # 3 pairs: (0,1), (0,2), (1,2)
        uf_rank = [[0] * n for _ in range(3)]
        
        def uf_find(pair_idx, x):
            p = uf_parent[pair_idx]
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x

        def uf_union(pair_idx, x, y):
            p = uf_parent[pair_idx]
            r = uf_rank[pair_idx]
            rx, ry = uf_find(pair_idx, x), uf_find(pair_idx, y)
            if rx == ry:
                return False
            if r[rx] < r[ry]:
                rx, ry = ry, rx
            p[ry] = rx
            if r[rx] == r[ry]:
                r[rx] += 1
            return True

        def get_pair_idx(c1, c2):
            if c1 == 0 and c2 == 1:
                return 0
            elif c1 == 0 and c2 == 2:
                return 1
            else:
                return 2

        # Precompute: for each vertex and color, which neighbors would cause issues
        # Variable ordering: use a heuristic - try vertices in a pattern that helps propagation
        # For circulant graphs, a good ordering might be by some modular pattern
        
        # Use backtracking with forward checking
        node_count = [0]

        def can_assign(v, c):
            """Check if vertex v can be assigned color c."""
            # Check proper coloring: no neighbor has color c
            for nb in adj_list[v]:
                if colors[nb] == c:
                    return False

            # Check bicolored cycle constraint
            # For each other color c2, if v has two or more neighbors with color c2,
            # those neighbors must be in different components in the (c, c2) DSU
            for c2 in range(3):
                if c2 == c:
                    continue
                pi = get_pair_idx(min(c, c2), max(c, c2))
                nbs_c2 = [nb for nb in adj_list[v] if colors[nb] == c2]
                if len(nbs_c2) < 2:
                    continue
                # Check if any two of these neighbors are already connected in DSU
                roots = set()
                for nb in nbs_c2:
                    r = uf_find(pi, nb)
                    if r in roots:
                        return False
                    roots.add(r)
            return True

        def assign(v, c):
            """Assign color c to vertex v and update DSUs. Returns list of undo operations."""
            colors[v] = c
            undo = []
            for c2 in range(3):
                if c2 == c:
                    continue
                pi = get_pair_idx(min(c, c2), max(c, c2))
                nbs_c2 = [nb for nb in adj_list[v] if colors[nb] == c2]
                for nb in nbs_c2:
                    rv = uf_find(pi, v)
                    rn = uf_find(pi, nb)
                    if rv == rn:
                        # This shouldn't happen if can_assign was called
                        undo.append((pi, None, None))  # marker for error
                        return undo
                    old_parent = uf_parent[pi][rv] if uf_parent[pi][rv] != rv else None
                    old_rank_v = uf_rank[pi][rv]
                    old_rank_n = uf_rank[pi][rn]
                    # Union
                    if uf_rank[pi][rv] < uf_rank[pi][rn]:
                        uf_parent[pi][rv] = rn
                        undo.append((pi, rv, rn, old_rank_v, old_rank_n, False))
                    else:
                        uf_parent[pi][rn] = rv
                        if uf_rank[pi][rv] == uf_rank[pi][rn]:
                            uf_rank[pi][rv] += 1
                            undo.append((pi, rn, rv, old_rank_v, old_rank_n, True))
                        else:
                            undo.append((pi, rn, rv, old_rank_v, old_rank_n, False))
            return undo

        def unassign(v, undo):
            """Undo the assignment."""
            for item in reversed(undo):
                if item[1] is None:
                    continue
                pi, child, parent_node, old_rank_v, old_rank_n, rank_increased = item
                uf_parent[pi][child] = child
                uf_rank[pi][parent_node] = old_rank_n
                if rank_increased:
                    uf_rank[pi][parent_node] -= 1
            colors[v] = -1

        # Variable ordering: try to use a good heuristic
        # For this circulant graph, let's try ordering by vertex index but with some pattern
        # Actually, let's use a static ordering based on degree and position
        var_order = list(range(n))
        # Sort by: prefer vertices with more already-colored neighbors later (dynamic would be better)
        # For now, use a fixed ordering that might help: group by mod 3 pattern
        # Let's just use index order for simplicity, but try a few orderings

        def bt(pos):
            if node_count[0] > 10000000:
                return False
            if time.time() > deadline - 5:
                return False
            if pos == n:
                return True

            v = var_order[pos]
            # Try colors 0, 1, 2
            for c in range(3):
                if not can_assign(v, c):
                    continue

                undo = assign(v, c)
                if undo and undo[0][1] is None:
                    unassign(v, undo)
                    continue

                node_count[0] += 1
                if bt(pos + 1):
                    return True

                unassign(v, undo)

            return False

        # Try different variable orderings
        orderings = [
            list(range(n)),  # natural order
            [i for i in range(n) if i % 3 == 0] + [i for i in range(n) if i % 3 == 1] + [i for i in range(n) if i % 3 == 2],
            [i for i in range(n) if i % 5 == 0] + [i for i in range(n) if i % 5 == 1] + [i for i in range(n) if i % 5 == 2] + [i for i in range(n) if i % 5 == 3] + [i for i in range(n) if i % 5 == 4],
        ]

        for ordering in orderings:
            var_order = ordering
            node_count[0] = 0
            # Reset DSUs
            for pi in range(3):
                uf_parent[pi] = list(range(n))
                uf_rank[pi] = [0] * n
            if bt(0):
                return colors[:]
            # Reset colors
            colors = [-1] * n
            for pi in range(3):
                uf_parent[pi] = list(range(n))
                uf_rank[pi] = [0] * n

        return None

    # First, try to find a 3-coloring
    result = solve_3color()
    if result is not None:
        if verify(result, 3):
            return [int(c) for c in result]

    # If 3-coloring fails, try 4-coloring with a simpler approach
    # Use greedy with backtracking for 4 colors
    def solve_4color():
        colors = [-1] * n
        node_count = [0]

        def can_assign(v, c):
            for nb in adj_list[v]:
                if colors[nb] == c:
                    return False
            # For 4 colors, the bicolored cycle check is more complex
            # But let's just do proper coloring first and verify at the end
            return True

        def bt(pos):
            if node_count[0] > 5000000:
                return False
            if time.time() > deadline - 5:
                return False
            if pos == n:
                return True

            v = pos
            for c in range(4):
                if not can_assign(v, c):
                    continue
                colors[v] = c
                node_count[0] += 1
                if bt(pos + 1):
                    return True
                colors[v] = -1
            return False

        if bt(0):
            return colors[:]
        return None

    result = solve_4color()
    if result is not None:
        if verify(result, 4):
            return [int(c) for c in result]

    # Fallback: try a known pattern for 3 colors
    # For circulant C(50, {1,7}), try patterns based on modular arithmetic
    patterns_3 = [
        [i % 3 for i in range(n)],
        [(i * 2) % 3 for i in range(n)],
        [(i * 3) % 3 for i in range(n)],
        [(i * 7) % 3 for i in range(n)],
        [(i * 11) % 3 for i in range(n)],
    ]
    for pat in patterns_3:
        if verify(pat, 3):
            return [int(c) for c in pat]

    # Try 4-color patterns
    patterns_4 = [
        [i % 4 for i in range(n)],
        [(i * 2) % 4 for i in range(n)],
        [(i * 3) % 4 for i in range(n)],
        [(i * 7) % 4 for i in range(n)],
    ]
    for pat in patterns_4:
        if verify(pat, 4):
            return [int(c) for c in pat]

    # Last resort: use many colors (always valid)
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
