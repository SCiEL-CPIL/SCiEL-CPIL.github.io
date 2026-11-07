"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Precompute adjacency
ADJ = [[] for _ in range(N)]
for a, b in EDGES:
    ADJ[a].append(b)
    ADJ[b].append(a)

# Precompute edge index per vertex (for fast "which edges incident to v")
# We'll track color per vertex; an edge is "active" for color pair (c1,c2) if
# both endpoints are assigned and have those colors.

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    # We'll search k = 2, 3, 4, ... increasing.
    # For each k, run a backtracking search with union-find per color-pair.
    #
    # State:
    #   color[v] for v in 0..N-1, -1 if unassigned.
    #   For each color pair (c1, c2) with c1<c2, we maintain a union-find over N
    #   vertices, but only vertices assigned colors c1 or c2 participate.
    #   Actually simpler: maintain DSU per pair; when we assign vertex v to color c,
    #   for each neighbor u already assigned color c' (c' != c), we union v and u
    #   in the DSU for pair (min(c,c'), max(c,c')). If they're already in the same
    #   set, that creates a bicolored cycle -> backtrack.
    #
    # To make this efficient, we maintain for each pair (c1,c2) a DSU.
    # Number of pairs for k colors is k*(k-1)/2. For k=5 that's 10 pairs.
    #
    # We also need to undo unions on backtrack. We'll use a stack of union operations.

    # Order vertices for assignment: use a static order that helps.
    # Try ordering by descending degree first, but all have degree 4.
    # Use a heuristic: assign in an order that connects the graph well.
    # Simple: 0, 1, 2, ..., 49. Or try a few orderings.

    # For each k, try to find a valid coloring.
    for k in range(2, 10):
        if time.time() > deadline:
            break
        # DSU per pair
        # pair_index: map (c1,c2) -> index
        pairs = []
        pair_map = {}
        for c1 in range(k):
            for c2 in range(c1+1, k):
                idx = len(pairs)
                pairs.append((c1, c2))
                pair_map[(c1, c2)] = idx
        npairs = len(pairs)

        # DSU: parent arrays, one per pair. We use a flat list.
        # parent[pair_idx][v] for v in 0..N-1
        # We'll use path compression but need undo, so we'll record changes.
        # Simpler: use union by rank without path compression, and record each union's effect.

        # Actually, for small N=50 and limited depth, let's use a simpler approach:
        # Maintain for each pair a set of "components" as a dict: vertex -> component_id.
        # When we union two vertices in a pair, we merge their components.
        # We record the merge for undo.

        # Even simpler: use a DSU with a stack. Let me implement a DSU that supports
        # undo via a stack of (pair_idx, a, b, parent_a, parent_b, rank_a, rank_b) or similar.

        # Let me use: for each pair, parent[pair][v] and rank[pair][v].
        # Union operation: find roots, if same -> cycle. Else, attach smaller to larger.
        # Record: (pair_idx, root_small, root_large, rank_large_before)
        # Undo: parent[root_small] = root_small, rank[root_large] = rank_large_before.

        parent = [[v for v in range(N)] for _ in range(npairs)]
        rank = [[0] * N for _ in range(npairs)]
        color = [-1] * N
        union_stack = []  # list of (pair_idx, small_root, large_root, rank_large_before)

        def find(pi, v):
            while parent[pi][v] != v:
                v = parent[pi][v]
            return v

        def do_union(pi, a, b):
            ra = find(pi, a)
            rb = find(pi, b)
            if ra == rb:
                return False  # cycle
            # union by rank
            if rank[pi][ra] > rank[pi][rb]:
                ra, rb = rb, ra
            # ra has rank <= rb
            parent[pi][ra] = rb
            if rank[pi][ra] == rank[pi][rb]:
                rank[pi][rb] += 1
            union_stack.append((pi, ra, rb, rank[pi][rb] - (1 if rank[pi][ra] == rank[pi][rb] - 1 else 0)))
            # Hmm, this is getting complicated. Let me just record (pi, ra, rb) and the old rank of rb.
            return True

        # Let me redo: record (pi, ra, rb, old_rank_rb)
        def do_union2(pi, a, b):
            ra = find(pi, a)
            rb = find(pi, b)
            if ra == rb:
                return False
            if rank[pi][ra] > rank[pi][rb]:
                ra, rb = rb, ra
            old_rank_rb = rank[pi][rb]
            parent[pi][ra] = rb
            if rank[pi][ra] == rank[pi][rb]:
                rank[pi][rb] += 1
            union_stack.append((pi, ra, rb, old_rank_rb))
            return True

        def undo_union():
            pi, ra, rb, old_rank_rb = union_stack.pop()
            parent[pi][ra] = ra
            rank[pi][rb] = old_rank_rb

        # Vertex ordering: try a few. Use degree ordering (all same), so use index order.
        # But let's try a smarter order: BFS from vertex 0.
        # For now, use 0..49.
        order = list(range(N))

        # Backtracking
        def backtrack(pos):
            if time.time() > deadline:
                return None
            if pos == N:
                return color[:]
            v = order[pos]
            # Try colors 0..k-1
            for c in range(k):
                # Check neighbors: no neighbor can have color c
                ok = True
                for u in ADJ[v]:
                    if color[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                # Assign v = c, do unions with colored neighbors
                color[v] = c
                stack_size = len(union_stack)
                valid = True
                for u in ADJ[v]:
                    cu = color[u]
                    if cu >= 0 and cu != c:
                        pi = pair_map[(c, cu) if c < cu else (cu, c)]
                        if not do_union2(pi, v, u):
                            valid = False
                            break
                if valid:
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result
                # Undo
                while len(union_stack) > stack_size:
                    undo_union()
                color[v] = -1
            return None

        result = backtrack(0)
        if result is not None:
            # Verify
            assert all(isinstance(x, int) for x in result)
            assert all(0 <= x < k for x in result)
            # Check adjacency
            for a, b in EDGES:
                assert result[a] != result[b]
            # Check bicolored forest
            from collections import defaultdict
            # For each pair, check no cycle
            for pi, (c1, c2) in enumerate(pairs):
                # Build subgraph of vertices with color c1 or c2
                verts = [v for v in range(N) if result[v] in (c1, c2)]
                if len(verts) <= 1:
                    continue
                # Check if induced subgraph is a forest: edges <= vertices - components
                # Use DFS to count components and edges
                adj_sub = defaultdict(list)
                for a, b in EDGES:
                    if result[a] in (c1, c2) and result[b] in (c1, c2):
                        adj_sub[a].append(b)
                        adj_sub[b].append(a)
                # Count edges
                e_count = 0
                for v in adj_sub:
                    e_count += len(adj_sub[v])
                e_count //= 2
                # Count components
                visited = set()
                comps = 0
                for v in verts:
                    if v not in visited:
                        comps += 1
                        stack = [v]
                        visited.add(v)
                        while stack:
                            x = stack.pop()
                            for y in adj_sub[x]:
                                if y not in visited:
                                    visited.add(y)
                                    stack.append(y)
                if e_count > len(verts) - comps:
                    raise AssertionError("Bicolored cycle detected!")
            return [int(x) for x in result]

    # Fallback: return the seed solution
    return [int(x) for x in range(N)]

# EVOLVE-BLOCK-END
