import time
import random
from functools import lru_cache

def construct():
    start = time.time()
    DEADLINE = start + 135.0
    N = 16
    FULL = (1 << N) - 1

    def time_ok():
        return time.time() < DEADLINE

    def is_3_colorable_fast(adj, edge_count):
        """
        Optimized 3-colorability check using bitmask DP with memoization.
        Key optimizations:
        1. Precompute neighbor masks in original vertex order
        2. Use iterative deepening with pruning
        3. Early termination on success
        """
        if edge_count <= 2 * N - 3:
            return True

        # Build adjacency masks
        adj_mask = list(adj)

        # Try all 3^N colorings is too slow; use backtracking with bitmasks
        # Assign colors greedily with constraint propagation

        # Fast path: check if graph is a forest or has small treewidth
        # For n=16, use exact backtracking with good ordering

        # Order vertices by degree (descending) for better pruning
        deg = [bin(adj_mask[i]).count('1') for i in range(N)]
        order = list(range(N))
        order.sort(key=lambda i: -deg[i])

        # Create position mapping
        pos_in_order = [0] * N
        for i, v in enumerate(order):
            pos_in_order[v] = i

        # Build ordered adjacency masks (in terms of positions)
        ordered_adj = [0] * N
        for i, v in enumerate(order):
            m = 0
            for u in range(N):
                if (adj_mask[v] >> u) & 1:
                    m |= (1 << pos_in_order[u])
            ordered_adj[i] = m

        # Backtracking with memoization
        # State: (position, color_mask_0, color_mask_1, color_mask_2)
        # But that's too large. Instead, track which vertices are colored and their colors.

        # Better: use the fact that we process in order, so state is just the color assignments
        # for vertices 0..i-1 in the ordered sequence.

        # For n=16, we can use a different approach:
        # Try to find a valid 3-coloring via DFS with constraint propagation

        memo = {}

        def dfs(i, colors):
            """
            i: next vertex to color (in order)
            colors: tuple of 3 bitmasks showing which ordered positions have each color
            """
            if i == N:
                return True

            key = (i, colors[0], colors[1], colors[2])
            if key in memo:
                return memo[key]

            v = order[i]
            mask = ordered_adj[i]

            # For each color, check if no neighbor has that color
            for c in range(3):
                if (colors[c] & mask) == 0:
                    newc = colors[c] | (1 << i)
                    if c == 0:
                        nc = (newc, colors[1], colors[2])
                    elif c == 1:
                        nc = (colors[0], newc, colors[2])
                    else:
                        nc = (colors[0], colors[1], newc)
                    if dfs(i + 1, nc):
                        memo[key] = True
                        return True

            memo[key] = False
            return False

        result = dfs(0, (0, 0, 0))
        return result

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count_of(adj):
        cnt = 0
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    cnt += 1
        return cnt

    def edges_from_adj(adj):
        return [[u, v] for u in range(N) for v in range(u + 1, N) if (adj[u] >> v) & 1]

    best_edges = None
    best_count = -1

    # Start with a known dense triangle-free non-3-colorable graph as seed
    # Clebsch graph: 16 vertices, 40 edges, triangle-free, not 3-colorable
    clebsch_vertices = [i for i in range(32) if bin(i).count('1') % 2 == 0]
    clebsch_adj = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            xor_val = clebsch_vertices[i] ^ clebsch_vertices[j]
            if bin(xor_val).count('1') == 4:
                add_edge(clebsch_adj, i, j)

    if not is_3_colorable_fast(clebsch_adj, 40):
        best_count = 40
        best_edges = edges_from_adj(clebsch_adj)

    # Local search: try to add edges to the best graph
    def local_search(adj, max_iter=500):
        nonlocal best_count, best_edges
        ec = edge_count_of(adj)
        for _ in range(max_iter):
            if not time_ok():
                break
            improved = False
            # Try all non-edges
            for u in range(N):
                for v in range(u + 1, N):
                    if not ((adj[u] >> v) & 1):
                        # Check triangle-free
                        if (adj[u] & adj[v]) == 0:
                            add_edge(adj, u, v)
                            if not is_3_colorable_fast(adj, ec + 1):
                                ec += 1
                                improved = True
                                if ec > best_count:
                                    best_count = ec
                                    best_edges = edges_from_adj(adj)
                            else:
                                remove_edge(adj, u, v)
            if not improved:
                break
        return adj

    # Multi-start local search
    rng = random.Random(42)
    attempts = 0

    while time_ok() and attempts < 2000:
        attempts += 1

        # Randomized greedy construction
        adj = [0] * N
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if not ((adj[u] >> v) & 1):
                if (adj[u] & adj[v]) == 0:
                    add_edge(adj, u, v)

        ec = edge_count_of(adj)
        if is_3_colorable_fast(adj, ec):
            continue

        if ec > best_count:
            best_count = ec
            best_edges = edges_from_adj(adj)

        # Local search from this starting point
        adj = local_search(adj, max_iter=200)
        ec = edge_count_of(adj)
        if ec > best_count:
            best_count = ec
            best_edges = edges_from_adj(adj)

        # Also try local search from the Clebsch graph with random perturbations
        if attempts % 10 == 0:
            adj2 = clebsch_adj[:]
            # Randomly remove some edges
            for _ in range(rng.randint(0, 5)):
                edges_list = edges_from_adj(adj2)
                if edges_list:
                    u, v = rng.choice(edges_list)
                    remove_edge(adj2, u, v)
            adj2 = local_search(adj2, max_iter=100)
            ec2 = edge_count_of(adj2)
            if ec2 > best_count:
                best_count = ec2
                best_edges = edges_from_adj(adj2)

    if best_edges is not None:
        return best_edges

    # Fallback: return Clebsch graph (40 edges)
    return edges_from_adj(clebsch_adj)
