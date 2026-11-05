import time
import random
from functools import lru_cache

def construct():
    start = time.time()
    DEADLINE = start + 140.0
    N = 16
    FULL = (1 << N) - 1
    adj = [0] * N
    edge_count = 0

    def time_ok():
        return time.time() < DEADLINE

    def is_3_colorable(adj, edge_count):
        # Fast exact 3-colorability test.
        if edge_count <= 2 * N - 3:
            return True  # any graph with m <= 2n-3 edges is 3-colorable
        # DSATUR-style ordering (by degree, descending)
        deg = [bin(adj[i]).count("1") for i in range(N)]
        order = list(range(N))
        order.sort(key=lambda i: -deg[i])
        pos = {v: i for i, v in enumerate(order)}
        # Build ordered adjacency as sorted lists
        adj_list = []
        for v in order:
            nb = [order.index(u) for u in range(N) if (adj[v] >> u) & 1 and u != v]
            nb.sort()
            adj_list.append(nb)
        # Convert to bitmask over ordered positions
        adj_mask = [0] * N
        for i, nb in enumerate(adj_list):
            m = 0
            for j in nb:
                m |= (1 << j)
            adj_mask[i] = m

        @lru_cache(maxsize=200000)
        def bt(i, colors):
            if i == N:
                return True
            mask = adj_mask[i]
            avail = 7
            # colors is a tuple of (c0, c1, c2) masks
            for c in range(3):
                if (colors[c] & mask) == 0:
                    newc = colors[c] | (1 << i)
                    if c == 0:
                        nc = (newc, colors[1], colors[2])
                    elif c == 1:
                        nc = (colors[0], newc, colors[2])
                    else:
                        nc = (colors[0], colors[1], newc)
                    if bt(i + 1, nc):
                        return True
            return False

        return bt(0, (0, 0, 0))

    def add_edge(u, v):
        nonlocal edge_count
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_count += 1

    def remove_edge(u, v):
        nonlocal edge_count
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_count -= 1

    # Greedy: build a dense triangle-free graph (randomized maximal triangle-free)
    def build_maximal():
        nonlocal edge_count
        adj = [0] * N
        edge_count = 0
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    # check triangle-free: no common neighbor
                    if (adj[u] & adj[v]) == 0:
                        add_edge(u, v)
        # return edges
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    best_edges = None
    best_count = -1

    # Try many random greedy constructions, keep densest non-3-colorable
    rng = random.Random(12345)
    attempts = 0
    while time_ok() and attempts < 4000:
        attempts += 1
        # randomize the order of edge consideration
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(pairs)
        adj = [0] * N
        edge_count = 0
        for u, v in pairs:
            if not ((adj[u] >> v) & 1):
                if (adj[u] & adj[v]) == 0:
                    add_edge(u, v)
        if is_3_colorable(adj, edge_count):
            continue
        if edge_count > best_count:
            best_count = edge_count
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if (adj[u] >> v) & 1]
        # Local search: try adding edges
        changed = True
        while changed and time_ok():
            changed = False
            for u in range(N):
                for v in range(u + 1, N):
                    if not ((adj[u] >> v) & 1):
                        if (adj[u] & adj[v]) == 0:
                            add_edge(u, v)
                            if not is_3_colorable(adj, edge_count):
                                changed = True
                            else:
                                remove_edge(u, v)
        if best_edges and edge_count > best_count:
            best_count = edge_count
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if (adj[u] >> v) & 1]

    if best_edges is not None:
        return best_edges

    # Fallback: return Clebsch graph (40 edges)
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    edges = []
    n = len(vertices)
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vertices[u_idx] ^ vertices[v_idx]
            if bin(xor_val).count('1') == 4:
                edges.append([u_idx, v_idx])
    return edges
