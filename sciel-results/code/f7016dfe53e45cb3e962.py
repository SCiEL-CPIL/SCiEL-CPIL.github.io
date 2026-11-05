import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 16
    total_edges = n * (n - 1) // 2  # 120

    # Bitmask adjacency
    adj = [0] * n
    edge_count = 0

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_set = set(all_pairs)

    def check_q3_with_edge(u, v):
        """Check if adding edge (u,v) creates a Q3.
        A Q3 containing edge (u,v): u,v are opposite (dist 3).
        6 middle vertices: 3 neighbors of u (forming a triangle),
        3 neighbors of v (forming a triangle), with perfect matching.
        """
        nu = [x for x in range(n) if x != u and x != v and (adj[u] >> x) & 1]
        nv = [x for x in range(n) if x != u and x != v and (adj[v] >> x) & 1]

        if len(nu) < 3 or len(nv) < 3:
            return False

        # Find triangles among nu (3-cycles)
        tri_u = []
        for a, b, c in combinations(nu, 3):
            if ((adj[a] >> b) & 1) and ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                tri_u.append((a, b, c))

        if not tri_u:
            return False

        # Find triangles among nv
        tri_v = []
        for a, b, c in combinations(nv, 3):
            if ((adj[a] >> b) & 1) and ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                tri_v.append((a, b, c))

        if not tri_v:
            return False

        # Check if there's a perfect matching between some triangle in tri_u
        # and some triangle in tri_v
        for (a, b, c) in tri_u:
            for (x, y, z) in tri_v:
                # Check all 6 possible matchings
                if ((adj[a] >> x) & 1) and ((adj[b] >> y) & 1) and ((adj[c] >> z) & 1):
                    return True
                if ((adj[a] >> x) & 1) and ((adj[b] >> z) & 1) and ((adj[c] >> y) & 1):
                    return True
                if ((adj[a] >> y) & 1) and ((adj[b] >> x) & 1) and ((adj[c] >> z) & 1):
                    return True
                if ((adj[a] >> y) & 1) and ((adj[b] >> z) & 1) and ((adj[c] >> x) & 1):
                    return True
                if ((adj[a] >> z) & 1) and ((adj[b] >> x) & 1) and ((adj[c] >> y) & 1):
                    return True
                if ((adj[a] >> z) & 1) and ((adj[b] >> y) & 1) and ((adj[c] >> x) & 1):
                    return True
        return False

    def add_edge(u, v):
        nonlocal adj, edge_count
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_count += 1

    def remove_edge(u, v):
        nonlocal adj, edge_count
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_count -= 1

    best_edge_count = 0
    best_adj = None

    def save_best():
        nonlocal best_edge_count, best_adj
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = adj[:]

    # Strategy 1: Greedy with multiple random restarts
    for attempt in range(30):
        if time.time() > deadline - 15:
            break

        # Reset
        adj = [0] * n
        edge_count = 0

        # Random edge ordering
        random.seed(attempt * 7919 + 31337)
        random.shuffle(all_pairs)

        for (u, v) in all_pairs:
            if time.time() > deadline - 10:
                break
            if not check_q3_with_edge(u, v):
                add_edge(u, v)

        save_best()

        # Local improvement: try adding remaining edges
        if time.time() > deadline - 8:
            break

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for (u, v) in all_pairs:
                if time.time() > deadline - 5:
                    break
                if not ((adj[u] >> v) & 1) and not check_q3_with_edge(u, v):
                    add_edge(u, v)
                    improved = True
                    save_best()

    # Strategy 2: Start from K_{3,13} and add edges
    if time.time() < deadline - 12:
        adj = [0] * n
        edge_count = 0
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        edge_count = 39

        random.seed(99999)
        remaining = [(i, j) for (i, j) in all_pairs if not ((adj[i] >> j) & 1)]
        random.shuffle(remaining)

        for (u, v) in remaining:
            if time.time() > deadline - 8:
                break
            if not check_q3_with_edge(u, v):
                add_edge(u, v)

        save_best()

        # Local improvement
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for (u, v) in remaining:
                    if time.time() > deadline - 3:
                        break
                    if not ((adj[u] >> v) & 1) and not check_q3_with_edge(u, v):
                        add_edge(u, v)
                        improved = True
                        save_best()

    # Strategy 3: Start from K_{4,12}
    if time.time() < deadline - 12:
        adj = [0] * n
        edge_count = 0
        for u in range(4):
            for v in range(4, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        edge_count = 48

        random.seed(88888)
        remaining = [(i, j) for (i, j) in all_pairs if not ((adj[i] >> j) & 1)]
        random.shuffle(remaining)

        for (u, v) in remaining:
            if time.time() > deadline - 8:
                break
            if not check_q3_with_edge(u, v):
                add_edge(u, v)

        save_best()

        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for (u, v) in remaining:
                    if time.time() > deadline - 3:
                        break
                    if not ((adj[u] >> v) & 1) and not check_q3_with_edge(u, v):
                        add_edge(u, v)
                        improved = True
                        save_best()

    # Strategy 4: Turan-like - complete multipartite
    if time.time() < deadline - 12:
        # K_{4,4,8}
        adj = [0] * n
        edge_count = 0
        parts = [list(range(4)), list(range(4, 8)), list(range(8, 16))]
        for pi in range(len(parts)):
            for pj in range(pi + 1, len(parts)):
                for u in parts[pi]:
                    for v in parts[pj]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        edge_count = sum(bin(a).count('1') for a in adj) // 2

        random.seed(77777)
        remaining = [(i, j) for (i, j) in all_pairs if not ((adj[i] >> j) & 1)]
        random.shuffle(remaining)

        for (u, v) in remaining:
            if time.time() > deadline - 8:
                break
            if not check_q3_with_edge(u, v):
                add_edge(u, v)

        save_best()

        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for (u, v) in remaining:
                    if time.time() > deadline - 3:
                        break
                    if not ((adj[u] >> v) & 1) and not check_q3_with_edge(u, v):
                        add_edge(u, v)
                        improved = True
                        save_best()

    # Final local improvement on best
    if best_adj is not None and time.time() < deadline - 3:
        adj = best_adj[:]
        edge_count = sum(bin(a).count('1') for a in adj) // 2

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for (u, v) in all_pairs:
                if time.time() > deadline - 2:
                    break
                if not ((adj[u] >> v) & 1) and not check_q3_with_edge(u, v):
                    add_edge(u, v)
                    improved = True
                    save_best()

    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges
