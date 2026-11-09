import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)

    def build_adj_deg(edge_set):
        adj = [0] * n  # bitmask adjacency
        deg = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def has_violation(adj, deg):
        """Return True if any same-degree pair has a 3-edge path between them."""
        deg_map = defaultdict(list)
        for i in range(n):
            deg_map[deg[i]].append(i)
        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s, t = verts[i], verts[j]
                    # Check if there's a path s-u-v-t
                    for u in range(n):
                        if u == s or u == t:
                            continue
                        if not (adj[s] & (1 << u)):
                            continue
                        for v in range(n):
                            if v == s or v == t or v == u:
                                continue
                            if not (adj[u] & (1 << v)):
                                continue
                            if adj[v] & (1 << t):
                                return True
        return False

    def is_valid(edge_set):
        adj, deg = build_adj_deg(edge_set)
        return not has_violation(adj, deg)

    # Base: complete bipartite K_{7,8} - always valid
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = base_edges[:]
    best_count = len(base_edges)

    # Greedy add with multiple random orderings
    for attempt in range(300):
        if time.time() > deadline - 10:
            break
        current = set(base_edges)
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        for edge in shuffled:
            if edge in current:
                continue
            test = current | {edge}
            if is_valid(test):
                current = test
        if len(current) > best_count:
            best_count = len(current)
            best_edges = list(current)

    # Local search: try removing one edge and adding multiple others
    for ls_attempt in range(100):
        if time.time() > deadline - 5:
            break
        current = set(best_edges)
        # Try removing each edge and then greedily adding
        removed = []
        for edge in list(current):
            if time.time() > deadline - 3:
                break
            candidate = current - {edge}
            # Try to add edges back
            shuffled = all_edges[:]
            random.shuffle(shuffled)
            for e in shuffled:
                if e in candidate:
                    continue
                test = candidate | {e}
                if is_valid(test):
                    candidate = test
            if len(candidate) > len(current):
                current = candidate
                break
        if len(current) > best_count:
            best_count = len(current)
            best_edges = list(current)

    out = [[u, v] for u, v in best_edges]
    return out
