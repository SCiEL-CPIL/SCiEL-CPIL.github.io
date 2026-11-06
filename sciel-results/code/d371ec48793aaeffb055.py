import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 125  # seconds; leave safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def deadline_exceeded():
        return elapsed() > DEADLINE

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_triangle(n, adj):
        for u in range(n):
            au = adj[u]
            v_mask = au & ~((1 << (u + 1)) - 1)
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                common = au & adj[v]
                if common:
                    return True
        return False

    def max_independent_set_size(n, adj, limit):
        best = [0]

        def dfs(candidates, size):
            if deadline_exceeded():
                return
            if not candidates:
                if size > best[0]:
                    best[0] = size
                return

            if size + candidates.bit_count() <= best[0]:
                return

            if best[0] >= limit:
                return

            # Pick a vertex to branch on.
            # Heuristic: pick vertex with highest degree in the induced subgraph?
            # Or just the first one.
            # Let's pick the first available vertex for speed.
            v = (candidates & -candidates).bit_length() - 1
            candidates_without_v = candidates & ~(1 << v)

            # Branch 1: Include v
            new_cands_incl = candidates_without_v & ~adj[v]
            dfs(new_cands_incl, size + 1)

            # Branch 2: Exclude v
            dfs(candidates_without_v, size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def is_valid(n, edges):
        if deadline_exceeded():
            return False
        adj = build_adj(n, edges)
        if has_triangle(n, adj):
            return False
        alpha = max_independent_set_size(n, adj, 10)
        if alpha >= 10:
            return False
        return True

    def andrasfai_9():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff % 3 == 1:
                    edges.append([i, j])
        return n, edges

    # Try to find a larger graph using random search + local improvement
    best_n, best_edges = andrasfai_9()
    if is_valid(best_n, best_edges):
        pass  # Baseline established

    # Target n values: 40, 39
    for target_n in [40, 39]:
        if deadline_exceeded():
            break

        for restart in range(100):
            if deadline_exceeded():
                break

            edges = []
            adj = [0] * target_n

            # Add random edges
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < 0.3:
                        common = adj[u] & adj[v]
                        if not common:
                            edges.append([u, v])
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

            # Now try to improve: add edges to reduce alpha
            improved = True
            while improved and not deadline_exceeded():
                improved = False
                for _ in range(100):
                    u = random.randint(0, target_n - 1)
                    v = random.randint(0, target_n - 1)
                    if u == v or (adj[u] >> v) & 1:
                        continue

                    common = adj[u] & adj[v]
                    if not common:
                        edges.append([u, v])
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break

                if random.random() < 0.1:
                    if is_valid(target_n, edges):
                        best_n = target_n
                        best_edges = edges
                        return {'n': best_n, 'edges': best_edges}

            if is_valid(target_n, edges):
                best_n = target_n
                best_edges = edges
                return {'n': best_n, 'edges': best_edges}

    return {'n': best_n, 'edges': best_edges}
