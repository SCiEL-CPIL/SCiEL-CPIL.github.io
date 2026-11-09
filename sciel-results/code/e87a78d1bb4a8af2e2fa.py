import time
import sys
from random import Random
from collections import deque
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _bfs_distances(adj, n, sources):
        """Multi-source BFS: return dist list mapping vertex -> dist."""
        dist = [-1] * n
        q = deque()
        for s in sources:
            dist[s] = 0
            q.append(s)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _min_weight_perfect_matching(odd, dist_matrix):
        """
        Find minimum weight perfect matching for a set of 2k vertices.
        Uses DP over subsets.
        odd: list of vertex indices
        dist_matrix: 2D list or array where dist_matrix[i][j] is distance between odd[i] and odd[j]
        Returns list of pairs (indices into odd list)
        """
        k = len(odd)
        if k == 0:
            return []
        if k % 2 != 0:
            # Should not happen for connected graphs, but safety
            return []
        
        # DP state: bitmask of used vertices
        # dp[mask] = min cost to match vertices in mask
        # mask is k bits long
        size = 1 << k
        INF = float('inf')
        dp = [INF] * size
        dp[0] = 0
        parent = [None] * size # To reconstruct the matching
        
        # Iterate over masks
        for mask in range(size):
            if dp[mask] == INF:
                continue
            # Find the first unset bit
            for i in range(k):
                if not (mask & (1 << i)):
                    # Try to match i with j > i
                    for j in range(i + 1, k):
                        if not (mask & (1 << j)):
                            new_mask = mask | (1 << i) | (1 << j)
                            cost = dp[mask] + dist_matrix[i][j]
                            if cost < dp[new_mask]:
                                dp[new_mask] = cost
                                parent[new_mask] = (mask, i, j)
                    break # Only need to process the first free vertex for this mask
        
        if dp[size - 1] == INF:
            return []
            
        # Reconstruct
        pairs = []
        mask = size - 1
        while mask != 0:
            if parent[mask] is None:
                break
            prev_mask, i, j = parent[mask]
            pairs.append((i, j))
            mask = prev_mask
        
        # Convert indices in odd list to actual vertex ids
        vertex_pairs = [(odd[i], odd[j]) for i, j in pairs]
        return vertex_pairs

    def _eulerian_circuit_break(edges, n, pairs):
        """
        Build Eulerian circuit on edges + dummy pairs, break at dummy edges.
        Returns list of paths (lists of vertices).
        """
        if not edges and not pairs:
            return []
        
        adj = {i: [] for i in range(n)}
        edge_list = []
        eid = 0
        for (u, v) in edges:
            edge_list.append((u, v, eid, False))
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            eid += 1
        for (u, v) in pairs:
            edge_list.append((u, v, eid, True))
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            eid += 1
        
        if eid == 0:
            return []
        
        # Hierholzer's algorithm
        used = [False] * eid
        # Find start vertex with non-zero degree
        sv = None
        for v in range(n):
            if adj[v]:
                sv = v
                break
        if sv is None:
            return []
        
        stack = [sv]
        circuit = []
        while stack:
            v = stack[-1]
            advanced = False
            # To ensure we don't get stuck, we can use a pointer array for adjacency
            # But simple loop is fine for small graphs
            for idx in range(len(adj[v])):
                nb, e = adj[v][idx]
                if not used[e]:
                    used[e] = True
                    stack.append(nb)
                    advanced = True
                    break
            if not advanced:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Build set of dummy edges for cutting
        # Note: Multiple dummy edges between same nodes are possible, but rare.
        # We identify dummy edges by their (u,v) pair.
        # However, if there are multiple edges between u,v, we need to be careful.
        # For random graphs, multi-edges are unlikely.
        dummy_set = set()
        for (u, v, e, is_d) in edge_list:
            if is_d:
                dummy_set.add(frozenset((u, v)))
        
        # Walk circuit, cutting after each dummy edge
        paths = []
        cur = [circuit[0]]
        for i in range(len(circuit) - 1):
            a, b = circuit[i], circuit[i + 1]
            cur.append(b)
            if frozenset((a, b)) in dummy_set:
                if len(cur) >= 2:
                    paths.append(cur)
                cur = [b]
        if len(cur) >= 2:
            paths.append(cur)
        
        return paths

    def _is_valid(paths, edges, n):
        """Check: simple paths, edges exist, edge-disjoint, cover all edges."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            seen = set()
            for v in p:
                if v in seen:
                    return False
                seen.add(v)
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                if e not in edge_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == edge_set

    def _greedy_path_cover(edges, n):
        """Greedy maximal simple path extraction - always produces valid cover."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(tuple(sorted(e)) for e in edges)
        new_paths = []
        while remaining:
            # Build subgraph from remaining edges
            sub = {i: set() for i in range(n)}
            for (u, v) in remaining:
                sub[u].add(v)
                sub[v].add(u)
            active = [v for v in range(n) if sub[v]]
            if not active:
                break
            # Start from vertex with max degree
            start = max(active, key=lambda v: len(sub[v]))
            path = [start]
            visited = {start}
            cur = start
            while True:
                cands = [nb for nb in sub[cur] if nb not in visited]
                if not cands:
                    break
                # Pick neighbor with smallest remaining degree (Wielun heuristic)
                nxt = min(cands, key=lambda nb: len(sub[nb]))
                path.append(nxt)
                visited.add(nxt)
                cur = nxt
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                remaining.discard(e)
            new_paths.append(path)
        return new_paths

    def path_cover(edges, n):
        if not edges:
            return []
        
        degree = {i: 0 for i in range(n)}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        odd = [v for v in range(n) if degree[v] % 2 == 1]
        
        if not odd:
            # Eulerian graph: one circuit
            res = _eulerian_circuit_break(edges, n, [])
            if res and _is_valid(res, edges, n):
                return res
            # If circuit is not simple (should be a cycle, so it is simple unless n<3 or multi-edge issues)
            # Actually, an Eulerian circuit in a simple graph is a cycle, which is simple.
            # But if the graph has self-loops or multi-edges, it might not be.
            # The instance generator creates simple graphs.
            return _greedy_path_cover(edges, n)
        
        # Compute distances between all pairs of odd vertices
        k = len(odd)
        # Map odd vertex index to its position in odd list
        # We need dist_matrix[i][j] = dist(odd[i], odd[j])
        dist_matrix = [[0] * k for _ in range(k)]
        
        # Build adjacency for BFS
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Run BFS from each odd vertex
        for i, s in enumerate(odd):
            dist = _bfs_distances(adj, n, [s])
            for j, t in enumerate(odd):
                if i != j:
                    dist_matrix[i][j] = dist[t]
        
        # Find min weight perfect matching
        pairs = _min_weight_perfect_matching(odd, dist_matrix)
        
        # Generate paths
        res = _eulerian_circuit_break(edges, n, pairs)
        
        if res and _is_valid(res, edges, n):
            return res
        
        # Fallback if optimal matching leads to non-simple paths (rare)
        # Try a few random matchings? Or just greedy.
        # Given the small size, greedy is a safe fallback.
        return _greedy_path_cover(edges, n)

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Emergency: return edge-by-edge (valid but many paths)
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        edges = instance(*spec)
        n = spec[0]
        paths = path_cover(edges, n)
        results.append(paths)
    return results

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print("Number of paths:", [len(r) for r in res])
