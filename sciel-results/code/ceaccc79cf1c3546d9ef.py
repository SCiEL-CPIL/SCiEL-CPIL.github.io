"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135  # 15s safety margin before 150s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bfs_distances(adj, n, sources):
        """BFS from multiple sources, return dist[source_index][vertex]."""
        dist = {}
        for s in sources:
            d = [-1] * n
            d[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if d[v] == -1:
                        d[v] = d[u] + 1
                        q.append(v)
            dist[s] = d
        return dist

    def greedy_extend(adj, start, n, used_edges_set):
        """Greedily extend a path from start, avoiding used edges and revisiting vertices.
        Returns (path, set_of_edges_used)."""
        path = [start]
        visited = {start}
        current = start
        edges_used = set()
        
        while True:
            candidates = []
            for nb in adj[current]:
                e = tuple(sorted((current, nb)))
                if nb not in visited and e not in used_edges_set:
                    candidates.append(nb)
            if not candidates:
                break
            
            # Choose neighbor with minimum remaining degree (Warnsdorff-like heuristic)
            best_next = min(candidates, key=lambda nb: len(adj[nb]))
            e = tuple(sorted((current, best_next)))
            edges_used.add(e)
            path.append(best_next)
            visited.add(best_next)
            current = best_next
        
        return path, edges_used

    def path_cover_with_matching(edges, n, time_limit):
        """Path cover using T-join matching to guide greedy construction."""
        if not edges:
            return []
        
        adj = build_adj(edges, n)
        edge_set = set(edges)
        
        # Find odd-degree vertices
        odd_verts = [v for v in range(n) if len(adj[v]) % 2 == 1]
        num_paths_lower_bound = len(odd_verts) // 2
        
        # If no odd vertices, we can do a single cycle -> break into one path
        if not odd_verts:
            # Find Eulerian circuit and break at one point
            # For simplicity, use greedy which should find a long path
            paths = []
            remaining = set(edges)
            while remaining:
                if time.time() > time_limit:
                    for e in remaining:
                        paths.append(list(e))
                    return paths
                active = [v for v in range(n) if adj[v]]
                if not active:
                    break
                # Start from a vertex
                start = active[0]
                path, used = greedy_extend(adj, start, n, set())
                if len(path) < 2:
                    e = next(iter(remaining))
                    paths.append(list(e))
                    remaining.discard(e)
                    continue
                for e in used:
                    remaining.discard(e)
                # Update adj for remaining
                for e in used:
                    u, v = e
                    adj[u].discard(v)
                    adj[v].discard(u)
                paths.append(path)
            return paths
        
        # Compute distances between odd vertices
        odd_set = set(odd_verts)
        dist = bfs_distances(adj, n, odd_verts)
        
        # Build cost matrix for minimum weight perfect matching
        m = len(odd_verts)
        if m % 2 != 0:
            # Should not happen, but handle gracefully
            return greedy_path_cover(edges, n, time_limit)
        
        # Create cost matrix
        cost = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    cost[i][j] = 0
                else:
                    d = dist[odd_verts[i]][odd_verts[j]]
                    if d == -1:
                        cost[i][j] = 10000  # disconnected, shouldn't happen
                    else:
                        cost[i][j] = d
        
        # Find minimum weight perfect matching
        # linear_sum_assignment finds min-cost assignment (bipartite matching)
        # For symmetric matrix, this gives us the matching
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Extract pairs
        pairs = set()
        for r, c in zip(row_ind, col_ind):
            if r != c:
                pair = tuple(sorted((odd_verts[r], odd_verts[c])))
                pairs.add(pair)
        
        # Now construct paths guided by the matching
        # Strategy: For each pair (u, v), try to find a path from u to v
        # that uses remaining edges
        remaining_edges = set(edges)
        paths = []
        adj_remaining = build_adj(edges, n)
        
        # Sort pairs by distance (closest first) to reduce complexity
        sorted_pairs = sorted(pairs, key=lambda p: dist[p[0]][p[1]])
        
        for (u, v) in sorted_pairs:
            if time.time() > time_limit:
                break
            if u not in adj_remaining and v not in adj_remaining:
                continue
            # Try to find a path from u to v using remaining edges
            # Use DFS with backtracking for short paths, greedy for longer
            best_path = None
            best_edges = None
            
            # Try greedy from u
            path_u, edges_u = greedy_extend(adj_remaining, u, n, set())
            if v in path_u:
                # Find v in path and truncate
                idx = path_u.index(v)
                best_path = path_u[:idx+1]
                best_edges = set()
                for i in range(len(best_path) - 1):
                    e = tuple(sorted((best_path[i], best_path[i+1])))
                    best_edges.add(e)
            
            # Try greedy from v
            if best_path is None:
                path_v, edges_v = greedy_extend(adj_remaining, v, n, set())
                if u in path_v:
                    idx = path_v.index(u)
                    best_path = path_v[:idx+1]
                    best_edges = set()
                    for i in range(len(best_path) - 1):
                        e = tuple(sorted((best_path[i], best_path[i+1])))
                        best_edges.add(e)
            
            # If greedy didn't work, try a direct BFS path
            if best_path is None:
                # BFS for shortest path from u to v in remaining graph
                parent = {u: None}
                q = deque([u])
                found = False
                while q:
                    curr = q.popleft()
                    if curr == v:
                        found = True
                        break
                    for nb in adj_remaining[curr]:
                        if nb not in parent:
                            parent[nb] = curr
                            q.append(nb)
                if found:
                    path = []
                    curr = v
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    path.reverse()
                    best_path = path
                    best_edges = set()
                    for i in range(len(best_path) - 1):
                        e = tuple(sorted((best_path[i], best_path[i+1])))
                        best_edges.add(e)
            
            if best_path is None or len(best_path) < 2:
                # Fallback: just connect u and v if edge exists
                e = tuple(sorted((u, v)))
                if e in remaining_edges:
                    best_path = [u, v]
                    best_edges = {e}
                else:
                    continue
            
            # Remove used edges
            for e in best_edges:
                remaining_edges.discard(e)
                u2, v2 = e
                adj_remaining[u2].discard(v2)
                adj_remaining[v2].discard(u2)
            
            paths.append(best_path)
        
        # Handle any remaining edges with greedy
        while remaining_edges:
            if time.time() > time_limit:
                for e in remaining_edges:
                    paths.append(list(e))
                return paths
            
            active = [v for v in range(n) if adj_remaining[v]]
            if not active:
                break
            
            start = active[0]
            path, used = greedy_extend(adj_remaining, start, n, set())
            if len(path) < 2:
                e = next(iter(remaining_edges))
                paths.append(list(e))
                remaining_edges.discard(e)
                continue
            for e in used:
                remaining_edges.discard(e)
                u2, v2 = e
                adj_remaining[u2].discard(v2)
                adj_remaining[v2].discard(u2)
            paths.append(path)
        
        return paths

    def greedy_path_cover(edges, n, time_limit):
        """Pure greedy path cover with Warnsdorff heuristic."""
        if not edges:
            return []
        
        adj = build_adj(edges, n)
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            if time.time() > time_limit:
                for e in remaining_edges:
                    paths.append(list(e))
                return paths
            
            active = [v for v in range(n) if adj[v]]
            if not active:
                break
            
            # Try multiple starts, pick longest path
            best_path = None
            best_edges = None
            best_len = 0
            
            # Prefer low-degree starts
            starts = sorted(active, key=lambda v: len(adj[v]))
            # Take up to 10 candidates
            candidates = starts[:min(10, len(starts))]
            
            for start in candidates:
                path, used = greedy_extend(adj, start, n, set())
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
                    best_edges = used
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining_edges))
                paths.append(list(e))
                remaining_edges.discard(e)
                continue
            
            for e in best_edges:
                remaining_edges.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
            
            paths.append(best_path)
        
        return paths

    def best_path_cover(edges, n, time_limit):
        """Try multiple strategies, return best."""
        if not edges:
            return []
        
        # Strategy 1: Matching-guided
        try:
            paths1 = path_cover_with_matching(edges, n, time_limit)
        except Exception:
            paths1 = greedy_path_cover(edges, n, time_limit)
        
        # Strategy 2: Pure greedy
        paths2 = greedy_path_cover(edges, n, time_limit)
        
        # Return the one with fewer paths
        if len(paths1) <= len(paths2):
            return paths1
        else:
            return paths2

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Fallback: return trivial path cover (each edge as a path)
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        # Allocate time based on remaining budget
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        # Give more time to larger graphs
        time_limit = start_time + min(remaining_time, 40)
        
        paths = best_path_cover(edges, n, time_limit)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
