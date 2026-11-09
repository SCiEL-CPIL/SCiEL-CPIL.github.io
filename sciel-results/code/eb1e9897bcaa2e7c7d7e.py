"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin before 150s

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

    def bfs_shortest_path(adj, n, start, end):
        """BFS to find shortest path from start to end. Returns list of vertices or None."""
        if start == end:
            return [start]
        parent = {start: None}
        q = deque([start])
        while q:
            curr = q.popleft()
            if curr == end:
                # Reconstruct path
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[curr]:
                if nb not in parent:
                    parent[nb] = curr
                    q.append(nb)
        return None

    def greedy_extend(adj, start, n, avoid_edges=None):
        """Greedily extend a path from start, avoiding used edges and revisiting vertices.
        Returns (path, set_of_edges_used)."""
        if avoid_edges is None:
            avoid_edges = set()
        path = [start]
        visited = {start}
        current = start
        edges_used = set()
        
        while True:
            candidates = []
            for nb in adj[current]:
                e = tuple(sorted((current, nb)))
                if nb not in visited and e not in avoid_edges:
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

    def find_eulerian_path(adj, n):
        """Find an Eulerian path in a connected graph using Hierholzer's algorithm.
        Assumes the graph has 0 or 2 odd-degree vertices.
        Returns a list of vertices representing the path."""
        if not any(adj[v] for v in range(n)):
            return []
        
        # Find start vertex (odd degree if exists, else any vertex with edges)
        start = None
        for v in range(n):
            if len(adj[v]) % 2 == 1:
                start = v
                break
        if start is None:
            for v in range(n):
                if adj[v]:
                    start = v
                    break
        if start is None:
            return []
        
        # Hierholzer's algorithm
        stack = [start]
        path = []
        visited_edges = set()
        
        while stack:
            v = stack[-1]
            next_v = None
            for nb in list(adj[v]):
                e = tuple(sorted((v, nb)))
                if e not in visited_edges:
                    next_v = nb
                    break
            
            if next_v is None:
                path.append(stack.pop())
            else:
                e = tuple(sorted((v, next_v)))
                visited_edges.add(e)
                stack.append(next_v)
        
        path.reverse()
        return path

    def improve_path_cover(paths, edges, n, time_limit):
        """Try to merge paths by finding connecting paths through unused edges."""
        if len(paths) <= 1:
            return paths
        
        # Build set of edges used in paths
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                used_edges.add(e)
        
        remaining_edges = set(edges) - used_edges
        if not remaining_edges:
            return paths
        
        adj_remaining = build_adj(list(remaining_edges), n)
        improved = False
        paths_copy = [list(p) for p in paths]
        
        # Try to merge pairs of paths
        for i in range(len(paths_copy)):
            for j in range(i + 1, len(paths_copy)):
                if time.time() > time_limit:
                    break
                
                path_i = paths_copy[i]
                path_j = paths_copy[j]
                
                # Try all 4 combinations: i_end to j_start, i_end to j_end, i_start to j_start, i_start to j_end
                combinations = [
                    (path_i[-1], path_j[0], path_i, path_j, False, False),
                    (path_i[-1], path_j[-1], path_i, path_j[::-1], False, True),
                    (path_i[0], path_j[0], path_i[::-1], path_j, True, False),
                    (path_i[0], path_j[-1], path_i[::-1], path_j[::-1], True, True),
                ]
                
                for start_v, end_v, p1, p2, rev1, rev2 in combinations:
                    # Check if there's a path from start_v to end_v using only remaining edges
                    # and not visiting any vertices already in p1 or p2 (except endpoints)
                    vertices_in_paths = set(p1) | set(p2)
                    
                    # BFS with constraint: can't visit vertices in paths except start and end
                    parent = {start_v: None}
                    q = deque([start_v])
                    found = False
                    while q:
                        curr = q.popleft()
                        if curr == end_v:
                            found = True
                            break
                        for nb in adj_remaining[curr]:
                            if nb not in parent:
                                # Can only use intermediate vertices not in any path
                                if nb in vertices_in_paths and nb != start_v and nb != end_v:
                                    continue
                                parent[nb] = curr
                                q.append(nb)
                    
                    if found:
                        # Reconstruct connecting path
                        conn_path = []
                        node = end_v
                        while node is not None:
                            conn_path.append(node)
                            node = parent[node]
                        conn_path.reverse()
                        
                        # Check that connecting path is simple and uses only remaining edges
                        conn_edges = set()
                        valid = True
                        for k in range(len(conn_path) - 1):
                            e = tuple(sorted((conn_path[k], conn_path[k+1])))
                            if e not in remaining_edges:
                                valid = False
                                break
                            conn_edges.add(e)
                        
                        if valid and len(conn_path) >= 2:
                            # Merge the paths
                            if rev1:
                                p1 = p1[::-1]
                            if rev2:
                                p2 = p2[::-1]
                            
                            # Remove endpoints that are shared
                            merged = p1 + conn_path[1:] + p2[1:]
                            
                            # Verify merged path is simple
                            if len(set(merged)) == len(merged):
                                # Update remaining edges
                                remaining_edges -= conn_edges
                                for e in conn_edges:
                                    u, v = e
                                    adj_remaining[u].discard(v)
                                    adj_remaining[v].discard(u)
                                
                                # Replace paths i and j with merged path
                                paths_copy[i] = merged
                                paths_copy.pop(j)
                                improved = True
                                break
                
                if time.time() > time_limit:
                    break
            
            if time.time() > time_limit:
                break
        
        return paths_copy

    def path_cover_with_matching(edges, n, time_limit):
        """Path cover using T-join matching to guide construction."""
        if not edges:
            return []
        
        adj = build_adj(edges, n)
        
        # Find odd-degree vertices
        odd_verts = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        # If no odd vertices, we can do a single Eulerian circuit
        if not odd_verts:
            euler_path = find_eulerian_path(adj, n)
            if len(euler_path) > 1:
                if euler_path[0] == euler_path[-1]:
                    return [euler_path[:-1]]
                else:
                    return [euler_path]
            # Fallback
            return greedy_path_cover(edges, n, time_limit)
        
        m = len(odd_verts)
        if m % 2 != 0:
            return greedy_path_cover(edges, n, time_limit)
        
        # Compute distances between odd vertices
        dist = {}
        for s in odd_verts:
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
        
        # Build cost matrix
        cost = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    cost[i][j] = 0
                else:
                    d = dist[odd_verts[i]][odd_verts[j]]
                    cost[i][j] = d if d != -1 else 10000
        
        # Find minimum weight perfect matching
        try:
            cost_sparse = csr_matrix(cost)
            matching = min_weight_full_bipartite_matching(cost_sparse)
            pairs = set()
            for r, c in matching:
                if r != c:
                    pair = tuple(sorted((odd_verts[r], odd_verts[c])))
                    pairs.add(pair)
        except Exception:
            row_ind, col_ind = linear_sum_assignment(cost)
            pairs = set()
            for r, c in zip(row_ind, col_ind):
                if r != c:
                    pair = tuple(sorted((odd_verts[r], odd_verts[c])))
                    pairs.add(pair)
        
        # Construct paths guided by matching
        remaining_edges = set(edges)
        paths = []
        adj_remaining = build_adj(edges, n)
        
        sorted_pairs = sorted(pairs, key=lambda p: dist[p[0]][p[1]] if dist[p[0]][p[1]] != -1 else 10000)
        
        for (u, v) in sorted_pairs:
            if time.time() > time_limit:
                break
            if not adj_remaining[u] and not adj_remaining[v]:
                continue
            
            # Find shortest path from u to v using remaining edges
            best_path = bfs_shortest_path(adj_remaining, n, u, v)
            
            if best_path is None or len(best_path) < 2:
                e = tuple(sorted((u, v)))
                if e in remaining_edges:
                    best_path = [u, v]
                else:
                    continue
            
            best_edges = set()
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                best_edges.add(e)
            
            for e in best_edges:
                remaining_edges.discard(e)
                u2, v2 = e
                adj_remaining[u2].discard(v2)
                adj_remaining[v2].discard(u2)
            
            paths.append(best_path)
        
        # Handle remaining edges with greedy
        while remaining_edges:
            if time.time() > time_limit:
                for e in remaining_edges:
                    paths.append(list(e))
                return paths
            
            active = [v for v in range(n) if adj_remaining[v]]
            if not active:
                break
            
            # Try multiple starts, pick longest path
            best_path = None
            best_edges = None
            best_len = 0
            
            starts = sorted(active, key=lambda v: len(adj_remaining[v]))
            candidates = starts[:min(10, len(starts))]
            
            for start in candidates:
                path, used = greedy_extend(adj_remaining, start, n, set())
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
                adj_remaining[u].discard(v)
                adj_remaining[v].discard(u)
            
            paths.append(best_path)
        
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
            
            best_path = None
            best_edges = None
            best_len = 0
            
            starts = sorted(active, key=lambda v: len(adj[v]))
            candidates = starts[:min(15, len(starts))]
            
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
        """Try multiple strategies, return best, then try to improve."""
        if not edges:
            return []
        
        # Strategy 1: Matching-guided
        try:
            paths1 = path_cover_with_matching(edges, n, time_limit)
        except Exception:
            paths1 = greedy_path_cover(edges, n, time_limit)
        
        # Strategy 2: Pure greedy
        paths2 = greedy_path_cover(edges, n, time_limit)
        
        # Pick the better one
        best = paths1 if len(paths1) <= len(paths2) else paths2
        
        # Try to improve by merging paths
        if len(best) > 1 and time.time() < time_limit:
            improved = improve_path_cover(best, edges, n, time_limit)
            if len(improved) < len(best):
                best = improved
        
        return best

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        time_limit = start_time + min(remaining_time, 45)
        
        paths = best_path_cover(edges, n, time_limit)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
