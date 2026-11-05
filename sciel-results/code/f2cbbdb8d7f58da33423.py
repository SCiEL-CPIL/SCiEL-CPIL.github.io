"""Path decomposition via odd-vertex pairing (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bfs_path(adj, src, dst, used_edges):
        """Find a simple path from src to dst using only unused edges."""
        if src == dst:
            return [src]
        parent = {src: None}
        prev_edge = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    if parent[cur] is None:
                        break
                    cur = parent[cur]
                path.reverse()
                return path
            for v in adj[u]:
                if v in parent:
                    continue
                edge = (min(u, v), max(u, v))
                if edge in used_edges:
                    continue
                parent[v] = u
                prev_edge[v] = (u, v)
                queue.append(v)
        return None

    def decompose_even_graph(adj, edges_set, n):
        """Decompose an all-even-degree graph into paths (via cycles)."""
        paths = []
        remaining = set(edges_set)
        
        while remaining:
            if time.time() > DEADLINE:
                break
            # Find a cycle using DFS
            start_edge = next(iter(remaining))
            u0, v0 = start_edge
            
            # Try to find a cycle starting from u0
            cycle = None
            visited_edges = set()
            stack = [(v0, [u0, v0])]  # (current_node, path_so_far)
            
            while stack and cycle is None:
                cur, path = stack.pop()
                for nxt in adj[cur]:
                    e = (min(cur, nxt), max(cur, nxt))
                    if e in remaining and e not in visited_edges:
                        visited_edges.add(e)
                        if nxt == u0 and len(path) >= 2:
                            cycle = path
                            break
                        stack.append((nxt, path + [nxt]))
            
            if cycle is None:
                # Fallback: take single edge
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.discard(e)
                continue
            
            # Remove cycle edges from remaining
            for i in range(len(cycle) - 1):
                a, b = cycle[i], cycle[i + 1]
                e = (min(a, b), max(a, b))
                remaining.discard(e)
            
            # Convert cycle to path by dropping last vertex
            path = cycle[:-1]
            paths.append(path)
        
        return paths

    def decompose_graph(edges, n):
        """Decompose graph into few paths using odd-vertex pairing."""
        adj = build_adjacency(edges, n)
        edges_set = set(edges)
        
        # Find odd-degree vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        paths = []
        used_edges = set()
        
        # Pair odd vertices greedily
        paired = [False] * len(odd_vertices)
        for i in range(len(odd_vertices)):
            if time.time() > DEADLINE:
                break
            if paired[i]:
                continue
            a = odd_vertices[i]
            paired[i] = True
            
            # Find nearest unpaired odd vertex
            best_j = -1
            for j in range(i + 1, len(odd_vertices)):
                if not paired[j]:
                    best_j = j
                    break
            
            if best_j == -1:
                continue
            
            b = odd_vertices[best_j]
            paired[best_j] = True
            
            # Find path between a and b
            path = bfs_path(adj, a, b, used_edges)
            if path is not None and len(path) >= 2:
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    e = (min(u, v), max(u, v))
                    if e in edges_set:
                        used_edges.add(e)
                        edges_set.discard(e)
                paths.append(path)
            else:
                # Path not found, these vertices remain odd
                pass
        
        # Decompose remaining even-degree graph
        if edges_set and time.time() < DEADLINE:
            remaining_paths = decompose_even_graph(adj, edges_set, n)
            paths.extend(remaining_paths)
        
        # Fallback for any remaining edges
        if edges_set:
            for e in edges_set:
                paths.append([e[0], e[1]])
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        paths = decompose_graph(edges, n)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
