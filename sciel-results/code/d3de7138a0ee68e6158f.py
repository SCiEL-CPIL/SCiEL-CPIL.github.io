"""Rewritten program for Gallai's conjecture path decomposition."""
import time
from collections import defaultdict, deque
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)
    
    def build_graph(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def degree_map(adj):
        return {u: len(v) for u, v in adj.items()}
    
    def find_eulerian_path(adj, start=None):
        """Find an Eulerian path in a graph that has one."""
        # Check if Eulerian path exists
        deg = degree_map(adj)
        odd_vertices = [u for u, d in deg.items() if d % 2 == 1]
        
        if len(odd_vertices) > 2:
            return None
        
        if len(odd_vertices) == 2:
            start = odd_vertices[0]
        elif start is None:
            # Find a vertex with non-zero degree
            start = next((u for u, d in deg.items() if d > 0), None)
            if start is None:
                return []
        
        # Hierholzer's algorithm
        adj_copy = defaultdict(set)
        for u, vs in adj.items():
            adj_copy[u] = set(vs)
        
        stack = [start]
        path = []
        
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                w = next(iter(adj_copy[v]))
                adj_copy[v].discard(w)
                adj_copy[w].discard(v)
                stack.append(w)
            else:
                path.append(stack.pop())
        
        path.reverse()
        return path
    
    def greedy_path_decomposition(edges, adj, max_paths=None):
        """Greedy approach: repeatedly find longest possible paths."""
        remaining_edges = set(edges)
        paths = []
        
        # Build adjacency from remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        while remaining_edges:
            # Find a vertex with odd degree in remaining graph
            odd_vertices = [u for u, vs in rem_adj.items() if len(vs) % 2 == 1]
            
            if len(odd_vertices) <= 2:
                # Try to find an Eulerian path
                start = odd_vertices[0] if odd_vertices else None
                if start is None:
                    start = next(iter(rem_adj.keys()), None)
                
                # Use Hierholzer on the remaining graph
                adj_copy = defaultdict(set)
                for u, vs in rem_adj.items():
                    adj_copy[u] = set(vs)
                
                stack = [start] if start else []
                if not stack and rem_adj:
                    stack = [next(iter(rem_adj.keys()))]
                
                path = []
                while stack:
                    v = stack[-1]
                    if adj_copy[v]:
                        w = next(iter(adj_copy[v]))
                        adj_copy[v].discard(w)
                        adj_copy[w].discard(v)
                        stack.append(w)
                    else:
                        path.append(stack.pop())
                
                path.reverse()
                
                # Convert path to edge list and remove those edges
                if len(path) >= 2:
                    path_edges = set()
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        e = (min(u, v), max(u, v))
                        path_edges.add(e)
                    
                    # Verify all edges are in remaining
                    if path_edges.issubset(remaining_edges):
                        paths.append(path)
                        remaining_edges -= path_edges
                        # Update rem_adj
                        for u, v in path_edges:
                            if v in rem_adj[u]:
                                rem_adj[u].discard(v)
                            if u in rem_adj[v]:
                                rem_adj[v].discard(u)
                            if not rem_adj[u]:
                                del rem_adj[u]
                            if not rem_adj[v]:
                                del rem_adj[v]
                        continue
                
                # If Eulerian path failed, fall back to single edge
                e = next(iter(remaining_edges))
                paths.append([e[0], e[1]])
                remaining_edges.remove(e)
                u, v = e
                if v in rem_adj[u]:
                    rem_adj[u].discard(v)
                if u in rem_adj[v]:
                    rem_adj[v].discard(u)
                if not rem_adj[u]:
                    del rem_adj[u]
                if not rem_adj[v]:
                    del rem_adj[v]
            else:
                # More than 2 odd vertices: pick a random edge and remove it
                # or find a short path
                # Try to find a path between two odd vertices
                odd_set = set(odd_vertices)
                found = False
                for i in range(len(odd_vertices)):
                    for j in range(i + 1, len(odd_vertices)):
                        s, t = odd_vertices[i], odd_vertices[j]
                        # BFS to find path from s to t
                        visited = {s}
                        queue = deque([(s, [s])])
                        while queue:
                            node, path = queue.popleft()
                            if node == t:
                                # Found a path
                                path_edges = set()
                                for k in range(len(path) - 1):
                                    u, v = path[k], path[k + 1]
                                    e = (min(u, v), max(u, v))
                                    if e in remaining_edges:
                                        path_edges.add(e)
                                    else:
                                        break
                                else:
                                    if path_edges.issubset(remaining_edges) and len(path_edges) > 0:
                                        paths.append(path)
                                        remaining_edges -= path_edges
                                        for u, v in path_edges:
                                            if v in rem_adj[u]:
                                                rem_adj[u].discard(v)
                                            if u in rem_adj[v]:
                                                rem_adj[v].discard(u)
                                            if not rem_adj[u]:
                                                del rem_adj[u]
                                            if not rem_adj[v]:
                                                del rem_adj[v]
                                        found = True
                                        break
                            for neighbor in list(rem_adj.get(node, [])):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append((neighbor, path + [neighbor]))
                        if found:
                            break
                    if found:
                        break
                
                if not found:
                    # Just take one edge
                    e = next(iter(remaining_edges))
                    paths.append([e[0], e[1]])
                    remaining_edges.remove(e)
                    u, v = e
                    if v in rem_adj[u]:
                        rem_adj[u].discard(v)
                    if u in rem_adj[v]:
                        rem_adj[v].discard(u)
                    if not rem_adj[u]:
                        del rem_adj[u]
                    if not rem_adj[v]:
                        del rem_adj[v]
        
        return paths
    
    def improved_decomposition(edges, adj):
        """
        Better approach: 
        1. Identify vertices with odd degree
        2. Pair them up and find paths between pairs
        3. Remove those paths, then the remaining graph has all even degrees
        4. Decompose the remaining graph into Eulerian cycles (convert to paths)
        """
        deg = degree_map(adj)
        odd_vertices = [u for u, d in deg.items() if d % 2 == 1]
        
        # Sort odd vertices for deterministic pairing
        odd_vertices.sort()
        
        remaining_edges = set(edges)
        paths = []
        
        # Pair odd vertices and find paths between them
        # Use greedy pairing: pair nearest odd vertices
        for i in range(0, len(odd_vertices), 2):
            if i + 1 < len(odd_vertices):
                s, t = odd_vertices[i], odd_vertices[i + 1]
                # Find shortest path between s and t in remaining graph
                rem_adj = defaultdict(set)
                for u, v in remaining_edges:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
                
                # BFS
                visited = {s}
                parent = {s: None}
                queue = deque([s])
                while queue:
                    node = queue.popleft()
                    if node == t:
                        break
                    for neighbor in rem_adj.get(node, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            parent[neighbor] = node
                            queue.append(neighbor)
                
                if t in parent:
                    # Reconstruct path
                    path = []
                    cur = t
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    
                    # Remove these edges
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k + 1]
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                    
                    if len(path) >= 2:
                        paths.append(path)
        
        # Now remaining graph should have all even degrees
        # Decompose into Eulerian cycles
        while remaining_edges:
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Find a vertex with non-zero degree
            start = next((u for u, vs in rem_adj.items() if vs), None)
            if start is None:
                break
            
            # Hierholzer
            adj_copy = defaultdict(set)
            for u, vs in rem_adj.items():
                adj_copy[u] = set(vs)
            
            stack = [start]
            cycle = []
            while stack:
                v = stack[-1]
                if adj_copy[v]:
                    w = next(iter(adj_copy[v]))
                    adj_copy[v].discard(w)
                    adj_copy[w].discard(v)
                    stack.append(w)
                else:
                    cycle.append(stack.pop())
            
            cycle.reverse()
            
            # Remove cycle edges
            for k in range(len(cycle) - 1):
                u, v = cycle[k], cycle[k + 1]
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    remaining_edges.remove(e)
            # The last edge closes the cycle
            if len(cycle) > 1:
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    remaining_edges.remove(e)
            
            # Convert cycle to path (break at any point)
            if len(cycle) > 1:
                paths.append(cycle)
        
        return paths
    
    # Main logic
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > DEADLINE - 10:
            # Fallback: single edges
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        adj = build_graph(edges)
        
        # Try improved decomposition
        try:
            paths = improved_decomposition(edges, adj)
            
            # Verify
            used_edges = set()
            valid = True
            for path in paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    e = (min(u, v), max(u, v))
                    if e not in adj[u] or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
            
            if not valid or used_edges != set(edges):
                # Fallback to greedy
                paths = greedy_path_decomposition(edges, adj)
            else:
                results.append(paths)
                continue
        except Exception:
            paths = greedy_path_decomposition(edges, adj)
        
        # Verify greedy result
        used_edges = set()
        valid = True
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                if e not in adj[u] or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            if not valid:
                break
        
        if not valid or used_edges != set(edges):
            # Last resort: single edges
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
    
    return results
