"""Decompose fixed random graphs into minimum number of cycles and edges."""
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # We'll work with a mutable copy of edges
        remaining = set(edges)
        pieces = []
        
        # Strategy: repeatedly find the longest cycle in the remaining graph,
        # remove it, and continue. This is a greedy heuristic but often good.
        # We use a randomized DFS to find long cycles.
        
        def find_longest_cycle(adj, remaining):
            """Find a long cycle using randomized DFS with backtracking."""
            if not remaining:
                return []
            
            best_cycle = []
            # Try multiple starting vertices
            vertices = sorted(adj.keys())
            random.shuffle(vertices)
            
            for start in vertices:
                if len(adj[start]) < 2:
                    continue
                
                # DFS to find a cycle starting and ending at start
                stack = [(start, [start])]
                visited_in_path = {start}
                found = False
                
                # Use iterative DFS with path tracking
                # This is a simplified approach: try to extend paths
                best_path = None
                
                # Try recursive-like DFS with depth limit
                def dfs(current, path, visited_set):
                    nonlocal best_path
                    if len(path) > 3:  # minimum cycle length
                        # Check if we can close the cycle
                        if current != start and start in adj[current]:
                            if best_path is None or len(path) > len(best_path):
                                best_path = path[:]
                            return
                        # Prune: if path is already longer than best, we can stop extending
                        # but we want longest, so continue
                        if best_path and len(path) > len(best_path) + 1:
                            return
                    
                    for neighbor in sorted(adj[current]):
                        if neighbor == start:
                            if len(path) >= 3:
                                if best_path is None or len(path) > len(best_path):
                                    best_path = path[:]
                                continue
                        if neighbor not in visited_set:
                            visited_set.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, path, visited_set)
                            path.pop()
                            visited_set.remove(neighbor)
                
                dfs(start, [start], {start})
                
                if best_path and (not best_cycle or len(best_path) > len(best_cycle)):
                    best_cycle = best_path
                    # Early termination if we found a very long cycle
                    if len(best_cycle) >= 10:
                        break
            
            return best_cycle
        
        # Greedy: repeatedly extract longest cycle
        while remaining:
            if time.time() > deadline - 5:
                break
            
            # Get current adjacency
            cur_adj = defaultdict(set)
            for u, v in remaining:
                cur_adj[u].add(v)
                cur_adj[v].add(u)
            
            cycle = find_longest_cycle(cur_adj, remaining)
            
            if not cycle or len(cycle) < 3:
                # No more cycles; add all remaining edges as single edges
                for e in remaining:
                    pieces.append(list(e))
                remaining = set()
                break
            
            # Remove the cycle's edges from remaining
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                cycle_edges.add(edge)
            
            # Verify all edges are in remaining
            valid = all(e in remaining for e in cycle_edges)
            if valid:
                pieces.append(cycle)
                remaining -= cycle_edges
            else:
                # Fallback: just add one edge
                e = next(iter(remaining))
                pieces.append(list(e))
                remaining.discard(e)
        
        # Add any remaining edges
        for e in remaining:
            pieces.append(list(e))
        
        return pieces

    def decompose_improved(edges, n):
        """Improved decomposition using better cycle finding."""
        if not edges:
            return []
        
        remaining = set(edges)
        pieces = []
        
        # Build adjacency
        def get_adj(edge_set):
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        def find_cycle(adj, start=None):
            """Find a cycle starting from a given vertex using DFS."""
            if not adj:
                return []
            
            if start is None:
                # Pick a vertex with highest degree
                start = max(adj.keys(), key=lambda x: len(adj[x]))
            
            # DFS to find a cycle
            visited = set()
            parent = {}
            stack = [start]
            visited.add(start)
            
            while stack:
                node = stack[-1]
                neighbors = list(adj[node])
                found_next = False
                for nb in neighbors:
                    if nb == start and len(visited) >= 3:
                        # Found a cycle
                        cycle = [start]
                        cur = node
                        while cur != start:
                            cycle.append(cur)
                            cur = parent.get(cur, start)
                        cycle.reverse()
                        return cycle
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = node
                        stack.append(nb)
                        found_next = True
                        break
                if not found_next:
                    stack.pop()
            
            return []
        
        def find_long_cycle(adj):
            """Find a long cycle by trying multiple starts."""
            best = []
            vertices = sorted(adj.keys(), key=lambda x: -len(adj[x]))
            
            for start in vertices[:10]:  # Try top 10 degree vertices
                if len(adj[start]) < 2:
                    continue
                
                # Use a different approach: try to build a long path and close it
                # This is a heuristic
                path = [start]
                visited = {start}
                current = start
                
                # Greedy: always go to unvisited neighbor with highest degree
                while True:
                    neighbors = [nb for nb in adj[current] if nb not in visited]
                    if not neighbors:
                        # Try to close cycle
                        if start in adj[current] and len(path) >= 3:
                            if len(path) > len(best):
                                best = path[:]
                        break
                    # Pick neighbor with highest degree (not start unless closing)
                    if start in neighbors and len(path) >= 3:
                        # Close the cycle
                        if len(path) > len(best):
                            best = path[:]
                        break
                    # Pick the neighbor with highest degree
                    next_nb = max(neighbors, key=lambda x: len(adj[x]))
                    visited.add(next_nb)
                    path.append(next_nb)
                    current = next_nb
                
                if len(best) >= 8:
                    break
            
            return best
        
        while remaining:
            if time.time() > deadline - 3:
                break
            
            adj = get_adj(remaining)
            
            if not adj:
                break
            
            cycle = find_long_cycle(adj)
            
            if not cycle or len(cycle) < 3:
                # No cycle found; add edges individually
                for e in remaining:
                    pieces.append(list(e))
                remaining = set()
                break
            
            # Verify cycle edges are in remaining
            cycle_edges = set()
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                if edge not in remaining:
                    valid = False
                    break
                cycle_edges.add(edge)
            
            if valid:
                pieces.append(cycle)
                remaining -= cycle_edges
            else:
                # Fallback
                e = next(iter(remaining))
                pieces.append(list(e))
                remaining.discard(e)
        
        for e in remaining:
            pieces.append(list(e))
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose_improved(edges, n)
        results.append(pieces)
    
    return results
