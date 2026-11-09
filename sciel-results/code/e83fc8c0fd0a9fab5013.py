"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import sys
import collections

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial available edges set
    available_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        available_edges.add((u, v))
        
    def build_adj(avail_edges):
        adj = {i: set() for i in range(n)}
        for u, v in avail_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_triangle(avail_edges):
        """Find a triangle (3-cycle)."""
        adj = build_adj(avail_edges)
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    v = neighbors[i]
                    w = neighbors[j]
                    if v < w:
                        e_key = (v, w)
                    else:
                        e_key = (w, v)
                    if e_key in avail_edges:
                        return [u, v, w]
        return None

    def find_4_cycle(avail_edges):
        """Find a 4-cycle."""
        adj = build_adj(avail_edges)
        for u in range(n):
            for x in range(u+1, n):
                if x not in adj[u]:
                    continue
                common = adj[u].intersection(adj[x])
                if len(common) >= 2:
                    v, w = list(common)[:2]
                    # Cycle: u - v - x - w - u
                    return [u, v, x, w]
        return None

    def find_any_cycle(avail_edges):
        """Find any simple cycle using DFS."""
        adj = build_adj(avail_edges)
        visited = [False] * n
        parent = [-1] * n
        
        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    if dfs(v):
                        return True
                elif v != parent[u]:
                    # Found a back edge to v
                    # Reconstruct cycle
                    cycle = [v]
                    curr = u
                    while curr != v:
                        cycle.append(curr)
                        curr = parent[curr]
                    cycle.reverse()
                    return True
            return False
        
        for i in range(n):
            if not visited[i]:
                if dfs(i):
                    # The cycle is stored in the recursion? 
                    # No, the code above returns True but doesn't return the cycle.
                    # I need to modify dfs to return the cycle.
                    pass
        return None

    # Corrected find_any_cycle
    def find_any_cycle_corrected(avail_edges):
        adj = build_adj(avail_edges)
        visited = [False] * n
        parent = [-1] * n
        found_cycle = None
        
        def dfs(u):
            nonlocal found_cycle
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    if dfs(v):
                        return True
                elif v != parent[u]:
                    # Found a back edge to v
                    cycle = [v]
                    curr = u
                    while curr != v:
                        cycle.append(curr)
                        curr = parent[curr]
                    cycle.reverse()
                    found_cycle = cycle
                    return True
            return False
        
        for i in range(n):
            if not visited[i]:
                if dfs(i):
                    return found_cycle
        return None

    cycles = []
    max_cycles = 10
    
    for _ in range(max_cycles):
        cycle = find_triangle(available_edges)
        if cycle is None:
            cycle = find_4_cycle(available_edges)
        if cycle is None:
            cycle = find_any_cycle_corrected(available_edges)
            
        if cycle is None:
            break
            
        # Remove edges of the cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e_key = (min(u, v), max(u, v))
            if e_key in available_edges:
                available_edges.remove(e_key)
                
        cycles.append(cycle)
        
    if not cycles:
        # Fallback: return a single valid cycle if possible, or empty?
        # The graph definitely has cycles. If we fail, it's a bug.
        # Let's just return an empty list, which might be invalid, but better than crashing?
        # Actually, let's just return [ [0,1,2] ] if edges exist? No, that's dangerous.
        # We assume we find at least one.
        return []
        
    return cycles
# EVOLVE-BLOCK-END
