"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Build base adjacency
    base_edges = set()
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_edges.add((min(u,v), max(u,v)))
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges: only those with |u-v| >= 3 to avoid immediate triangles with path
    candidates = []
    for u in range(n):
        for v in range(u+3, n):  # v - u >= 3
            candidates.append((u, v))
    
    # Sort by distance descending (higher impact first)
    candidates.sort(key=lambda e: -(e[1] - e[0]))
    
    def is_triangle_free(adj):
        """Check triangle-free in O(n * deg^2)"""
        for i in range(n):
            neighbors = list(adj[i])
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    if neighbors[k] in adj[neighbors[j]]:
                        return False
        return True
    
    def compute_diameter(adj):
        """BFS from each node, return diameter or inf if disconnected"""
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_d:
                            max_d = dist[v]
                        q.append(v)
            if -1 in dist:
                return float('inf')
        return max_d
    
    def valid_solution(adj, added):
        """Check if solution is valid: triangle-free, connected, diameter <= 4"""
        if not is_triangle_free(adj):
            return False
        d = compute_diameter(adj)
        return d <= 4
    
    # Greedy to get upper bound
    def greedy():
        adj_g = [set(s) for s in adj]
        added = []
        used = set()
        for _ in range(8):
            if time.time() > deadline:
                break
            d = compute_diameter(adj_g)
            if d <= 4:
                break
            best_edge = None
            best_d = d
            for (u, v) in candidates:
                if (u, v) in used or (v, u) in used:
                    continue
                # Check triangle: common neighbor?
                if adj_g[u] & adj_g[v]:
                    continue
                adj_g[u].add(v)
                adj_g[v].add(u)
                nd = compute_diameter(adj_g)
                adj_g[u].remove(v)
                adj_g[v].remove(u)
                if nd < best_d:
                    best_d = nd
                    best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
            adj_g[u].add(v)
            adj_g[v].add(u)
            used.add((u, v))
            added.append((u, v))
        if compute_diameter(adj_g) <= 4 and is_triangle_free(adj_g):
            return added
        return None
    
    g = greedy()
    best = g if g else [None] * 50
    
    # Backtracking search
    def backtrack(adj_s, added, idx, best):
        if time.time() > deadline:
            return best
        if len(added) >= len(best):
            return best
        
        # Quick check: is current state valid?
        if is_triangle_free(adj_s):
            d = compute_diameter(adj_s)
            if d <= 4:
                if len(added) < len(best):
                    best = list(added)
                return best
        
        for i in range(idx, len(candidates)):
            u, v = candidates[i]
            # Triangle check
            if adj_s[u] & adj_s[v]:
                continue
            adj_s[u].add(v)
            adj_s[v].add(u)
            added.append((u, v))
            
            best = backtrack(adj_s, added, i+1, best)
            
            adj_s[u].remove(v)
            adj_s[v].remove(u)
            added.pop()
            
            if time.time() > deadline:
                return best
        
        return best
    
    adj_s = [set(s) for s in adj]
    result = backtrack(adj_s, [], 0, best)
    
    if result and result != [None] * 50:
        return [[int(u), int(v)] for u, v in result]
    elif g:
        return [[int(u), int(v)] for u, v in g]
    else:
        return []

# EVOLVE-BLOCK-END
