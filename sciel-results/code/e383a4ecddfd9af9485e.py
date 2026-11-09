"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u, v), max(u, v)))
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(added_edges):
        adj = build_adj(added_edges)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True
    
    def get_diameter(added_edges):
        adj = build_adj(added_edges)
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            if -1 in dist:
                return -1
            max_diam = max(max_diam, max(dist))
        return max_diam
    
    def is_valid(added_edges):
        added_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in base_edges:
                return False
            if key in added_set:
                return False
            added_set.add(key)
        
        if not is_triangle_free(added_edges):
            return False
        
        diam = get_diameter(added_edges)
        if diam < 0 or diam > 4:
            return False
        
        return True

    def local_search_minimize(added_edges):
        """Try to remove edges while maintaining validity, using randomization to avoid local minima."""
        current = list(added_edges)
        if not is_valid(current):
            return None
            
        improved = True
        while improved:
            improved = False
            # Shuffle indices to try different removal orders
            indices = list(range(len(current)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(current):
                    continue
                trial = current[:i] + current[i+1:]
                if is_valid(trial):
                    current = trial
                    improved = True
                    break
        return current

    best_solution = None
    
    # Try different jump lengths for initial guess
    for jump in [5, 6, 7, 8]:
        candidate = []
        for i in range(n - jump):
            candidate.append((i, i + jump))
        
        # Add some "cross" edges if needed? 
        # For now, just try the band.
        
        if is_valid(candidate):
            minimized = local_search_minimize(candidate)
            if minimized is not None:
                if best_solution is None or len(minimized) < len(best_solution):
                    best_solution = minimized
    
    # Try mixed jumps
    for j1 in [4, 5]:
        for j2 in [6, 7, 8]:
            if j1 == j2:
                continue
            candidate = []
            for i in range(n - j1):
                candidate.append((i, i + j1))
            for i in range(n - j2):
                if (i, i + j2) not in candidate:
                    candidate.append((i, i + j2))
            if is_valid(candidate):
                minimized = local_search_minimize(candidate)
                if minimized is not None:
                    if best_solution is None or len(minimized) < len(best_solution):
                        best_solution = minimized

    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback: Greedy add
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_possible.append((u, v))
                
    current = []
    for u, v in all_possible:
        trial = current + [(u, v)]
        if is_valid(trial):
            current = trial
            
    if is_valid(current):
        minimized = local_search_minimize(current)
        if minimized is not None:
            return [[int(u), int(v)] for u, v in minimized]
            
    return []
# EVOLVE-BLOCK-END
