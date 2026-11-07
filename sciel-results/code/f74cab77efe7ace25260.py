import time
import itertools
from collections import deque

def construct():
    n = 16
    deadline = time.time() + 80  # Leave 10s safety margin
    
    # Start with K_8,8 (the maximum triangle-free graph on 16 vertices)
    # This gives us 64 edges. We try to add more edges while maintaining
    # the property that every edge is essential for diameter 2.
    
    half = n // 2
    base_edges = set()
    for u in range(half):
        for v in range(half, n):
            base_edges.add((u, v))
    
    # All possible edges not in base
    all_possible = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_possible.add((u, v))
    
    candidate_edges = all_possible - base_edges
    
    # We want to add as many edges as possible while maintaining the property.
    # Strategy: greedily try to add edges one by one, checking the property.
    
    def build_adjacency(edge_set):
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def has_diameter_two(edge_set):
        adj = build_adjacency(edge_set)
        for start in range(n):
            # BFS
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 2:
                            return False
                        q.append(v)
            if any(d == -1 for d in dist):
                return False
        return True
    
    def all_edges_essential(edge_set):
        """Check that removing any single edge increases diameter above 2 or disconnects."""
        for (u, v) in edge_set:
            reduced = set(edge_set)
            reduced.discard((u, v))
            reduced.discard((v, u))
            
            adj = build_adjacency(reduced)
            
            # Check if disconnected
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    x = q.popleft()
                    for y in adj[x]:
                        if dist[y] == -1:
                            dist[y] = dist[x] + 1
                            q.append(y)
                if any(d == -1 for d in dist):
                    continue  # Disconnected is OK
                if any(d > 2 for d in dist):
                    continue  # Diameter > 2 is OK
                # Otherwise, diameter is still <= 2, so edge is NOT essential
                return False
        return True
    
    current_edges = set(base_edges)
    
    # Sort candidate edges by some heuristic (e.g., try edges that are "most central")
    # For now, just try them in a fixed order
    cand_list = list(candidate_edges)
    
    added_count = 0
    for (u, v) in cand_list:
        if time.time() > deadline - 5:
            break
        
        trial = set(current_edges)
        trial.add((u, v))
        
        # Quick check: does it still have diameter 2?
        if not has_diameter_two(trial):
            continue
        
        # Full check: are all edges essential?
        # This is expensive, so only do it if we're confident
        if added_count < 20 or (u + v) % 3 == 0:  # Limit full checks
            if all_edges_essential(trial):
                current_edges = trial
                added_count += 1
    
    # Convert to list of lists
    edges = []
    for (u, v) in current_edges:
        edges.append([int(u), int(v)])
    
    return edges
