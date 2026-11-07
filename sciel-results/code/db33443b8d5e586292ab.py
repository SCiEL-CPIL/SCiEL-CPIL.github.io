import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(e) for e in P['edges'])
    base_edges.update((v, u) for u, v in base_edges)
    
    # Precompute all possible edges not in base
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_possible.append((u, v))
                
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45.0  # 45s limit out of 60s
    
    best_edges = []
    
    # Greedy Set Cover approach
    # State: set of added edges
    current_edges = set()
    
    def get_adjacency(added):
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            if u < v:
                adj[u].add(v)
                adj[v].add(u)
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_triangle_free(added):
        # Check for triangles
        # A triangle exists if there is an edge (u,v) and a common neighbor w
        # such that (u,w) and (v,w) are edges.
        # Since base is a path, triangles can only involve added edges.
        # 1. Triangle with 1 added edge (u,v) and 2 path edges:
        #    This happens if u and v are distance 2 in the path.
        #    i.e., |u-v| == 2.
        for u, v in added:
            if abs(u - v) == 2:
                return False
        # 2. Triangle with 2 added edges (u,v), (v,w) and 1 path edge (u,w) or similar.
        #    Since added edges are a set, we check if any two added edges share a vertex
        #    and the third side is a path edge or another added edge.
        #    Actually, simpler: just check all triples? No, n=24 is small.
        #    But we can be smarter.
        #    If we add edges one by one, we only need to check triangles involving the new edge.
        #    Here we check the whole set.
        
        adj = get_adjacency(added)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    # Check common neighbors
                    common = adj[u].intersection(adj[v])
                    if common:
                        return False
        return True

    def calculate_diameter(added):
        adj = get_adjacency(added)
        max_dist = 0
        for start in range(n):
            # BFS
            dist = {start: 0}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if len(dist) < n:
                return float('inf') # Disconnected
            max_dist = max(max_dist, max(dist.values()))
        return max_dist

    # Initial check
    if calculate_diameter(set()) <= 4:
        return []

    # Greedy search
    current = set()
    
    while True:
        if time.time() > deadline:
            break
        
        if calculate_diameter(current) <= 4:
            break
            
        best_gain = -1
        best_edge = None
        
        for u, v in all_possible:
            if (u, v) in current or (v, u) in current:
                continue
                
            # Check if adding this edge keeps it triangle-free
            new_set = current | {(u, v)}
            if not is_triangle_free(new_set):
                continue
                
            # Calculate diameter (or just check if it improves)
            # To save time, we can just check if diameter <= 4 or if it reduces max distance
            # For greedy, we want the edge that reduces diameter the most.
            # Calculating full diameter is O(N^2). N=24, so it's fast.
            d = calculate_diameter(new_set)
            
            # We want to minimize diameter.
            # If d <= 4, it's a winner.
            # Otherwise, we pick the one with smallest d.
            if d < best_gain or best_gain == -1:
                best_gain = d
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        current.add(best_edge)
        
    # If we found a solution
    if calculate_diameter(current) <= 4:
        # Convert to list of lists
        result = [[u, v] for u, v in current]
        # Ensure types are Python int
        return [[int(u), int(v)] for u, v in result]
    else:
        # Fallback: return a known valid solution or empty if none found
        # The seed program returns all (u,v) with (u-v)%2 and not in base.
        # That is definitely valid but large.
        # We should return the best we found, even if invalid? No, invalid scores 0.
        # We must return a valid solution.
        # If greedy fails, use the seed logic (which is valid but large).
        seed_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u - v) % 2 and (u, v) not in base_edges:
                    seed_edges.append([u, v])
        return seed_edges

# EVOLVE-BLOCK-END
