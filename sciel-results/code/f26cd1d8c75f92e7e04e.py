import time
from collections import deque

# Task Constants
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s limit to be safe
    
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    # 1. Setup Bipartition
    # The path graph is bipartite. Even vertices are 0, 2, ..., 22. Odd are 1, 3, ..., 23.
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    # Candidate edges: All cross-partity pairs not in base graph.
    # Adding edges within the same part creates odd cycles (triangles) if connected via path?
    # Actually, adding (even, even) creates a cycle of length 2k+1? No.
    # Path is bipartite. Adding (u, v) where u, v in same part creates an odd cycle.
    # Since we need triangle-free, and any cycle of length 3 is an odd cycle, we must avoid same-part edges.
    # Wait, does same-part edge *always* create a triangle?
    # If we add (0, 2), we have edge (0,1), (1,2). 0-1-2-0 is a triangle.
    # So yes, for a path graph, any shortcut (u, v) with dist(u,v) > 1 creates a cycle of length dist+1.
    # If u, v same parity, dist is even, cycle is odd.
    # If cycle length is 3, it's a triangle.
    # If cycle length is 5, 7, etc., it is triangle-free but not bipartite.
    # The prompt requires "triangle-free", not "bipartite".
    # However, the current strategy relies on bipartite.
    # Let's check if non-bipartite triangle-free graphs allow fewer edges.
    # A 5-cycle is triangle-free.
    # But finding the global minimum over all triangle-free graphs is harder.
    # The "Bipartite Distance Domination" strategy is a specific, tractable subset.
    # Given the stagnation, let's stick to the Bipartite constraint which is proven safe and allows
    # for a clean Branch and Bound formulation. It is a structural change from "Fixed Hub" to "Optimal Bipartite Search".
    
    candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (min(u, v), max(u, v)) not in base_set:
                candidates.append((u, v))
                
    # Precompute base distances
    # Base graph is a path. dist(u, v) = abs(u - v).
    # We need to ensure that with added edges, diameter <= 4.
    
    # Branch and Bound State
    best_edges = None
    best_count = len(candidates) + 1 # Upper bound
    
    # Initial heuristic to get a good upper bound (the current solution)
    # Hub 11 connected to even vertices excluding neighbors.
    hub = 11
    init_edges = []
    for v in even_verts:
        if v == 10 or v == 12:
            continue
        init_edges.append((hub, v))
    best_count = len(init_edges)
    best_edges = list(init_edges)
    
    # Check if we can do better.
    # Lower bound: 
    # The diameter of the path is 23. We need to reduce it to 4.
    # Each edge can reduce the diameter.
    
    # To make B&B efficient, we order candidates by potential impact.
    # Edges that connect vertices far apart in the path are more useful.
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    # We will use a recursive search.
    # State: index in candidates, current edge set, current diameter (if we can compute it fast).
    # Computing diameter every step is O(N^2) = 576 ops. 
    # If depth is ~10, and branching factor is high, this might be slow.
    # However, N=24 is small.
    
    # Optimization: Only check diameter if current_count + remaining_potential < best_count?
    # Actually, we can prune if current_count >= best_count.
    
    # Let's use a smarter pruning:
    # If we have selected k edges, and k >= best_count, prune.
    
    # We also need a lower bound on the *additional* edges needed.
    # This is hard to compute exactly.
    # But we can check: if the current graph (base + selected) has diameter > 4,
    # and we have run out of candidates, it's invalid.
    
    # Let's implement a simple DFS with backtracking.
    
    def bfs_diameter_fast(adj_list):
        # adj_list: list of sets
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_d:
                            max_d = dist[v]
                        q.append(v)
            if max_d > 4: # Early exit if we already exceed limit from this source? 
                # Actually, if any source has dist > 4, diameter > 4.
                # But we need to finish BFS to know if it's reachable? 
                # No, if dist[v] > 4, diameter is > 4. We can return early.
                return max_d
        return max_d

    # Build initial adjacency list for base graph
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    # Deep copy base_adj for the search?
    # It's better to maintain a dynamic adjacency list and add/remove edges.
    
    current_adj = [set() for _ in range(n)]
    for u in range(n):
        current_adj[u] = base_adj[u].copy()
        
    def search(idx, count):
        nonlocal best_count, best_edges, current_adj
        
        if time.time() > deadline:
            return
            
        if count >= best_count:
            return
            
        if idx == len(candidates):
            # Check if valid
            if bfs_diameter_fast(current_adj) <= 4:
                best_count = count
                best_edges = [list(e) for e in selected_edges]
            return
            
        # Branch 1: Include candidate[idx]
        u, v = candidates[idx]
        current_adj[u].add(v)
        current_adj[v].add(u)
        selected_edges.append((u, v))
        
        search(idx + 1, count + 1)
        
        # Backtrack
        current_adj[u].remove(v)
        current_adj[v].remove(u)
        selected_edges.pop()
        
        # Branch 2: Exclude candidate[idx]
        search(idx + 1, count)
        
    selected_edges = []
    search(0, 0)
    
    # Fallback to initial heuristic if B&B didn't find anything better or timed out with empty
    if not best_edges:
        best_edges = [list(e) for e in init_edges]
        
    # Final verification
    final_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        final_adj[u].add(v)
        final_adj[v].add(u)
    for u, v in best_edges:
        final_adj[u].add(v)
        final_adj[v].add(u)
        
    # Check triangle-free
    for u in range(n):
        for v in final_adj[u]:
            if v > u:
                for w in final_adj[u]:
                    if w != v and w in final_adj[v]:
                        # Triangle found. This should not happen with bipartite constraints.
                        # If it does, fallback to init_edges
                        best_edges = [list(e) for e in init_edges]
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break
        
    # Check diameter
    if bfs_diameter_fast(final_adj) > 4:
         # Should not happen, but fallback
         best_edges = [list(e) for e in init_edges]

    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
