"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import math
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_dist(adj, n):
    """Compute all-pairs shortest paths. Returns list of lists of distances."""
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
    return dist

def _get_diameter(dist):
    max_d = 0
    for i in range(len(dist)):
        for j in range(len(dist)):
            if dist[i][j] > max_d:
                max_d = dist[i][j]
    return max_d

def _verify_solution(added_edges):
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    seen_added = set()
    for u, v in added_edges:
        if (u, v) in base_edges_set or (v, u) in base_edges_set:
            return False
        if (u, v) in seen_added or (v, u) in seen_added:
            return False
        seen_added.add((u, v))
        seen_added.add((v, u))
        
        adj[u].append(v)
        adj[v].append(u)
        
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
                    
    # Check connected and diameter
    dist = _bfs_dist(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False
                
    diam = _get_diameter(dist)
    if diam > 4:
        return False
        
    return True

def _construct_hub_solution(hub):
    """
    Construct a solution using a hub-centric strategy.
    1. Add edges (hub, v) for all v != hub.
    2. This creates a star-like structure on top of the path.
    3. Check diameter. If > 4, add edges to reduce it.
    
    Why this works:
    - Hub is connected to everyone.
    - Path edges still exist.
    - Triangle-free constraint: 
      - Hub's neighbors are all other vertices.
      - If we add (hub, u) and (hub, v), do they form a triangle?
      - Triangle requires (u, v) to be an edge.
      - In the base path, (u, v) is an edge only if |u-v|=1.
      - So if we add (hub, u) and (hub, v) where |u-v|=1, we create a triangle (hub, u, v).
      - WAIT. The base path already has edges (u, u+1).
      - If we add (hub, u) and (hub, u+1), and (u, u+1) is in base, then (hub, u, u+1) is a triangle.
      - So we CANNOT add edges from hub to adjacent vertices in the path?
      - Let's check: Base has (0,1). If hub=12, we add (12,0) and (12,1).
      - Triangle (12, 0, 1)? Edges: (12,0), (12,1), (0,1). Yes, triangle!
      - So we cannot add (hub, v) if v is adjacent to another vertex w that is also connected to hub.
      - Since hub connects to ALL, any pair of adjacent vertices (u, u+1) in the path will form a triangle with hub if hub is connected to both.
      - This means the "connect hub to all" strategy is INVALID for triangle-free if the path has edges.
      
      Correction:
      We must ensure that for any added edge (hub, u), there is no v such that (hub, v) is added AND (u, v) is in base (or added).
      Since base has all (i, i+1), if we add (hub, u) and (hub, u+1), we get a triangle.
      So we can only add (hub, u) if we do NOT add (hub, u-1) or (hub, u+1)?
      No, we want to add as many as possible to reduce diameter.
      Actually, if we add (hub, u), we must NOT have added (hub, u-1) or (hub, u+1) IF (u, u-1) or (u, u+1) is a base edge.
      Since ALL consecutive pairs are base edges, we can only add (hub, u) for u such that we don't add (hub, u±1).
      This implies the set of vertices connected to hub must be an independent set in the path graph (no two adjacent).
      
      So, Strategy v2:
      1. Choose hub H.
      2. Select a maximal independent set I of vertices (excluding H) to connect to H.
         To minimize diameter, we want I to be "dense" or cover the path well?
         Actually, if we connect H to I, the distance from H to any v in I is 1.
         Distance from H to any v not in I is 2 (via some neighbor in I? No, via path?).
         Let's analyze distances.
         
      Let's try a different hub strategy that is valid.
      The constraint is: The set of neighbors of any vertex must be an independent set.
      For hub H, its neighbors N(H) must be an independent set in G (including base edges).
      Since base is a path, N(H) cannot contain adjacent vertices.
      
      So, we pick H. We pick a set S of vertices to connect to H.
      S must be an independent set in the path (no two elements in S are adjacent in the path).
      To minimize diameter, we want S to be as large as possible? Or strategically placed?
      If S is large, many vertices are at dist 1 from H.
      Vertices not in S are at dist 2 from H (if they are adjacent to S) or more.
      
      Let's just implement a greedy search over hubs and independent sets.
      But "maximal independent set" is easy to find (take every other vertex).
      If we take every other vertex, the graph is bipartite-ish?
      
      Let's try:
      Hub = 12.
      Connect 12 to {0, 2, 4, 6, 8, 10, 14, 16, 18, 20, 22}. (All even, except 12 is even, so 12 is not in S? 12 is hub).
      S = {0, 2, 4, 6, 8, 10, 14, 16, 18, 20, 22}.
      Is S independent? Yes, no two are adjacent in path.
      Edges added: (12, 0), (12, 2), ..., (12, 22).
      Count: 11 edges.
      Diameter?
      Dist(0, 23): 0-12-...-23? 0-12 is 1. 12-23 is 11 (path). Total 12? No, 12 is connected to 22. 22-23 is 1. So 0-12-22-23 is 3.
      Dist(1, 23): 1-0-12-22-23 (4) or 1-2-12-22-23 (4).
      Dist(1, 21): 1-2-12-22-21 (4).
      Dist(0, 1): 1 (base).
      Dist(1, 3): 1-2-3 (2) or 1-2-12... no.
      Let's check diameter of this 11-edge solution.
      It might be > 4.
      
      Can we do better with fewer edges?
      The inspiration mentioned "hub-star with vertex 12 plus a minimum independent dominating set of the two far path regions".
      This suggests not connecting to ALL, but to a dominating set.
      
      Let's implement a search that:
      1. Tries each hub h in 0..23.
      2. For each hub, constructs a valid set of edges incident to h (independent set in path).
      3. Adds other edges if needed to reduce diameter.
      4. Verifies and keeps the best.
      
      To keep it fast and simple, I will use a greedy "repair" approach.
      Start with base.
      Try adding edges from a predefined list of "good" edges (long chords) in a greedy manner, checking validity.
      But the "different strategy" requirement pushes me to the Hub-Independent-Set approach.
      
      Let's refine the Hub approach:
      For a fixed hub h:
      - We can add edges (h, v) for v in S, where S is an independent set in P \ {h}.
      - We want to choose S to minimize the diameter of G + {(h, v) : v in S}.
      - If diameter is still > 4, we can add non-hub edges.
      
      How to choose S?
      Greedy: Start with S = empty.
      While diam > 4:
         Find v that minimizes new diameter if we add (h, v) (and v is not adjacent to any in S).
         Add (h, v) to S.
      Then add other edges if needed.
      
      This is a constructive heuristic. It's structurally different from IDDFS.
    """
    n = P['n']
    base_adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    best_sol = None
    best_size = float('inf')
    
    # Precompute base dist
    base_dist = _bfs_dist(base_adj, n)
    
    for h in range(n):
        # State for this hub
        adj = [list(x) for x in base_adj]
        added = []
        added_set = set()
        
        # S: set of vertices connected to h
        S = set()
        
        # Greedy selection of S
        # We want to reduce diameter.
        # Initially, diam is 23.
        # Adding (h, v) reduces dist(h, v) to 1.
        
        current_diam = 23 # approx
        
        while current_diam > 4:
            best_v = -1
            best_new_diam = current_diam + 1
            
            for v in range(n):
                if v == h:
                    continue
                if v in S:
                    continue
                # Check if v can be added to S
                # v must not be adjacent to any u in S in the base path
                # i.e., |v - u| != 1 for all u in S
                valid = True
                for u in S:
                    if abs(v - u) == 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Test adding (h, v)
                adj[h].append(v)
                adj[v].append(h)
                
                new_dist = _bfs_dist(adj, n)
                new_diam = _get_diameter(new_dist)
                
                adj[h].remove(v)
                adj[v].remove(h)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_v = v
            
            if best_v == -1:
                break # No more valid hub edges can be added
            
            # Commit
            adj[h].append(best_v)
            adj[best_v].append(h)
            added.append([h, best_v])
            added_set.add((h, best_v))
            added_set.add((best_v, h))
            S.add(best_v)
            current_diam = best_new_diam
            
        # If diam is still > 4, try adding non-hub edges
        # Simple greedy for remaining
        while current_diam > 4 and len(added) < 15: # Safety cap
            best_u, best_v = -1, -1
            best_new_diam = current_diam + 1
            
            # Candidates: all non-base, non-added edges
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) in added_set or (v, u) in added_set:
                        continue
                    if (u, v) in [(a, b) for a, b in P['edges']] or (v, u) in [(a, b) for a, b in P['edges']]:
                        continue
                    # Triangle check
                    # Common neighbor?
                    common = set(adj[u]).intersection(adj[v])
                    if common:
                        continue
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    new_dist = _bfs_dist(adj, n)
                    new_diam = _get_diameter(new_dist)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                    if new_diam < best_new_diam:
                        best_new_diam = new_diam
                        best_u, best_v = u, v
                        
            if best_u == -1:
                break
                
            adj[best_u].append(best_v)
            adj[best_v].append(best_u)
            added.append([best_u, best_v])
            added_set.add((best_u, best_v))
            added_set.add((best_v, best_u))
            current_diam = best_new_diam
            
        if current_diam <= 4 and len(added) < best_size:
            best_sol = [e[:] for e in added]
            best_size = len(added)
            
    return best_sol

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    sol = _construct_hub_solution(0) # Pass 0, loop handles all
    
    if sol is None:
        # Fallback: try a simple greedy without hub bias
        # Or return empty if nothing works (should not happen)
        return []
        
    if _verify_solution(sol):
        res = []
        for u, v in sol:
            res.append([int(u), int(v)])
        return res
    else:
        return []

# EVOLVE-BLOCK-END
