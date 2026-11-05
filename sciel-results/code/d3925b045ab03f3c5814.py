"""Construct a C4-free graph with high minimum degree to certify R(C4,K1,39) > n."""
# EVOLVE-BLOCK-START
import numpy as np
import random
from itertools import combinations

def construct():
    """
    Construct a C4-free graph with maximum n and minimum degree >= n-39.
    
    Approach: Model as maximum independent set in a conflict graph.
    - Vertices of conflict graph = potential edges of K_n (pairs (u,v) with u<v)
    - Two potential edges conflict if:
      1. They share a vertex (can't both be in a matching... wait, they can)
      Actually, the constraint is C4-freeness. Two edges (a,b) and (c,d) create a C4
      together with two other edges. But for the independent set formulation:
      We want to select a set of edges such that no C4 exists.
      
      A C4 is 4 vertices a,b,c,d with edges (a,b),(b,c),(c,d),(d,a).
      So the conflict is: if we have edges (a,b) and (c,d), and also (b,c) and (d,a),
      that forms a C4. But that's a 4-edge constraint, not a 2-edge constraint.
      
      Alternative: track for each pair of vertices, the number of common neighbors.
      A C4 exists iff some pair has >= 2 common neighbors... no, that's for the 
      graph itself. Actually a C4 exists iff there are 4 vertices a,b,c,d with 
      a-b, b-c, c-d, d-a all present.
      
      For the greedy approach: when considering adding edge (u,v), check if it
      creates a C4. A C4 using (u,v) would be u-v-x-y-u where x~v, x~y, y~u.
      So we need: exists x in N(v) and y in N(u) with x!=y, x!=u, y!=v, and x~y.
      
      Let me use the greedy edge addition with C4 checking, but with a better
      heuristic and multiple random restarts.
    """
    
    def verify_c4_free(adj, n):
        """Check if graph is C4-free using bitsets for speed."""
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    continue
                # Check for C4: u-v-x-y-u
                # x must be neighbor of v (not u), y must be neighbor of u (not v)
                # and x-y must be an edge, x != y
                for x in range(n):
                    if x == u or x == v or not adj[v][x]:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == x or not adj[u][y]:
                            continue
                        if adj[x][y]:
                            return False
        return True
    
    def try_build(n, seed, min_deg_target):
        """Try to build a C4-free graph on n vertices with min degree >= min_deg_target."""
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        degree = [0] * n
        
        def can_add_edge(u, v):
            """Check if adding edge (u,v) creates a C4 in current graph."""
            # C4: u-v-x-y-u where x~v, x~y, y~u, all distinct
            for x in range(n):
                if x == u or x == v or not adj[v][x]:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x or not adj[u][y]:
                        continue
                    if adj[x][y]:
                        return False
            return True
        
        # Phase 1: Greedy - add edges prioritizing low-degree vertices
        max_rounds = 500
        for rnd in range(max_rounds):
            min_deg = min(degree)
            if min_deg >= min_deg_target:
                break
            
            # Collect all valid edges
            valid_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        continue
                    if can_add_edge(u, v):
                        valid_edges.append((u, v))
            
            if not valid_edges:
                break
            
            # Score each edge: prefer edges connecting low-degree vertices
            def edge_score(e):
                u, v = e
                # Higher score = more urgent
                return -(min(degree[u], degree[v]) * 100 + max(degree[u], degree[v]))
            
            valid_edges.sort(key=edge_score)
            
            # Add top edges (maybe multiple per round for speed)
            added_this_round = 0
            for u, v in valid_edges:
                if adj[u][v]:
                    continue
                if not can_add_edge(u, v):
                    continue
                adj[u][v] = adj[v][u] = True
                degree[u] += 1
                degree[v] += 1
                added_this_round += 1
                if added_this_round >= 5:
                    break
            
            if added_this_round == 0:
                break
        
        # Phase 2: Try to boost minimum degree further
        for rnd in range(200):
            min_deg = min(degree)
            if min_deg >= min_deg_target:
                break
            
            # Find vertices below target
            low_verts = [i for i in range(n) if degree[i] < min_deg_target]
            if not low_verts:
                break
            
            # Try to add edges involving low-degree vertices
            improved = False
            for u in low_verts:
                for v in range(n):
                    if v == u or adj[u][v]:
                        continue
                    if can_add_edge(u, v):
                        adj[u][v] = adj[v][u] = True
                        degree[u] += 1
                        degree[v] += 1
                        improved = True
                        break
                if degree[u] >= min_deg_target:
                    continue
            if not improved:
                break
        
        # Verify
        min_deg = min(degree)
        if min_deg < min_deg_target:
            return None
        if not verify_c4_free(adj, n):
            return None
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges
    
    # Try n=47 first, then 46, etc.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        min_deg_target = n - 39
        if min_deg_target < 0:
            min_deg_target = 0
        
        # Try multiple random seeds
        for seed in range(20):
            result = try_build(n, seed, min_deg_target)
            if result is not None:
                return {'n': n, 'edges': result}
    
    # Fallback: return n=39 with no edges (min degree 0 >= 39-39=0)
    return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
