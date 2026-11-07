# EVOLVE-BLOCK-START
import copy
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Initialize base distance matrix
    base_dist = [[abs(i - j) for j in range(n)] for i in range(n)]
    
    # Generate candidate edges: pairs (u, v) with u < v, (u, v) not in base, and no triangle with base.
    # Triangle with base means common neighbor.
    # In a path, common neighbor implies dist(u, v) == 2.
    # So dist(u, v) >= 3.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            if v - u < 3:
                continue
            candidates.append((u, v))
            
    # Sort candidates: prefer longer jumps first? Or those that reduce diameter most?
    # For backtracking, ordering affects speed.
    # Let's sort by distance descending.
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    def get_diameter(dist):
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d

    def check_conflict(new_u, new_v, current_edges):
        """
        Check if adding (new_u, new_v) creates a triangle with any edge in current_edges.
        Triangle (a, b, c) exists if edges (a,b), (b,c), (c,a) are all present.
        Base edges are fixed. New edges are added.
        Conflict cases:
        1. (new_u, new_v) and (a, b) in current_edges.
           If they share a vertex, say new_u == a.
           Then we have edges (new_u, new_v) and (new_u, b).
           Triangle exists if (new_v, b) is a base edge.
           i.e., abs(new_v - b) == 1.
        2. If they don't share a vertex, no triangle is formed immediately by just these two edges
           and base edges, unless there's a path of length 2 between non-shared vertices via base?
           No, triangle requires 3 edges.
           Edges involved: (new_u, new_v), (a, b), and base edges.
           If no shared vertex, the only way to form a triangle is if new_u is connected to a or b via base?
           No, base edges are fixed.
           If (new_u, new_v) and (a, b) are disjoint, they can't form a triangle with each other directly.
           They could form a triangle with a third edge, but we are checking incremental addition.
           The only new triangles formed by adding (new_u, new_v) must involve (new_u, new_v) and two other edges.
           The other two edges must be either base or previously added.
           If one is base and one is added:
             Base (x, y), Added (z, w).
             If (new_u, new_v) shares vertex with Base?
             We already checked (new_u, new_v) against Base in candidate generation (dist >= 3).
             So (new_u, new_v) does not form triangle with any single base edge.
             Does it form triangle with Base + Added?
             Triangle vertices A, B, C.
             Edges: (A,B) base, (B,C) added (old), (C,A) added (new).
             Or (A,B) base, (B,C) added (new), (C,A) added (old).
             This reduces to: Does (new_u, new_v) share a vertex with an old added edge (a,b) such that the other endpoints are adjacent in base?
             Yes, that's case 1 above.
             
             What about: (A,B) added (new), (B,C) base, (C,A) added (old)?
             This is symmetric to case 1.
             
             What about: (A,B) added (new), (B,C) added (old), (C,A) base?
             This is also symmetric to case 1.
             
             So checking shared vertex with old added edge and base-adjacency of other endpoints is sufficient.
        """
        for a, b in current_edges:
            # Check if they share a vertex
            if new_u == a or new_u == b or new_v == a or new_v == b:
                # Determine the other two vertices
                if new_u == a:
                    # Shared new_u (a). Other endpoints: new_v, b
                    if abs(new_v - b) == 1:
                        return True
                elif new_u == b:
                    # Shared new_u (b). Other endpoints: new_v, a
                    if abs(new_v - a) == 1:
                        return True
                elif new_v == a:
                    # Shared new_v (a). Other endpoints: new_u, b
                    if abs(new_u - b) == 1:
                        return True
                elif new_v == b:
                    # Shared new_v (b). Other endpoints: new_u, a
                    if abs(new_u - a) == 1:
                        return True
        return False

    # Search for minimum k
    for k in range(0, 11):
        # DFS
        # State: index in candidates, remaining edges to pick, current list of edges, current dist matrix
        def dfs(idx, remaining, current_edges, dist):
            if remaining == 0:
                if get_diameter(dist) <= 4:
                    return list(current_edges)
                return None
            
            if idx >= len(candidates):
                return None
                
            # Pruning: not enough candidates left
            if len(candidates) - idx < remaining:
                return None
                
            # Option 1: Skip candidates[idx]
            res = dfs(idx + 1, remaining, current_edges, dist)
            if res is not None:
                return res
                
            # Option 2: Take candidates[idx]
            u, v = candidates[idx]
            if not check_conflict(u, v, current_edges):
                # Add edge
                new_edges = current_edges + [(u, v)]
                new_dist = copy.deepcopy(dist)
                # Update distances
                for i in range(n):
                    for j in range(n):
                        # new path via u-v
                        nd1 = new_dist[i][u] + 1 + new_dist[v][j]
                        nd2 = new_dist[i][v] + 1 + new_dist[u][j]
                        nd = min(new_dist[i][j], nd1, nd2)
                        if nd < new_dist[i][j]:
                            new_dist[i][j] = nd
                            new_dist[j][i] = nd
                            
                res = dfs(idx + 1, remaining - 1, new_edges, new_dist)
                if res is not None:
                    return res
                    
            return None

        # Start with base dist
        initial_dist = copy.deepcopy(base_dist)
        solution = dfs(0, k, [], initial_dist)
        if solution is not None:
            return [[u, v] for u, v in solution]
            
    # Fallback (should not happen)
    return []

# EVOLVE-BLOCK-END
