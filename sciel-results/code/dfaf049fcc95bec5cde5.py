import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85
    
    # Precompute base adjacency as bitmasks
    base_adj = [0] * n
    for u, v in base_edges:
        base_adj[u] |= (1 << v)
        base_adj[v] |= (1 << u)
    
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    # Generate candidate edges (must be cross-parity to avoid immediate triangles on path? 
    # Actually, if u and v have same parity, they are at even distance. 
    # If dist(u,v) is even, there is a path of even length.
    # If we add edge (u,v), we create a cycle of length dist(u,v)+1.
    # If dist(u,v) is even, cycle length is odd. 
    # Triangle-free requires no 3-cycles. 
    # An edge (u,v) creates a triangle if there is a path of length 2 between u and v.
    # In a path graph, dist(u,v) <= 2 implies triangle if we add (u,v) and u!=v and dist!=1.
    # If dist(u,v)=2, adding (u,v) creates a triangle.
    # So we must avoid edges with distance 2.
    # Distance 1 is already in base.
    # So we can only add edges with distance >= 3.
    # If distance is 3, adding (u,v) creates a 4-cycle (if no other chords). No triangle.
    # So candidates are pairs (u,v) with |u-v| >= 3 and (u,v) not in base.
    # Note: Cross-parity is NOT strictly required for triangle-freeness, only distance >= 3.
    # However, if u,v same parity, dist is even. If dist=4, adding (u,v) creates 5-cycle. OK.
    # If u,v diff parity, dist is odd. If dist=3, adding (u,v) creates 4-cycle. OK.
    # So just filter by distance >= 3.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if abs(u - v) >= 3:
                candidates.append((u, v))
    
    # Precompute conflict mask for each candidate
    # Edge i conflicts with edge j if they share a vertex OR form a triangle.
    # Triangle: (u,v) and (v,w) with u-w edge? No, triangle is 3 nodes a,b,c with all 3 edges.
    # If we add (u,v) and (u,w), triangle exists if (v,w) is in graph.
    # Graph has base edges + added edges.
    # So conflict between added edge e1=(u1,v1) and e2=(u2,v2) if:
    # 1. They share a vertex (u1==u2, etc.) -> This is NOT a conflict in terms of validity, 
    #    it's just that they are incident. But wait, if they share a vertex, say u1==u2=u, 
    #    then we have edges (u,v1) and (u,v2). Triangle if (v1,v2) is in graph.
    # 2. They form a triangle: e.g. e1=(a,b), e2=(b,c). Triangle if (a,c) in graph.
    # 3. They are the same edge.
    
    # Actually, the "independent set" formulation is on the set of added edges.
    # Two added edges e1, e2 are incompatible if adding both creates a triangle.
    # Triangle can be formed by:
    # - Two added edges + one base edge
    # - One added edge + two base edges
    # - Three added edges (if we add 3 edges forming a triangle)
    
    # Let's precompute for each pair of candidates (i, j) if they are compatible.
    # Compatible means: adding both to the graph does NOT create a triangle involving these edges.
    # A triangle involves 3 vertices.
    # If e1=(u,v) and e2=(x,y).
    # Case 1: Disjoint vertices. No triangle can be formed by just these two edges and base edges?
    #   Wait, if e1=(u,v) and e2=(x,y), and base has (u,x) and (v,y)? Then we have a 4-cycle u-v-y-x-u. No triangle.
    #   Triangle requires 3 vertices. If e1 and e2 are disjoint, they involve 4 vertices. No triangle can be formed by ONLY these two edges.
    #   However, a triangle could be formed by e1, e2, and a base edge? No, triangle has 3 edges.
    #   So if e1 and e2 are disjoint, they cannot form a triangle together.
    #   BUT, e1 might form a triangle with a base edge, and e2 might form a triangle with a base edge.
    #   We already filtered candidates to not form triangles with base edges (dist >= 3).
    #   So disjoint candidates are always compatible with each other?
    #   Yes, because any triangle must use at least 2 edges from the added set if the 3rd is base?
    #   No. If we add e1=(u,v), and base has (u,w) and (v,w), that's a triangle. We filtered these out.
    #   So any single candidate is safe.
    #   Two candidates e1, e2. If they share a vertex, say u. e1=(u,v), e2=(u,w).
    #   Triangle if (v,w) is in graph. (v,w) is in graph if it's base or another added edge.
    #   If (v,w) is base, then e1 and e2 are incompatible.
    #   If (v,w) is another added edge e3, then e1,e2,e3 form a triangle.
    #   So for pairwise compatibility, we only need to check if (v,w) is in BASE.
    #   If (v,w) is not in base, then e1 and e2 are pairwise compatible (triangle only forms if e3 is also added).
    
    # So, conflict graph G_conf:
    # Nodes: candidates.
    # Edge between i, j if:
    # 1. They share a vertex AND the third vertex pair is connected in BASE.
    # 2. They are the same edge (trivial).
    
    # Let's build this conflict graph.
    m = len(candidates)
    # conflict_mask[i] is a bitmask of candidates that conflict with i
    conflict_mask = [0] * m
    
    for i in range(m):
        u1, v1 = candidates[i]
        for j in range(i + 1, m):
            u2, v2 = candidates[j]
            # Check if they share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                # They share a vertex. Let's find the "other" endpoints.
                # The potential triangle would use the shared vertex and the two other endpoints.
                # Let a, b be the other endpoints.
                # We need to check if (a,b) is in base.
                if u1 == u2:
                    a, b = v1, v2
                elif u1 == v2:
                    a, b = v1, u2
                elif v1 == u2:
                    a, b = u1, v2
                else: # v1 == v2
                    a, b = u1, u2
                
                # Check if (a,b) is in base
                if base_adj[a] & (1 << b):
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
            # If they don't share a vertex, they are pairwise compatible (no triangle of size 3 using only these 2 edges + base).
            # Wait, is it possible to form a triangle with 2 added edges that DON'T share a vertex?
            # Triangle has 3 edges. If we use 2 added edges that are disjoint, we have 4 vertices involved.
            # The 3rd edge must connect two of these 4 vertices.
            # If the 3rd edge is base, it connects two vertices.
            # But a triangle is a cycle of length 3. Disjoint edges cannot be part of a 3-cycle.
            # So yes, disjoint candidates are always pairwise compatible.
    
    # Now we need to find the smallest set of candidates (independent set in G_conf)
    # such that the diameter of (Base + Set) <= 4.
    
    # Helper: Check diameter of graph defined by base_adj + added edges
    # We represent added edges as a bitmask over candidates? Or just a list of (u,v)?
    # For speed, let's maintain an adjacency list or bitmask for the current graph.
    
    def check_diameter(added_indices, limit=4):
        # Build adjacency bitmasks
        adj = [0] * n
        for i in range(n):
            adj[i] = base_adj[i]
        
        for idx in added_indices:
            u, v = candidates[idx]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # BFS from each node
        for src in range(n):
            # visited bitmask
            visited = 1 << src
            frontier = 1 << src
            dist = 0
            while frontier:
                dist += 1
                if dist > limit:
                    return False
                # Expand frontier
                new_frontier = 0
                f = frontier
                while f:
                    lsb = f & -f
                    u = lsb.bit_length() - 1
                    f ^= lsb
                    # Neighbors of u
                    nbrs = adj[u] & ~visited
                    new_frontier |= nbrs
                visited |= new_frontier
                frontier = new_frontier
            if visited != (1 << n) - 1:
                return False
        return True
    
    # Search for minimum k
    # k=1
    for i in range(m):
        if time.time() > deadline:
            break
        if check_diameter([i]):
            return [[int(candidates[i][0]), int(candidates[i][1])]]
    
    # k=2
    # Iterate over pairs (i, j) that are compatible (not in conflict)
    # To optimize, we can iterate i, then j > i where (conflict_mask[i] >> j) & 1 == 0
    for i in range(m):
        if time.time() > deadline:
            break
        # Skip if i alone works (already checked)
        for j in range(i + 1, m):
            if conflict_mask[i] & (1 << j):
                continue
            if check_diameter([i, j]):
                return [[int(candidates[i][0]), int(candidates[i][1])], 
                        [int(candidates[j][0]), int(candidates[j][1])]]
    
    # k=3
    # Iterate i, j, k compatible
    # This is O(m^3). m is roughly 24*23/2 - 23 - (distance 2 pairs) ~ 276 - 23 - 22 ~ 230.
    # 230^3 is ~ 12 million. Too slow if check_diameter is slow.
    # check_diameter is O(n^2) ~ 576 ops. 12M * 576 is ~ 7 billion ops. Too slow.
    # We need a better way for k=3.
    # Maybe precompute which pairs of edges reduce diameter significantly?
    # Or use a heuristic to prune the search space for k=3.
    # Or just limit the candidates for k=3 to a smaller set?
    
    # Let's try to prune candidates for k=3.
    # If an edge is "bad" (doesn't help much), skip it.
    # Or, we can just run a limited search.
    # Given the time budget, let's try to be smart.
    # We can check k=3 by iterating i, then j compatible with i, then k compatible with i and j.
    # To speed up, we can precompute for each i, the list of j compatible with i.
    
    compat_list = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if not (conflict_mask[i] & (1 << j)):
                compat_list[i].append(j)
                # We don't need to add i to compat_list[j] if we only use i<j ordering, 
                # but for k=3 we need k>j. So compat_list[i] should contain all j>i compatible.
                # Wait, my loop above only adds j to compat_list[i].
                # For k=3, we pick i < j < k.
                # So we need j in compat_list[i] and k in compat_list[j] (and k>j).
    
    # Let's refine compat_list to be strictly increasing indices.
    # The loop above does that.
    
    # Optimize check_diameter for small k?
    # It's already bit-parallel.
    
    # Let's try k=3 with a time limit per triple? No, just break on deadline.
    # To make it faster, we can skip triples where the "coverage" is poor?
    # For now, just brute force with bitmasks.
    # To reduce m, we can filter out candidates that are "dominated"?
    # Hard to define dominance for diameter.
    
    # Let's just run the k=3 search.
    # If m is ~200, m^3/6 is ~ 1.3 million triples.
    # 1.3M * check_diameter cost.
    # check_diameter cost: n=24, BFS 24 times. Each BFS is bit ops. Very fast.
    # 1.3M * (24 * 24) ~ 750M bit ops. In Python, this might be too slow (seconds to minutes).
    # We need to prune.
    
    # Pruning idea: 
    # For each candidate i, compute the "reach" or how much it reduces diameter.
    # Actually, just try the most central candidates first.
    # Sort candidates by "centrality" (e.g., sum of distances to all nodes in base graph).
    # Central edges are more likely to reduce diameter.
    
    # Calculate base distance from each node to all others
    base_dist = [[0]*n for _ in range(n)]
    for src in range(n):
        dist = [0]*n
        q = deque([src])
        visited = [False]*n
        visited[src] = True
        while q:
            u = q.popleft()
            for v in range(n):
                if base_adj[u] & (1 << v) and not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + 1
                    q.append(v)
        base_dist[src] = dist
    
    # Score candidate (u,v) by how much it helps.
    # Maybe just sum of distances reduced?
    # Or just prioritize edges that connect distant parts.
    # Let's sort candidates by |u-v| descending.
    candidates.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
    
    # Rebuild conflict masks and compat lists after sorting
    m = len(candidates)
    conflict_mask = [0] * m
    for i in range(m):
        u1, v1 = candidates[i]
        for j in range(i + 1, m):
            u2, v2 = candidates[j]
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                if u1 == u2: a, b = v1, v2
                elif u1 == v2: a, b = v1, u2
                elif v1 == u2: a, b = u1, v2
                else: a, b = u1, u2
                if base_adj[a] & (1 << b):
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
    
    compat_list = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if not (conflict_mask[i] & (1 << j)):
                compat_list[i].append(j)
    
    # Retry k=1 and k=2 with sorted candidates? 
    # The order doesn't matter for correctness, but for early termination, central edges first is good.
    
    # k=1
    for i in range(m):
        if time.time() > deadline:
            break
        if check_diameter([i]):
            return [[int(candidates[i][0]), int(candidates[i][1])]]
    
    # k=2
    for i in range(m):
        if time.time() > deadline:
            break
        for j in compat_list[i]:
            if check_diameter([i, j]):
                return [[int(candidates[i][0]), int(candidates[i][1])], 
                        [int(candidates[j][0]), int(candidates[j][1])]]
    
    # k=3
    # To speed up, limit the search to the first N candidates?
    # Or just run it and hope.
    # Let's add a time check inside the innermost loop.
    for i in range(m):
        if time.time() > deadline:
            break
        for j in compat_list[i]:
            if time.time() > deadline:
                break
            for k in compat_list[j]:
                # k > j is guaranteed by construction
                if check_diameter([i, j, k]):
                    return [[int(candidates[i][0]), int(candidates[i][1])], 
                            [int(candidates[j][0]), int(candidates[j][1])],
                            [int(candidates[k][0]), int(candidates[k][1])]]
                if time.time() > deadline:
                    break
    
    # Fallback to greedy if exact search fails or times out
    # Greedy: pick edge that reduces diameter most
    adj = [0] * n
    for i in range(n):
        adj[i] = base_adj[i]
    
    added = []
    remaining = list(range(m))
    
    for _ in range(10):
        best_idx = -1
        best_diam = 999
        for idx in remaining:
            u, v = candidates[idx]
            if adj[u] & (1 << v):
                continue
            # Check triangle free
            # If we add (u,v), triangle if common neighbor in adj[u] and adj[v]
            common = adj[u] & adj[v]
            if common:
                continue
            
            # Test diameter
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Fast diameter check
            diam = 0
            ok = True
            for src in range(n):
                visited = 1 << src
                frontier = 1 << src
                d = 0
                while frontier:
                    d += 1
                    if d > 4:
                        ok = False
                        break
                    new_frontier = 0
                    f = frontier
                    while f:
                        lsb = f & -f
                        node = lsb.bit_length() - 1
                        f ^= lsb
                        new_frontier |= (adj[node] & ~visited)
                    visited |= new_frontier
                    frontier = new_frontier
                if not ok:
                    break
                if visited != (1 << n) - 1:
                    ok = False
                    break
                diam = max(diam, d)
            
            if ok and diam < best_diam:
                best_diam = diam
                best_idx = idx
            
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            if best_diam <= 4:
                break
        
        if best_idx == -1:
            break
        
        u, v = candidates[best_idx]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        added.append((u, v))
        remaining.remove(best_idx)
        
        if best_diam <= 4:
            break
    
    if added:
        return [[int(u), int(v)] for u, v in added]
    
    # Final fallback
    return []

# EVOLVE-BLOCK-END
