import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85
    
    # Precompute base adjacency as bitsets
    base_adj_bits = [0] * n
    for u, v in base_edges:
        base_adj_bits[u] |= (1 << v)
        base_adj_bits[v] |= (1 << u)
        
    # Precompute coverage bitsets for each potential star center s
    # s covers s-1, s, s+1
    cover_bits = [0] * n
    for s in range(n):
        mask = 0
        for v in [s-1, s, s+1]:
            if 0 <= v < n:
                mask |= (1 << v)
        cover_bits[s] = mask

    def is_triangle_free(c, S):
        # S is a list of ints
        # Check if any s in S is adjacent to c in base graph (forbidden edge)
        for s in S:
            if base_adj_bits[c] & (1 << s):
                return False
        # Check if any s in S is adjacent to any neighbor of c (triangle c-nbr-s)
        # Neighbors of c are in base_adj_bits[c]
        nbrs_c = base_adj_bits[c]
        for s in S:
            # If s is adjacent to any neighbor of c, triangle exists
            if base_adj_bits[s] & nbrs_c:
                return False
        # Check if S is independent (no two stars adjacent)
        # If s1, s2 in S and adjacent, then c-s1-s2-c is a 4-cycle, not a triangle.
        # Wait, triangle is 3-cycle. c-s1-s2-c is not a triangle unless s1-s2 is an edge and s1-c, s2-c are edges.
        # The edges are (c,s1) and (c,s2). The path edge is (s1,s2).
        # This forms a triangle c-s1-s2-c ONLY IF s1 and s2 are adjacent?
        # No, a triangle is 3 vertices with 3 edges.
        # Vertices: c, s1, s2.
        # Edges: (c,s1), (c,s2), (s1,s2).
        # If s1 and s2 are adjacent in the base graph, and we add (c,s1) and (c,s2), we form a triangle.
        # So S MUST be an independent set.
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if base_adj_bits[S[i]] & (1 << S[j]):
                    return False
        return True

    def check_diameter(c, S):
        # Build adjacency bitsets
        adj = base_adj_bits[:]
        for s in S:
            adj[c] |= (1 << s)
            adj[s] |= (1 << c)
            
        # BFS from c to get distances
        dist = [-1] * n
        dist[c] = 0
        queue = deque([c])
        while queue:
            u = queue.popleft()
            neighbors = adj[u]
            # Iterate through bits
            # For speed, we can just check all v in 0..n-1
            for v in range(n):
                if (neighbors >> v) & 1:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
        
        # Check if all reached
        if -1 in dist:
            return False
            
        # Check diameter
        # For each u, we can BFS again? O(N^2) = 24^2 = 576. Fast.
        # Or just check pairs.
        for u in range(n):
            # If dist[u] is large, check pairs with others
            # Actually, just do BFS from each u? 24 BFS is fine.
            # But we can optimize: if max dist from c is D_c, then d(u,v) <= D_c(u) + D_c(v).
            # We need d(u,v) <= 4.
            # So if D_c(u) + D_c(v) > 4, we must check if d_path(u,v) <= 4.
            # Since base is a path, d_path(u,v) = abs(u-v).
            for v in range(u+1, n):
                if dist[u] + dist[v] <= 4:
                    continue
                if abs(u - v) <= 4:
                    continue
                # Otherwise, need to check if there's a shorter path via other nodes?
                # The shortest path is min(d_path, d_via_c, d_via_other).
                # d_via_c = dist[u] + dist[v].
                # d_via_other? In a tree + chords, the shortest path is unique?
                # The graph is a tree (path) plus star edges.
                # It's not a tree anymore.
                # To be safe, run BFS from u?
                # Given N=24, running BFS from each u is 24 * (24+23) ops ~ 1000 ops.
                # Very fast.
                # Let's just do full BFS from each u to be absolutely sure.
                # But we can prune: if dist[u] + dist[v] > 4 and abs(u-v) > 4, 
                # is it possible that a path via some other s' is shorter?
                # d(u, s') + d(s', v).
                # d(u, s') <= d(u, c) + 1 = dist[u] + 1.
                # d(s', v) <= 1 + dist[v].
                # So d(u,v) <= dist[u] + 2 + dist[v]. This is worse.
                # The only shortcut is via c.
                # So d(u,v) = min(abs(u-v), dist[u] + dist[v]).
                # So the condition is: min(abs(u-v), dist[u] + dist[v]) <= 4.
                # If this fails, return False.
                if min(abs(u - v), dist[u] + dist[v]) > 4:
                    return False
        return True

    best_solution = None
    best_size = 8 # Start with 7, try 6
    
    # Try to find solution with 6 edges
    target_sizes = [6, 7]
    
    for k in target_sizes:
        for c in range(n):
            if time.time() > deadline:
                break
            
            # Vertices that MUST be covered by a star to have dist <= 2 from c
            # If d_path(c, v) <= 2, dist[v] <= 2 is guaranteed by path.
            # If d_path(c, v) > 2, we need v to be in N(s) for some s in S to get dist[v] <= 2.
            # If we don't cover it, dist[v] >= 3.
            # If dist[v] >= 3, we need to ensure d(v, u) <= 4 for all u.
            # For simplicity, let's first try to cover ALL v with d_path(c, v) > 2.
            
            far_mask = 0
            for v in range(n):
                # Calculate path distance from c to v
                d = abs(v - c)
                if d > 2:
                    far_mask |= (1 << v)
            
            # Candidates for S
            # s cannot be c
            # s cannot be adjacent to c (existing edge)
            # s cannot be at dist 2 from c (triangle with c's neighbor)
            # So s must have d_path(c, s) >= 3?
            # Let's check triangle condition:
            # If s is adjacent to c, edge exists. Forbidden.
            # If s is adjacent to n (where n adjacent to c), triangle c-n-s.
            # n is c-1 or c+1.
            # s adjacent to c-1 means s is c-2 or c.
            # So s cannot be c-2, c-1, c, c+1, c+2.
            # So d_path(c, s) >= 3.
            
            candidates = []
            for s in range(n):
                if abs(s - c) < 3:
                    continue
                # Check if s is in far_mask?
                # If s is not in far_mask, it means d_path(c,s) <= 2.
                # But we excluded s with d < 3. So all candidates are in far_mask.
                # Actually, we just need s to be a valid star center.
                candidates.append(s)
            
            # We need to pick k candidates from 'candidates' such that:
            # 1. They form an independent set.
            # 2. Their coverage covers 'far_mask'.
            # 3. The resulting graph has diameter <= 4.
            
            # Sort candidates by coverage of far_mask (greedy hint)
            def cov_size(s):
                return (cover_bits[s] & far_mask).bit_count()
            
            candidates.sort(key=cov_size, reverse=True)
            
            # Backtracking
            def backtrack(idx, chosen, covered_mask):
                nonlocal best_solution
                if time.time() > deadline:
                    return False
                
                if len(chosen) == k:
                    if (covered_mask & far_mask) == far_mask:
                        # Check triangle-free and diameter
                        if is_triangle_free(c, chosen) and check_diameter(c, chosen):
                            best_solution = [(c, s) for s in chosen]
                            return True
                    return False
                
                if idx >= len(candidates):
                    return False
                
                # Pruning: if remaining candidates cannot cover remaining far_mask
                rem_cover = 0
                for i in range(idx, len(candidates)):
                    rem_cover |= cover_bits[candidates[i]]
                if (covered_mask | rem_cover) & far_mask != far_mask:
                    return False
                
                # Option 1: Skip
                if backtrack(idx + 1, chosen, covered_mask):
                    return True
                
                # Option 2: Take
                s = candidates[idx]
                # Check independence
                independent = True
                for w in chosen:
                    if base_adj_bits[s] & (1 << w):
                        independent = False
                        break
                if independent:
                    new_covered = covered_mask | cover_bits[s]
                    if backtrack(idx + 1, chosen + [s], new_covered):
                        return True
                
                return False
            
            if backtrack(0, [], 0):
                break
        
        if best_solution is not None:
            break

    if best_solution is None:
        return []
        
    return [[int(u), int(v)] for u, v in best_solution]
