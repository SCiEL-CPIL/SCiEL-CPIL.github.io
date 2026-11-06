"""Bitset-based backtracking solver for R(K2,2,2) > 30."""
import numpy as np
from itertools import combinations

def construct():
    """
    Attempt to find a 2-coloring of K_30 with no monochromatic K_2,2,2.
    Uses a backtracking approach with pruning based on the K_2,2,2-free constraint.
    
    K_2,2,2 is a complete tripartite graph with parts of size 2.
    A coloring contains a monochromatic K_2,2,2 if there exist 6 vertices
    partitioned into 3 pairs such that all 12 edges between different pairs
    have the same color.
    
    Equivalently, for any 3 vertices a,b,c, if we look at the common neighbors
    in color c (vertices adjacent to all of a,b,c in color c), then those common
    neighbors must not contain a K_2,2 (i.e., two vertices both connected to 
    two others). Actually, let me think more carefully.
    
    K_2,2,2 has parts A={a1,a2}, B={b1,b2}, C={c1,c2}. Edges exist between
    all pairs from different parts. So a monochromatic K_2,2,2 in color c means
    there are 6 vertices partitioned into 3 pairs such that all cross-pair edges
    are color c.
    
    This is equivalent to: there exist 3 disjoint pairs (a1,a2), (b1,b2), (c1,c2)
    such that for all x in {a1,a2}, y in {b1,b2}, z in {c1,c2} with x,y from
    different pairs, edge(x,y) has color c.
    
    A simpler characterization: A graph G contains K_2,2,2 as a subgraph iff
    there exist 3 vertices u,v,w such that the common neighborhood N(u) ∩ N(v) ∩ N(w)
    contains at least 3 vertices that can be partitioned into... hmm, this is getting complex.
    
    Let me use a different approach. For a graph G, G contains K_2,2,2 iff there exist
    three disjoint pairs of vertices such that every vertex in one pair is adjacent to
    every vertex in each of the other two pairs.
    
    For efficient checking during backtracking, when we assign a color to edge (i,j),
    we need to check that neither color class contains K_2,2,2.
    
    Key insight: If we fix three vertices a,b,c all mutually connected in color c,
    then any vertex d that is connected to all of a,b,c in color c, together with another
    vertex e also connected to all of a,b,c in color c, where d and e are NOT connected
    in color c... wait, that's not right either.
    
    Let me think about this differently. K_2,2,2 is the complete tripartite graph T(2,2,2).
    It has 6 vertices. A subgraph isomorphic to K_2,2,2 in color c means we can find
    6 vertices v1,...,v6 and a partition into 3 pairs such that all edges between
    different pairs have color c.
    
    For the backtracking solver, I'll use a simpler but effective pruning:
    - Maintain the current coloring as bitmasks for each color.
    - When adding a new edge, check if it creates a K_2,2,2 in either color.
    - To check efficiently: for each triple of vertices (a,b,c) that are mutually
      connected in color c, count how many vertices are connected to all three in color c.
      If that count is >= 4, then among those 4 vertices, by pigeonhole or by checking,
      we might have a K_2,2,2. Actually, we need 2 more vertices that form a pair
      with each other NOT in color c... no wait.
    
    Actually, let me reconsider. The condition for K_2,2,2: we need 3 pairs. 
    If we have vertices a,b,c all mutually connected in color c, and we have 4 vertices
    d1,d2,d3,d4 all connected to a,b,c in color c, then if any two of d1,d2,d3,d4 are
    NOT connected in color c, we can't immediately form K_2,2,2 with just those. 
    But if we take d1,d2 as one pair, we need another pair from the remaining vertices
    that are also connected to a,b,c. 
    
    This is getting complicated. Let me just implement a direct check for K_2,2,2
    in the current graph, but only on the vertices that have been fully assigned.
    
    Given the time constraint (120s) and the difficulty of finding a 30-vertex coloring,
    I'll implement a backtracking solver that tries to build a valid coloring up to n=30,
    with aggressive pruning. If it fails within the time limit, I'll fall back to the
    known 29-vertex coloring.
    """
    
    # First, let's try to construct a 29-vertex coloring using a known method,
    # then attempt to extend it to 30 vertices.
    
    # Known: R(K_2,2,2) > 29, so a 29-vertex coloring exists.
    # Let's use a greedy/backtracking approach to find one.
    
    def contains_k222(color_graph, n):
        """Check if the graph with adjacency matrix color_graph (n x n, symmetric, 0/1)
        contains K_2,2,2 as a subgraph."""
        # K_2,2,2: 6 vertices partitioned into 3 pairs, all cross-pair edges present.
        # For each choice of 3 vertices a,b,c that form a triangle, 
        # find common neighbors, and check if any 4 common neighbors contain
        # a K_2,2 in the complement... 
        # 
        # Simpler: iterate over all 6-subsets and check if any is K_2,2,2.
        # This is O(C(n,6) * 36) which for n=30 is C(30,6)=593775, times 36 checks = ~21M.
        # Too slow for repeated calls.
        
        # Better approach: For each triple (a,b,c) forming a triangle in the graph,
        # let C = common neighbors of a,b,c. If |C| >= 4, then we need to check
        # if C contains an independent set of size 2 (in the original graph, i.e.,
        # two non-adjacent vertices). If so, take those two as one pair, and we need
        # another pair from C that is also non-adjacent... wait, no.
        
        # Let me re-examine. K_2,2,2 = T(2,2,2). The vertices can be labeled
        # (a1,a2), (b1,b2), (c1,c2) where ai-bj and ai-cj and bj-cj edges all exist.
        # 
        # If we fix the "center" triangle a1,b1,c1 (all mutually adjacent), then
        # a2 must be adjacent to b1,c1 (and a1), b2 must be adjacent to a1,c1 (and b1),
        # c2 must be adjacent to a1,b1 (and c1). Also a2-b2, a2-c2, b2-c2 must exist.
        #
        # So: pick a triangle (a,b,c). Let N_a = neighbors of a excluding b,c.
        # Let N_b = neighbors of b excluding a,c.
        # Let N_c = neighbors of c excluding a,b.
        # We need a' in N_a ∩ N_b ∩ N_c, b' in N_a ∩ N_b ∩ N_c, c' in N_a ∩ N_b ∩ N_c
        # such that a' is adjacent to b',c' (already true if in N_b ∩ N_c), 
        # b' adjacent to a',c', c' adjacent to a',b'. And a',b',c' form a triangle among themselves?
        # No! In K_2,2,2, a2 is NOT adjacent to b2 or c2... wait, yes it is!
        # In K_2,2,2, ALL cross-pair edges exist. So a2-b2 IS an edge.
        # 
        # So actually, if a', b', c' are all in the common neighborhood of a,b,c,
        # and a',b',c' form a triangle, then {a,b,c,a',b',c'} contains K_2,2,2?
        # Let's verify: pairs are (a,a'), (b,b'), (c,c'). 
        # a-a' : yes (a' in N_a)
        # b-b' : yes
        # c-c' : yes
        # a-b : yes (triangle)
        # a-c : yes
        # b-c : yes
        # a'-b' : yes (triangle among a',b',c')
        # a'-c' : yes
        # b'-c' : yes
        # a-b' : yes (a' in N_b means a' adjacent to b, but we need a adjacent to b')
        #   b' is in N_a, so b' adjacent to a. Yes!
        # Similarly for all other cross edges.
        # So YES: if there's a triangle (a,b,c) and a triangle (a',b',c') in the common
        # neighborhood, with a' adjacent to b,c; b' adjacent to a,c; c' adjacent to a,b,
        # then we have K_2,2,2.
        
        # Even simpler: if the common neighborhood of any triangle contains a triangle,
        # we have K_2,2,2.
        
        for a in range(n):
            for b in range(a+1, n):
                if not color_graph[a][b]:
                    continue
                for c in range(b+1, n):
                    if not color_graph[a][c] or not color_graph[b][c]:
                        continue
                    # a,b,c form a triangle
                    # Find common neighbors
                    common = []
                    for v in range(n):
                        if v == a or v == b or v == c:
                            continue
                        if color_graph[a][v] and color_graph[b][v] and color_graph[c][v]:
                            common.append(v)
                    if len(common) < 3:
                        continue
                    # Check if common contains a triangle
                    for i in range(len(common)):
                        for j in range(i+1, len(common)):
                            if not color_graph[common[i]][common[j]]:
                                continue
                            for k in range(j+1, len(common)):
                                if color_graph[common[i]][common[k]] and color_graph[common[j]][common[k]]:
                                    return True
        return False
    
    # This is too slow for backtracking. Let me use a different strategy.
    
    # Given the difficulty, let me try to use a known construction for n=29
    # and verify it, then attempt extension to 30.
    
    # For now, let's return a valid 29-vertex coloring using a systematic approach.
    
    # Use a backtracking solver with good heuristics to find a 29-vertex coloring.
    
    import time
    start_time = time.time()
    time_limit = 110  # Leave 10 seconds for cleanup
    
    def solve(n, timeout=110):
        """Backtracking solver to find a 2-coloring of K_n with no mono K_2,2,2."""
        # colors[i][j] for i<j, -1 means unassigned
        colors = [[-1] * n for _ in range(n)]
        
        # Precompute list of edges in order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_idx = 0
        
        # For pruning: maintain adjacency sets for each color
        adj0 = [set() for _ in range(n)]
        adj1 = [set() for _ in range(n)]
        
        def check_k222(adj, n):
            """Check if graph with adjacency sets contains K_2,2,2."""
            # For each triangle, check if common neighborhood contains a triangle
            for a in range(n):
                for b in adj[a]:
                    if b <= a:
                        continue
                    for c in adj[a]:
                        if c <= b:
                            continue
                        if c not in adj[b]:
                            continue
                        # a,b,c triangle
                        common = adj[a] & adj[b] & adj[c]
                        common -= {a, b, c}
                        if len(common) < 3:
                            continue
                        # Check for triangle in common
                        cl = list(common)
                        for i in range(len(cl)):
                            for j in range(i+1, len(cl)):
                                if cl[j] not in adj[cl[i]]:
                                    continue
                                for k in range(j+1, len(cl)):
                                    if cl[k] in adj[cl[i]] and cl[k] in adj[cl[j]]:
                                        return True
            return False
        
        def backtrack():
            nonlocal edge_idx
            if time.time() - start_time > timeout:
                return False
            if edge_idx >= len(edges):
                return True
            
            i, j = edges[edge_idx]
            edge_idx += 1
            
            # Try color 0
            colors[i][j] = colors[j][i] = 0
            adj0[i].add(j)
            adj0[j].add(i)
            
            if not check_k222(adj0, n):
                if backtrack():
                    return True
            
            adj0[i].discard(j)
            adj0[j].discard(i)
            
            # Try color 1
            colors[i][j] = colors[j][i] = 1
            adj1[i].add(j)
            adj1[j].add(i)
            
            if not check_k222(adj1, n):
                if backtrack():
                    return True
            
            adj1[i].discard(j)
            adj1[j].discard(i)
            
            colors[i][j] = colors[j][i] = -1
            edge_idx -= 1
            return False
        
        if backtrack():
            result = []
            for i in range(n):
                for j in range(i+1, n):
                    result.append(colors[i][j])
            return {'n': n, 'colors': result}
        return None
    
    # Try n=30 first
    result = solve(30, timeout=100)
    if result is not None:
        return result
    
    # Fall back to n=29
    result = solve(29, timeout=50)
    if result is not None:
        return result
    
    # If backtracking fails, use a heuristic construction
    # Return a simple valid coloring for small n
    # For n=17, use Paley
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}

# EVOLVE-BLOCK-END
