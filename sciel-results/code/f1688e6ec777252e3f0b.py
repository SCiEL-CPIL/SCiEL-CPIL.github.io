import time

def construct():
    # G6: C5 (0-4) plus vertex 5 connected to 0 and 2.
    # Edges:
    g6_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 0], [5, 2]
    ]
    n6 = 6
    m6 = len(g6_edges)
    
    # Mycielski Construction M(G6)
    # Vertices: 0..5 (V), 6..11 (V'), 12 (w)
    # Edges:
    # 1. Original edges E(G)
    # 2. For each edge (u, v) in E(G), add (u+6, v) and (v+6, u)? 
    #    Standard Mycielski: (u', v) for (u, v) in E(G).
    #    So if (0, 1) in E, add (6, 1).
    # 3. w (12) connected to all u' (6..11).
    
    edges = list(g6_edges)
    
    # Map original edges to primed edges
    for u, v in g6_edges:
        # u' is u+6
        # Edge (u', v) => (u+6, v)
        edges.append([u + 6, v])
        # Note: Mycielski usually adds (u', v) for each undirected edge (u, v).
        # Since (u, v) and (v, u) are the same, we add one edge per undirected edge?
        # Definition: M(G) has V(G) U V'(G) U {w}.
        # E(M(G)) = E(G) U { (u', v) | (u, v) in E(G) } U { (w, u') | u in V(G) }.
        # So for each edge [u, v] in E(G), we add [u+6, v].
        # We do NOT add [v+6, u] unless [v, u] is also in E(G) as a separate entry?
        # No, E(G) is a set of undirected edges.
        # So we add one edge per undirected edge in E(G).
        
    # Correction: The loop above adds (u+6, v) for each [u, v].
    # If [u, v] is in g6_edges, we add [u+6, v].
    # Do we need [v+6, u]?
    # The definition says { (u', v) | (u, v) in E(G) }.
    # If E(G) is undirected, (u, v) is the same as (v, u).
    # So we add exactly one edge per edge in G.
    
    # Add w connections
    for u in range(n6):
        edges.append([12, u + 6])
        
    # Now we have 13 vertices (0..12), edges count: m6 + m6 + n6 = 2*m6 + n6.
    # m6 = 7. Edges = 14 + 6 = 20.
    
    # We need 16 vertices. Add 3: 13, 14, 15.
    # Connect them to an independent set in M(G6).
    
    # Find independent set in M(G6) (vertices 0..12)
    # Simple greedy or exact?
    # For now, let's use a known large independent set.
    # Or just use all vertices 0..5? No, not independent.
    
    # Let's try to find a large independent set in M(G6) using backtracking or greedy.
    # Given time limits, greedy might be enough if we know a good one.
    # Alpha(G6) = 3.
    # Independent set in M(G6) can be size 9.
    
    # Let's hardcode a likely large independent set or compute it.
    # Vertices 0..5 is V. 6..11 is V'. 12 is w.
    
    # Independent set S:
    # Take 1, 3, 5 from V?
    # 1-3 not edge. 1-5 not edge. 3-5 not edge.
    # So {1, 3, 5} is independent in V.
    # Corresponding in V': {7, 9, 11}.
    # Are there edges between {1, 3, 5} and {7, 9, 11}?
    # Edge (u', v) exists if (u, v) in E(G).
    # 7 is 1'. Edges from 7 to V: (7, v) if (1, v) in E(G).
    # N(1) = {0, 2}. So 7 connected to 0, 2.
    # 9 is 3'. N(3) = {2, 4}. So 9 connected to 2, 4.
    # 11 is 5'. N(5) = {0, 2}. So 11 connected to 0, 2.
    
    # If we pick S_V = {1, 3, 5} and S_Vp = {7, 9, 11}.
    # Check edges between S_V and S_Vp.
    # 1 connected to 7? (7, 1) exists if (1, 1) in E? No.
    # Edge is (u', v). So 7 connected to v if (1, v) in E.
    # Is 1 in S_V? Yes. Is 1 connected to 7?
    # 7 is 1'. Edge (1', 1) exists if (1, 1) in E? No.
    # Edge (1', v) for v in N(1). N(1) = {0, 2}.
    # So 7 is connected to 0, 2. Not to 1, 3, 5.
    # Similarly, 9 (3') connected to 2, 4. Not to 1, 3, 5.
    # 11 (5') connected to 0, 2. Not to 1, 3, 5.
    
    # So S = {1, 3, 5, 7, 9, 11} is independent?
    # Check within S_Vp: 7-9? (1', 3') no edge.
    # So yes, S of size 6 is independent.
    
    # Can we add more?
    # What about 12? 12 connected to 7, 9, 11. So no.
    # What about 0, 2, 4?
    # 0 connected to 7, 11. So no.
    # 2 connected to 7, 9, 11. So no.
    # 4 connected to 9. So no.
    
    # So max independent set might be 6?
    # Or maybe we can pick different ones.
    # If we pick S_V = {0, 2, 4}?
    # 0-2 no. 0-4 edge! No.
    
    # Let's stick with size 6.
    # If alpha is 6, we add 3 * 6 = 18 edges.
    # Total edges: 20 + 18 = 38.
    
    # Wait, earlier I estimated 47 based on alpha=9.
    # If alpha is only 6, then 38 is the score.
    # 38 is still better than 20.
    
    # Let's try to find a larger independent set programmatically in the code.
    
    # Add 3 vertices 13, 14, 15 connected to S.
    S = [1, 3, 5, 7, 9, 11]
    
    for v in [13, 14, 15]:
        for u in S:
            edges.append([v, u])
            
    return edges

