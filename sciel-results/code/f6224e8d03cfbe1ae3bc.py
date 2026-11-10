import time

def construct():
    n = 15
    # Pre-compute all pairs (u, v) with u < v
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    
    # Build a lookup for edge existence using a set of frozensets or a 2D bool array
    # 2D bool array is faster for O(1) lookup
    adj = [[False] * n for _ in range(n)]
    
    # We will build the graph. Based on analysis, the maximum is 68 edges
    # with degree sequence [14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 12, 11, 10, 9, 8]
    # Vertices 0..8 have degree 14 (connected to all others)
    # Vertex 9 has degree 13 (connected to all except 14? No, let's check connectivity)
    # Vertex 10 has degree 12
    # Vertex 11 has degree 11
    # Vertex 12 has degree 10
    # Vertex 13 has degree 9
    # Vertex 14 has degree 8
    
    # Let's try to construct a specific valid graph.
    # Vertices 0-8 form a clique K9. They are connected to each other.
    # Each of 0-8 is also connected to 9, 10, 11, 12, 13, 14.
    # So 0-8 are connected to all other 14 vertices. Degree 14. Correct.
    
    # Now consider vertices 9-14.
    # Vertex 9 is connected to 0-8 (8 edges). Needs degree 13. So 5 more edges among 9-14.
    # Vertex 10 is connected to 0-8 (8 edges). Needs degree 12. So 4 more edges among 9-14.
    # Vertex 11 is connected to 0-8 (8 edges). Needs degree 11. So 3 more edges among 9-14.
    # Vertex 12 is connected to 0-8 (8 edges). Needs degree 10. So 2 more edges among 9-14.
    # Vertex 13 is connected to 0-8 (8 edges). Needs degree 9. So 1 more edge among 9-14.
    # Vertex 14 is connected to 0-8 (8 edges). Needs degree 8. So 0 more edges among 9-14.
    
    # So within the subgraph induced by {9, 10, 11, 12, 13, 14}, the degrees are [5, 4, 3, 2, 1, 0].
    # This is the path graph P6: 9-10-11-12-13. 14 is isolated within this subgraph.
    # Let's check if this works.
    # Edges in subgraph: (9,10), (10,11), (11,12), (12,13).
    # Degrees in subgraph:
    # 9: 1
    # 10: 2
    # 11: 2
    # 12: 2
    # 13: 1
    # 14: 0
    # This does NOT match [5, 4, 3, 2, 1, 0].
    
    # Let's re-evaluate the degree requirement for the subgraph H on V' = {9, 10, 11, 12, 13, 14}.
    # deg_H(9) = 5. Since |V'|=6, 9 is connected to all other 5 vertices in V'.
    # deg_H(10) = 4. 10 is not connected to one vertex in V'.
    # deg_H(11) = 3.
    # deg_H(12) = 2.
    # deg_H(13) = 1.
    # deg_H(14) = 0. 14 is isolated in H.
    
    # If 14 is isolated in H, it has no edges to 9, 10, 11, 12, 13.
    # So the subgraph on {9, 10, 11, 12, 13} must have degrees [5, 4, 3, 2, 1].
    # Wait, 9 is connected to all 5 others in V'. So 9 is connected to 10, 11, 12, 13, 14.
    # But 14 is isolated in H, so 9 is NOT connected to 14. Contradiction.
    
    # Therefore, 14 cannot be isolated in H if 9 needs to connect to it.
    # Let's find a graphical sequence for the subgraph H with degrees [5, 4, 3, 2, 1, 0].
    # Havel-Hakimi:
    # Sort: [5, 4, 3, 2, 1, 0]
    # Remove 5. Connect to 5 others.
    # New: [3, 2, 1, 0, 0, -1]. Negative! Not graphical.
    
    # So the sequence [14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 12, 11, 10, 9, 8] is NOT graphical.
    # My previous manual check was flawed.
    
    # Let's try to find the actual maximum.
    # The constraint is: for any path s-u-v-t, deg(s) != deg(t).
    # This implies that if two vertices s and t have the same degree, there can be NO path of length 3 between them.
    # In a dense graph, it's hard to avoid paths of length 3.
    # If deg(s) = deg(t), then s and t must be at distance > 3 or distance 1 (but s!=t and simple path implies distinct vertices, so distance 1 is not a path of length 3).
    # Actually, if s and t are adjacent, a path s-u-v-t requires u, v distinct from s, t.
    # If the graph is complete, any two vertices have many paths of length 3.
    # So we need to structure the graph such that vertices of the same degree are "far apart" in terms of 3-path connectivity.
    
    # A simple strategy: Make the graph bipartite?
    # If the graph is bipartite, paths of length 3 go from Part A to Part B.
    # If we assign degrees such that all vertices in Part A have distinct degrees, and all vertices in Part B have distinct degrees, AND no degree appears in both parts, then we are safe?
    # No, if s in A and t in B have the same degree, and there is a path s-u-v-t, it's a violation.
    # In a bipartite graph, s-u-v-t is a valid path.
    # So we need deg(s) != deg(t) for any s in A, t in B that are connected by a 3-path.
    # In a complete bipartite graph K_{n1, n2}, any s in A and t in B have paths of length 3 (if n1, n2 >= 2).
    # So in a complete bipartite graph, all degrees in A must be distinct from all degrees in B.
    # Degrees in A are all n2. Degrees in B are all n1.
    # So we need n1 != n2.
    # Max edges in bipartite graph on 15 vertices is floor(15^2 / 4) = 56.
    # This is better than 14.
    
    # Can we do better than bipartite?
    # If we add edges within parts, we create odd cycles and paths of length 3 within the same part.
    # If s, t in A, and there is a path s-u-v-t, then u in B, v in A? No.
    # Path s(A) - u(B) - v(A) - t(B). Endpoints s in A, t in B.
    # Path s(A) - u(B) - v(B) - t(A). Endpoints s in A, t in A.
    # So edges within parts create paths of length 3 between vertices in the same part.
    # This makes the constraint stricter.
    
    # Let's stick with a complete bipartite graph first.
    # Split 15 into 7 and 8.
    # Part A: 0..6 (7 vertices)
    # Part B: 7..14 (8 vertices)
    # Edges: all (u, v) with u in A, v in B.
    # Total edges: 7 * 8 = 56.
    # Degrees:
    # Vertices in A have degree 8.
    # Vertices in B have degree 7.
    # Check constraint:
    # Path s-u-v-t.
    # If s in A, t in A: path is A-B-B-A.
    #   s(A)-u(B)-v(B)-t(A).
    #   deg(s) = 8, deg(t) = 8.
    #   This is a violation!
    #   So complete bipartite K_{7,8} is INVALID because paths of length 3 exist between vertices in the same part, and they have the same degree.
    
    # To fix this, we need to ensure that for any path s-u-v-t, deg(s) != deg(t).
    # In a bipartite graph, paths of length 3 connect vertices in the same part.
    # So all vertices in Part A must have distinct degrees?
    # And all vertices in Part B must have distinct degrees?
    # If s, t in A have distinct degrees, then deg(s) != deg(t) is satisfied.
    # So if we construct a bipartite graph where degrees within each part are all distinct, it is valid.
    
    # Max edges in such a graph?
    # Let Part A have size n1, Part B have size n2. n1 + n2 = 15.
    # Degrees in A must be distinct. Max degree in A is n2. Min is 0.
    # So n1 <= n2 + 1.
    # Degrees in B must be distinct. Max degree in B is n1. Min is 0.
    # So n2 <= n1 + 1.
    # This implies |n1 - n2| <= 1.
    # So n1 = 7, n2 = 8 or n1 = 8, n2 = 7.
    
    # Let's try n1 = 7, n2 = 8.
    # Degrees in A: 0..8 (distinct). But max degree is 8.
    # Can we assign degrees 1, 2, ..., 8? (Cannot be 0 if we want max edges? Maybe.)
    # Degrees in B: 0..7 (distinct). Max degree 7.
    # Can we assign degrees 1, 2, ..., 7?
    
    # We want to maximize sum of degrees.
    # Sum_A = sum of distinct values <= 8. Best is 1+2+...+7+8? No, there are 7 vertices.
    # So degrees in A: 7, 6, 5, 4, 3, 2, 1. Sum = 28.
    # Degrees in B: 6, 5, 4, 3, 2, 1, 0? No, 8 vertices.
    # Degrees in B: 7, 6, 5, 4, 3, 2, 1, 0. Sum = 28.
    # Total Sum = 56. Edges = 28.
    # This is much worse than 56 edges in complete bipartite (which was invalid).
    
    # Can we do better?
    # What if we don't use distinct degrees, but ensure that for any pair (s,t) in the same part, they are NOT connected by a 3-path?
    # In a bipartite graph, s and t in A are connected by a 3-path if there exist u, v in B such that s-u, u-v, v-t are edges.
    # This requires u adjacent to both s and v? No.
    # s-u-v-t means s~u, u~v, v~t.
    # u, v are in B. So u~v is an edge in B? No, B is independent in bipartite graph.
    # Wait, in a bipartite graph, there are NO edges within B.
    # So u~v is impossible.
    # Therefore, in a STRICT bipartite graph, there are NO paths of length 3.
    # Paths have length 1 (A-B), 2 (A-B-A), 3 (A-B-B-A) - but B-B edge doesn't exist.
    # So max path length is 2?
    # No, path is a sequence of vertices v0, v1, v2... with edges (vi, vi+1).
    # In bipartite graph, v0 in A, v1 in B, v2 in A, v3 in B.
    # So path of length 3 is A-B-A-B.
    # My previous analysis "A-B-B-A" was wrong.
    # Path s-u-v-t:
    # s in A, u in B, v in A, t in B.
    # So s and t are in DIFFERENT parts.
    # So the constraint applies to pairs (s,t) where s in A, t in B.
    # We need deg(s) != deg(t) for any s in A, t in B that are connected by a 3-path.
    # In a complete bipartite graph, any s in A and t in B are connected by many 3-paths (e.g. s-u-v-t where u, v are other vertices in B, A respectively? No.
    # s(A)-u(B)-v(A)-t(B).
    # u is in B, v is in A.
    # Edges: s~u (A-B), u~v (B-A), v~t (A-B).
    # This path exists if s~u, u~v, v~t.
    # In K_{7,8}, all cross edges exist.
    # So for any s in A, t in B, we can pick any u in B (u!=s? s in A, u in B, so u!=s always) and any v in A (v!=t? t in B, v in A, so v!=t always).
    # We need u~v. In K_{7,8}, u~v always exists.
    # So ANY s in A and t in B are connected by a 3-path.
    # Therefore, in a complete bipartite graph, we need deg(s) != deg(t) for ALL s in A, t in B.
    # deg(s) is constant for all s in A (equals n2).
    # deg(t) is constant for all t in B (equals n1).
    # So we need n1 != n2.
    # If n1 != n2, then deg(s) = n2 and deg(t) = n1, so deg(s) != deg(t).
    # So ANY complete bipartite graph K_{n1, n2} with n1 != n2 is VALID.
    
    # Max edges is max(n1 * n2) subject to n1 + n2 = 15, n1 != n2.
    # n1 = 7, n2 = 8 -> 56 edges.
    # n1 = 6, n2 = 9 -> 54 edges.
    # So 56 is the best for bipartite.
    
    # Can we exceed 56?
    # If we add an edge within A, say (a1, a2).
    # This creates paths of length 3 involving a1, a2.
    # Example: path t(A)-u(B)-a1(A)-a2(A)? No, a1-a2 is edge.
    # Path t(B)-u(A)-a1(A)-v(B)?
    # Let's check if adding an edge (a1, a2) in A creates a violation.
    # It creates new 3-paths.
    # One such path: b1(B)-a1(A)-a2(A)-b2(B).
    # Endpoints b1 in B, b2 in B.
    # We need deg(b1) != deg(b2).
    # In the base bipartite graph, all b in B have degree n1.
    # If we just add (a1, a2), deg(b) remains n1 for all b.
    # So deg(b1) = deg(b2) = n1.
    # This is a violation!
    # So we cannot simply add edges within parts without adjusting degrees.
    
    # To add an edge (a1, a2), we must ensure that for any 3-path ending in B, the endpoints have different degrees.
    # This likely requires making degrees in B non-uniform.
    # This gets complex. 56 is a solid baseline.
    
    # Let's construct K_{7,8}.
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    edges = []
    for u in part_a:
        for v in part_b:
            edges.append([u, v])
            
    return edges
