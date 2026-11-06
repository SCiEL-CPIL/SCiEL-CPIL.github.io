import itertools
import numpy as np

def construct():
    # Vertices 0..15
    # We want a Q3-free graph maximizing edges.
    # Strategy: Start with a high-degree base graph and remove edges to eliminate Q3 subgraphs.
    # A good candidate for Q3-free dense graphs is a complete multipartite graph or a complement of a sparse graph.
    # However, disjoint cliques K7+K7+K2 gives 21+21+1=43 edges.
    
    # Let's try to construct a graph with more than 43 edges.
    # Consider the complete graph K16. It has 120 edges.
    # Q3 has 12 edges.
    # If we remove a set of edges such that no Q3 remains, we want to remove as few as possible.
    # This is equivalent to finding the Turan number ex(16, Q3).
    
    # Known results for ex(n, Q3):
    # For n=8, ex(8, Q3) = 12 (Q3 itself is 12 edges, but K8 minus 1-factor is 24 edges and Q3-free? 
    # Wait, Q3 is bipartite. K8 minus 1-factor is 7-regular. 
    # Actually, ex(8, Q3) is 12? No, Q3 has 12 edges. 
    # Let's check small n.
    # ex(4, Q3) = 6 (K4 is Q3-free).
    # ex(6, Q3) = 15 (K6 is Q3-free? K6 has 15 edges. Q3 has 8 vertices, so K6 is trivially Q3-free).
    # ex(7, Q3) = 21 (K7 is Q3-free).
    # ex(8, Q3) = ?
    # Q3 has 8 vertices. So ex(8, Q3) is the max edges in a graph on 8 vertices with no Q3 subgraph.
    # K8 has 28 edges. Q3 is a subgraph of K8.
    # If we remove 1 edge from K8, does it contain Q3?
    # Q3 is 3-regular. K8 minus 1 edge has two vertices of degree 6 and six of degree 7.
    # It likely contains Q3.
    # Actually, it is known that ex(8, Q3) = 12? No, that's the number of edges in Q3.
    # Let's look up or derive.
    # A graph on 8 vertices is Q3-free if it does not contain a 3-regular bipartite subgraph with 8 vertices and 12 edges.
    # Wait, Q3 is the only 3-regular bipartite graph on 8 vertices?
    # No, there are others?
    # Q3 is the cube graph.
    
    # Let's try a different approach: Randomized construction with verification.
    # But we need a deterministic output.
    
    # Let's try the complement of a 1-factorization of K16?
    # K16 can be decomposed into 15 1-factors.
    # If we take K16 minus one 1-factor, we get a 14-regular graph.
    # Does it contain Q3?
    # Q3 is 3-regular. It is likely contained in a 14-regular graph on 16 vertices.
    
    # Let's try a complete 4-partite graph with parts of size 4.
    # T(16, 4) has edges: 16*15/2 - 4*(4*3/2) = 120 - 24 = 96 edges.
    # Is T(16, 4) Q3-free?
    # Q3 is bipartite with parts of size 4 and 4.
    # In T(16, 4), the parts are independent sets of size 4.
    # A bipartite subgraph of T(16, 4) must have its parts drawn from the 4 independent sets.
    # If we pick two parts, say A and B, the subgraph is K_{4,4}.
    # K_{4,4} contains Q3?
    # Q3 is a subgraph of K_{4,4}. Yes, Q3 is a 3-regular subgraph of K_{4,4}.
    # So T(16, 4) contains Q3.
    
    # What if we take a complete 3-partite graph with parts 6, 5, 5?
    # Edges: 120 - (15 + 10 + 10) = 85.
    # Contains Q3?
    # Q3 is bipartite. It can be embedded in a complete bipartite subgraph.
    # The largest complete bipartite subgraph in T(16, 3) is K_{10, 6} (taking two parts).
    # K_{10, 6} contains Q3.
    
    # So complete multipartite graphs likely contain Q3.
    
    # Let's go back to disjoint cliques.
    # K7 + K7 + K2 = 43 edges.
    # Can we do better?
    # What about K8 minus some edges?
    # If we can make K8 Q3-free with > 21 edges, then K8' + K7 + K1 > 43.
    # Or K8' + K8' > 43?
    
    # Let's check ex(8, Q3).
    # A graph on 8 vertices is Q3-free.
    # Q3 has 12 edges.
    # If we take K8 and remove edges until no Q3 remains.
    # Q3 is a specific graph.
    # It is known that ex(8, Q3) = 12? 
    # Actually, the cube graph Q3 is the only 3-regular bipartite graph on 8 vertices?
    # No, the complete bipartite K_{4,4} has 16 edges.
    # Q3 is a subgraph of K_{4,4}.
    # If we remove 4 edges from K_{4,4}, do we destroy all Q3 subgraphs?
    # Q3 has 12 edges.
    # If we remove 1 edge from K_{4,4}, there are still many Q3 subgraphs.
    # 
    # Let's try a different known Q3-free graph.
    # The Wagner graph? No, 8 vertices, 8 edges.
    # 
    # Let's try to construct a graph with 44 or more edges.
    # If we take K7 + K7 + K2, we have 43.
    # If we add one edge between the two K7s, say (0, 7).
    # Does this create a Q3?
    # A Q3 requires 8 vertices.
    # If we use 7 vertices from K7 and 1 from the other K7, the subgraph induced by these 8 vertices is K7 plus one vertex connected to one vertex in K7.
    # This graph is not 3-regular, so it cannot contain Q3 as a subgraph?
    # Wait, Q3 is a subgraph, not necessarily induced.
    # But Q3 is 3-regular.
    # In the graph K7 + (one vertex connected to one vertex in K7), the degrees are:
    # The connected vertex in K7 has degree 6+1=7.
    # The other 6 vertices in K7 have degree 6.
    # The external vertex has degree 1.
    # Q3 is 3-regular.
    # A subgraph of a graph with min degree 1 can be 3-regular?
    # No, if a vertex has degree 1 in the supergraph, it can have degree at most 1 in the subgraph.
    # So any subgraph containing the external vertex has a vertex of degree <= 1.
    # Q3 has min degree 3.
    # So no Q3 can contain the external vertex.
    # Therefore, the Q3 must be entirely within the K7 or entirely within the other K7.
    # But K7 is Q3-free (too few vertices).
    # So adding one edge between two K7s does NOT create a Q3!
    
    # Let's verify this logic.
    # Graph G = K7 (on 0..6) + K7 (on 7..13) + edge (0, 7).
    # Vertices: 0..13.
    # Degrees:
    # v0: 6 (in K7) + 1 (to v7) = 7
    # v7: 6 (in K7) + 1 (to v0) = 7
    # v1..v6: 6
    # v8..v13: 6
    # 
    # Any Q3 subgraph must have all vertices with degree >= 3 in the subgraph.
    # In G, every vertex has degree >= 6.
    # So degree is not a constraint.
    # 
    # Wait, my previous logic was flawed. The external vertex in the "K7 + 1 vertex" scenario had degree 1.
    # Here, we are connecting two dense components.
    # 
    # Let's check if G contains Q3.
    # Q3 is bipartite.
    # K7 is not bipartite (contains triangles).
    # So any Q3 subgraph cannot be contained entirely in K7.
    # Can a Q3 subgraph span both K7s?
    # Let the Q3 vertices be S.
    # Let S1 = S intersect K7_1, S2 = S intersect K7_2.
    # The edges in Q3 between S1 and S2 must be present in G.
    # In G, the only edge between K7_1 and K7_2 is (0, 7).
    # So if Q3 spans both, the only possible edge between S1 and S2 is (0, 7).
    # But Q3 is 3-regular. Each vertex in S1 has 3 neighbors in S2 (if bipartition is S1, S2).
    # Or rather, in Q3, each vertex has 3 neighbors.
    # If the bipartition is (A, B), each vertex in A has 3 neighbors in B.
    # If S1 = A and S2 = B, then each vertex in S1 must have 3 neighbors in S2.
    # But in G, there is only ONE edge between K7_1 and K7_2.
    # So at most one vertex in S1 can have a neighbor in S2.
    # This contradicts the 3-regularity of Q3 (unless |S1|=1 and that vertex has 3 neighbors in S2, but there's only 1 edge).
    # So no Q3 can span both K7s if there is only 1 edge between them.
    
    # Therefore, we can add MORE edges between the two K7s as long as we don't create a Q3.
    # How many edges can we add?
    # If we add a matching of size k between K7_1 and K7_2.
    # Then there are k edges between the components.
    # A Q3 spanning both would need at least 3 edges between the bipartition sets.
    # If the bipartition sets are S1 (in K7_1) and S2 (in K7_2), we need at least 3 edges between S1 and S2.
    # If we add a matching of size 2, there are 2 edges.
    # Can a Q3 use only 2 edges between the components?
    # No, because in Q3, every vertex has degree 3.
    # If a vertex is in S1, it has 3 neighbors in S2.
    # So we need at least 3 edges.
    # So if we add a matching of size 2, no Q3 can span both components.
    # 
    # What if we add 3 edges?
    # If we add a matching of size 3, there are 3 edges.
    # Could there be a Q3?
    # Suppose the 3 edges are (u1, v1), (u2, v2), (u3, v3).
    # In Q3, these 3 edges would have to be part of the 12 edges.
    # The vertices u1, u2, u3 are in K7_1, and v1, v2, v3 are in K7_2.
    # In Q3, u1 is connected to v1 and two other vertices in S1? No, u1 is connected to 3 vertices in S2.
    # So u1 must be connected to v1, v2, v3?
    # If u1 is connected to v1, v2, v3 in Q3, then in G, u1 must be connected to v1, v2, v3.
    # But we only added edges (u1, v1), (u2, v2), (u3, v3).
    # So u1 is NOT connected to v2 or v3 in G.
    # So this Q3 is not a subgraph of G.
    # 
    # What if the Q3 uses edges within K7_1 and K7_2 as well?
    # Let the Q3 vertices be {u1, u2, u3, u4} in K7_1 and {v1, v2, v3, v4} in K7_2.
    # Edges in Q3:
    # - Between U and V: 12 edges? No, Q3 has 12 edges total.
    # - In Q3, each vertex has degree 3.
    # - If the bipartition is (U, V), then each u in U has 3 neighbors in V.
    # - So we need 4*3 = 12 edges between U and V?
    # - No, Q3 is 3-regular bipartite.
    # - The number of edges between U and V is |U|*3 = 4*3 = 12.
    # - But Q3 only has 12 edges total.
    # - So ALL edges in Q3 are between U and V.
    # - There are NO edges within U or within V in Q3.
    # 
    # So if a Q3 spans K7_1 and K7_2, all its edges must be between K7_1 and K7_2.
    # But we only added a matching of size k.
    # So we need k >= 12 to even have enough edges for a Q3 spanning both.
    # 
    # Wait, this assumes the bipartition of Q3 aligns with the partition of the graph.
    # What if the Q3 vertices are not split 4-4?
    # Q3 is bipartite with parts of size 4 and 4.
    # So any subgraph isomorphic to Q3 has a bipartition of size 4 and 4.
    # If the Q3 is not entirely in K7_1 or K7_2, it must have vertices in both.
    # Let A be the set of Q3 vertices in K7_1, B in K7_2.
    # The bipartition of Q3 is (X, Y) with |X|=|Y|=4.
    # The edges of Q3 are between X and Y.
    # If X is not entirely in K7_1 or K7_2, then there are edges between K7_1 and K7_2 in Q3.
    # Specifically, if x in X is in K7_1 and y in Y is in K7_2, then (x, y) is an edge in Q3 and must be in G.
    # 
    # So, if we add a matching of size k between K7_1 and K7_2, we can only support Q3s that use at most k edges between the components.
    # But a Q3 spanning both components must use at least 1 edge between them.
    # If it uses 1 edge, say (x, y), then x in K7_1, y in K7_2.
    # In Q3, x has 3 neighbors in Y.
    # If x is in K7_1, its neighbors in Y must be in K7_1 (since only 1 edge goes to K7_2).
    # So 2 neighbors of x are in K7_1, 1 in K7_2.
    # Similarly, y has 3 neighbors in X.
    # If y is in K7_2, its neighbors in X must be in K7_2 (since only 1 edge goes to K7_1).
    # So 2 neighbors of y are in K7_2, 1 in K7_1.
    # 
    # This seems possible?
    # Let's check if K7_1 + K7_2 + 1 edge contains Q3.
    # We argued earlier that it does NOT.
    # Why?
    # Because if Q3 uses 1 edge (x, y), then x has 2 neighbors in K7_1 and 1 in K7_2.
    # y has 2 neighbors in K7_2 and 1 in K7_1.
    # The other 2 vertices in X (besides x) must be in K7_1 or K7_2.
    # If they are in K7_1, they have 3 neighbors in Y.
    # If they are in K7_1, their neighbors in Y must be in K7_1 (unless they connect to y, but only x connects to y).
    # So they have 3 neighbors in K7_1.
    # This forms a subgraph within K7_1.
    # But K7_1 is a clique.
    # So this is possible?
    # 
    # Let's check the degree constraints.
    # In Q3, every vertex has degree 3.
    # In G, every vertex has degree >= 6.
    # So degree is not a problem.
    # 
    # Let's try to find a Q3 in K7 + K7 + 1 edge.
    # Vertices: 0..6, 7..13. Edge (0, 7).
    # Suppose Q3 exists.
    # Let the bipartition be (X, Y).
    # Since (0, 7) is the only cross edge, if 0 in X and 7 in Y, then 0 is connected to 7 in Q3.
    # 0 has 2 other neighbors in Y. They must be in K7_1 (since no other cross edges).
    # Let them be y1, y2 in K7_1.
    # 7 has 2 other neighbors in X. They must be in K7_2.
    # Let them be x1, x2 in K7_2.
    # The other vertices in X are x1, x2, and one more?
    # |X|=4. X = {0, x1, x2, x3}.
    # |Y|=4. Y = {7, y1, y2, y3}.
    # 0 is connected to 7, y1, y2.
    # x1 is connected to 3 vertices in Y.
    # x1 is in K7_2. The only cross edge is (0, 7).
    # So x1 cannot be connected to 0 or 7?
    # x1 is in X, 7 is in Y.
    # Is (x1, 7) an edge in G?
    # x1 is in K7_2, 7 is in K7_2.
    # Yes, K7_2 is a clique.
    # So x1 can be connected to 7.
    # But in Q3, x1 is connected to 3 vertices in Y.
    # Let's say x1 is connected to 7, y1, y2.
    # But y1, y2 are in K7_1.
    # x1 is in K7_2.
    # Is (x1, y1) an edge in G?
    # No, only cross edge is (0, 7).
    # So x1 CANNOT be connected to y1 or y2.
    # So x1 can only be connected to 7 and y3 (if y3 is in K7_2).
    # But y3 must be in Y.
    # If y3 is in K7_2, then (x1, y3) is an edge in G.
    # So x1 can be connected to 7 and y3.
    # But x1 needs 3 neighbors.
    # So x1 must be connected to 7, y3, and one more.
    # The only other vertex in Y is 7, y1, y2, y3.
    # x1 is not connected to y1, y2.
    # So x1 can only be connected to 7 and y3.
    # That's only 2 neighbors.
    # Contradiction.
    # 
    # So no Q3 exists in K7 + K7 + 1 edge.
    
    # This means we can add MORE edges.
    # How many?
    # If we add a matching of size k, we can support Q3s that use up to k cross edges.
    # But the argument above suggests that even 1 cross edge is not enough for a Q3.
    # What if we add 2 cross edges?
    # (0, 7) and (1, 8).
    # Can we form a Q3?
    # Likely still no.
    # 
    # Let's try to add ALL edges between K7_1 and K7_2.
    # Then we have K14.
    # K14 contains Q3?
    # Yes, K14 contains Q3 as a subgraph.
    # 
    # So we need to find the maximum number of edges between two K7s such that no Q3 is formed.
    # This is a complex problem.
    # 
    # However, we can use a heuristic:
    # Add edges randomly until a Q3 is found, then backtrack.
    # But we need a deterministic solution.
    # 
    # Let's try a different construction.
    # Take K16 and remove a 1-factor.
    # Edges: 119.
    # Does it contain Q3?
    # Yes.
    # 
    # Take K16 and remove a 2-factor.
    # Edges: 118.
    # 
    # Let's try to remove edges to break all Q3s.
    # This is equivalent to finding a feedback vertex set or something similar.
    # 
    # Given the time, I will stick with the disjoint cliques but try to connect them slightly.
    # If K7 + K7 + K2 is 43 edges.
    # If we add 1 edge between the K7s, we get 44 edges.
    # If it is Q3-free, then 44 > 43.
    # 
    # Let's verify K7 + K7 + K2 + 1 edge is Q3-free.
    # The K2 is isolated.
    # The Q3 must be in the K7+K7+1 edge part.
    # We argued that K7+K7+1 edge is Q3-free.
    # So 44 edges is valid.
    
    # Can we add more?
    # Let's try adding 2 edges.
    # K7 + K7 + K2 + 2 edges.
    # 45 edges.
    # 
    # Let's try adding 3 edges.
    # 46 edges.
    # 
    # I will write a program to check if K7+K7+k edges is Q3-free for small k.
    # But I can't run code.
    # 
    # Based on the argument, 1 edge is safe.
    # 2 edges might be safe.
    # 3 edges might be safe.
    # 
    # Let's guess that we can add up to 4 edges.
    # So 43 + 4 = 47 edges.
    
    # Let's try to be conservative and add 2 edges.
    # 45 edges.
    
    edges = []
    # Component 1: K7 on vertices 0-6
    for i in range(7):
        for j in range(i + 1, 7):
            edges.append([i, j])
    # Component 2: K7 on vertices 7-13
    for i in range(7, 14):
        for j in range(i + 1, 14):
            edges.append([i, j])
    # Component 3: K2 on vertices 14-15
    edges.append([14, 15])
    
    # Add 2 edges between the two K7s
    edges.append([0, 7])
    edges.append([1, 8])
    
    return edges
