"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    n = 15
    
    # Try to build a threshold graph with distinct degrees 0..14
    # Threshold graph construction: start with isolated vertices, add universal or isolated
    # For degree sequence d_0 <= d_1 <= ... <= d_{n-1}, the threshold graph can be built
    # by processing vertices in order of degree.
    
    # The threshold graph with degree sequence (0,1,2,...,n-1) is the "complete split"
    # Actually, let's think: in a threshold graph, vertices can be ordered so that each is
    # either isolated or universal w.r.t. remaining vertices.
    
    # For degrees 0..14 on 15 vertices:
    # Vertex with degree 0: isolated
    # Vertex with degree 1: connected to one other
    # ...
    # Vertex with degree 14: connected to all others
    
    # A threshold graph with this degree sequence: 
    # Label vertices 0..14. Vertex i has degree i.
    # Construction: process from highest degree to lowest.
    # Vertex 14 (deg 14): connect to all 14 others
    # Vertex 13 (deg 13): should have 13 edges. Already connected to 14? No wait.
    
    # Let me use the standard threshold graph construction:
    # Given a degree sequence, the threshold graph is unique.
    # For (0,1,2,...,14): 
    # The adjacency matrix has a specific pattern.
    
    # Actually, let's just build it directly:
    # In a threshold graph with distinct degrees, the vertex with degree k is connected 
    # to exactly the k vertices with the highest degrees among the others.
    
    # So: sort vertices by degree descending: v_14, v_13, ..., v_0
    # v_i is connected to v_j if i > j and ... hmm, need to be careful.
    
    # Standard result: the threshold graph with degree sequence (0,1,...,n-1) 
    # has edge set: {(i,j) : i+j >= n-1} for 0-indexed? Let me verify for n=4.
    # Degrees (0,1,2,3). Edges: (0,3), (1,3), (2,3)? That gives degs 1,1,1,3. Wrong.
    
    # Let me just construct it properly using the characterization:
    # A graph is a threshold graph iff it has no induced P4, C4, 2K2.
    # The threshold graph with degree sequence d_0 <= ... <= d_{n-1} can be built as follows:
    # Start with n isolated vertices. Process vertices in order of increasing degree.
    # When adding vertex v_i, connect it to d_i vertices that have already been "activated" 
    # in a specific way.
    
    # Simpler: just use the known structure. For the threshold graph T_n with 
    # degrees (0,1,...,n-1), the adjacency is:
    # Edge (i,j) exists iff i + j >= n - 1, where i,j are 0-indexed and we label 
    # vertices so that vertex i has degree i.
    
    # Verify for n=4: vertices 0,1,2,3 with degrees 0,1,2,3.
    # Edges where i+j >= 3: (0,3), (1,2), (1,3), (2,3). 
    # Degrees: v0:1, v1:2, v2:2, v3:3. Not (0,1,2,3). Wrong.
    
    # Let me try: edge (i,j) iff i > j and i >= some function...
    
    # Actually, the correct threshold graph for (0,1,...,n-1):
    # It's the graph where vertex i is connected to vertices j > i for j in some range,
    # plus all vertices k < i for k in some range.
    
    # Let me just use a direct construction. The threshold graph with degree sequence
    # (0,1,2,...,n-1) is the graph G where:
    # - Vertex n-1 is universal (connected to all)
    # - Vertex n-2 is connected to n-1 and... 
    # 
    # I'll use the following known fact: the complement of a threshold graph is also a 
    # threshold graph. The degree sequence (0,1,...,n-1) is self-complementary in the 
    # sense that complement has degrees (n-1, n-2, ..., 0) which is the same set.
    
    # Let me just build it by hand for n=15 using the following algorithm:
    # The threshold graph with degree sequence (0,1,...,n-1) can be constructed by:
    # Start with an empty graph on n vertices.
    # For k from 1 to n-1:
    #   Add edge (k, n-1)  [vertex n-1 gets degree n-1]
    #   But this gives vertex n-1 degree n-1, and others degree 1. Not right.
    
    # OK let me use the proper threshold graph construction from the degree sequence.
    # The threshold graph is uniquely determined. Here's an algorithm:
    # 1. Sort vertices by degree: v_0 (deg 0), v_1 (deg 1), ..., v_{n-1} (deg n-1)
    # 2. The vertex with maximum degree (v_{n-1}) is universal. Remove it and its incident edges.
    #    The remaining graph is a threshold graph on n-1 vertices with degree sequence 
    #    (0,1,...,n-2) where each degree is reduced by 1 if the vertex was connected to v_{n-1}.
    #    Since v_{n-1} is universal, all remaining vertices had their degree reduced by 1.
    #    So remaining degrees: (0-1, 1-1, ..., n-2-1) = (-1, 0, 1, ..., n-3). 
    #    But degrees can't be negative, so v_0 (deg 0) was NOT connected to v_{n-1}? 
    #    Contradiction since v_{n-1} is universal.
    
    # I think I'm overcomplicating. Let me just use a SAT/constraint approach or 
    # directly verify a candidate.
    
    # Candidate: The graph where edge (i,j) exists iff min(i,j) + max(i,j) >= n-1
    # For n=15: edge (i,j) iff i+j >= 14
    # Let me compute degrees:
    # Vertex i: connected to j where j >= 14-i and j != i
    # If i <= 7: j ranges from 14-i to 14, count = 14-(14-i)+1 = i+1, minus 1 if i is in range
    #   For i=0: j in 14..14, count=1. deg=1. But we want deg 0.
    
    # This doesn't give (0,1,...,14).
    
    # Let me try a different approach: just use the fact that we need distinct degrees,
    # and find ANY graph with distinct degrees, not necessarily the threshold one.
    # 
    # A simple construction: start with a random graph and add edges to make degrees distinct.
    # Or: use a "nested" construction.
    
    # Actually, the simplest valid construction: 
    # Make vertex 0 isolated (deg 0), vertex 1 connected only to vertex 14 (deg 1),
    # vertex 2 connected to vertices 13,14 (deg 2), etc.
    # Vertex i connected to vertices (14-i), (15-i), ..., 14 for i=0..14?
    # Vertex i: connect to j where j >= 14-i. 
    # For i=0: j>=14, so j=14. deg=1. Not 0.
    
    # Let me just do: vertex i has degree i by connecting to the i highest-degree vertices
    # that aren't itself. Process in order of increasing degree.
    
    # Build adjacency matrix
    adj = [[0]*n for _ in range(n)]
    
    # We want vertex i to have degree i.
    # Process vertices from 0 to 14. When processing vertex i, we need to give it degree i.
    # It can connect to vertices with higher index (not yet processed) or lower index (already processed).
    # 
    # Strategy: vertex i connects to vertices i+1, i+2, ..., i+i (if they exist) and/or 
    # some lower-index vertices.
    # 
    # Simpler: just assign edges greedily.
    # Vertex 0: degree 0, no edges.
    # Vertex 1: degree 1, connect to vertex 14 (highest available).
    # Vertex 2: degree 2, connect to vertices 13, 14.
    # ...
    # Vertex i: degree i, connect to vertices (14-i+1), (14-i+2), ..., 14. That's i vertices.
    #   For i=1: connect to 14. (1 vertex)
    #   For i=2: connect to 13,14. (2 vertices)
    #   For i=14: connect to 1,2,...,14. (14 vertices)
    
    # Let's verify this gives distinct degrees.
    # Vertex j's degree: it's connected to vertex i if j >= 14-i+1, i.e., i >= 14-j+1 = 15-j.
    # And i < j (since we only connect i to higher-index vertices in this scheme? No, i connects to 
    # vertices with index >= 14-i+1, which could be less than i for large i).
    
    # Let me just code it up and check.
    
    for i in range(n):
        # Vertex i should have degree i
        # Connect to i vertices with the highest indices
        if i == 0:
            continue
        # Connect to vertices: 14, 13, ..., 14-i+1 (that's i vertices)
        for k in range(i):
            j = n - 1 - k  # 14, 13, ..., 14-i+1
            if j != i:
                adj[i][j] = 1
                adj[j][i] = 1
    
    # Check degrees
    degrees = [sum(adj[i]) for i in range(n)]
    
    # Verify constraint: no 3-path s-u-v-t with deg(s)==deg(t)
    # If all degrees are distinct, this is automatically satisfied.
    if len(set(degrees)) != n:
        # Degrees not distinct, need to fix or use different approach
        # For now, let's check if the constraint is still satisfied
        pass
    
    # Check the actual constraint
    valid = True
    for u in range(n):
        for v in range(n):
            if u == v or adj[u][v] == 0:
                continue
            for s in range(n):
                if s == u or s == v or adj[s][u] == 0:
                    continue
                for t in range(n):
                    if t == s or t == u or t == v or adj[t][v] == 0:
                        continue
                    if degrees[s] == degrees[t]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use a different construction or random search
        # For now, return the star as fallback
        return [[0, v] for v in range(1, n)]
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([int(i), int(j)])
    
    return edges
# EVOLVE-BLOCK-END
