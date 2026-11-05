import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # Build conflict graph: nodes are edges of K16.
    # Two edges conflict if they can appear together in a Q3.
    # A Q3 has 12 edges. We need to find all sets of 12 edges that form a Q3.
    
    # First, enumerate all Q3 embeddings in K16.
    # A Q3 is a 3-dimensional cube. Its vertices can be labeled as elements of {0,1}^3.
    # An embedding maps {0,1}^3 -> {0,...,15} injectively.
    # Two embeddings give the same edge set if they differ by an automorphism of Q3.
    
    # Q3 automorphism group has 48 elements (hyperoctahedral group B3).
    # But for our purposes, we just need to find all distinct edge sets.
    
    # Strategy: For each 8-subset of vertices, check if it can be the vertex set of a Q3.
    # There are C(16,8) = 12870 such subsets. For each, we need to check if there's
    # a bijection to {0,1}^3 that makes it a Q3.
    
    # Better: enumerate all ways to pick 8 vertices and assign them to cube positions.
    # But that's too many. Instead, use the fact that Q3 has specific structure.
    
    # Let's enumerate Q3s by choosing the 8 vertices and checking if the induced
    # subgraph of K16 on those vertices contains a Q3 as a subgraph (not necessarily induced).
    
    # Actually, for the conflict graph, we need: two edges e1, e2 conflict if there
    # exists a Q3 containing both e1 and e2.
    
    # Let's enumerate all Q3 edge sets in K16.
    # A Q3 on vertices v0..v7 with v_i corresponding to binary i.
    # Edges are between v_i and v_j where i,j differ in exactly one bit.
    
    # To enumerate: pick 8 distinct vertices, try all 8! / 48 = 1680 assignments to cube
    # positions (accounting for automorphisms). That's too slow.
    
    # Better approach: Use the structure of Q3.
    # In Q3, each vertex has degree 3. The graph is bipartite (even/odd parity).
    # 4 vertices of even parity, 4 of odd parity.
    # Each even vertex connects to 3 odd vertices, and vice versa.
    
    # So a Q3 embedding: pick 4 "even" vertices E and 4 "odd" vertices O from our 16.
    # Then we need a 3-regular bipartite graph between E and O that is Q3.
    # There are only a few 3-regular bipartite graphs on 4+4 vertices that are Q3.
    
    # Actually, Q3 is the unique 3-regular bipartite graph on 8 vertices up to isomorphism?
    # No, there are others (e.g., the 3x3 grid minus corners... no).
    # Let me just enumerate all 3-regular bipartite graphs on 4+4 and check which are Q3.
    
    # The 3-regular bipartite graphs on 4+4 vertices: 
    # K_{4,4} minus a perfect matching. There are 3 distinct such graphs up to isomorphism?
    # Actually, K_{4,4} has 16 edges. Remove a perfect matching (4 edges) to get 12 edges.
    # The number of perfect matchings in K_{4,4} is 4! = 24.
    # But different matchings can give isomorphic graphs.
    
    # Let me just brute force: for each 8-subset of vertices, split into two 4-sets (even/odd),
    # check if there's a Q3. But the split matters.
    
    # Simpler: enumerate all 8-subsets, and for each, check all ways to assign to cube.
    # C(16,8) * 8! / 48 is too much.
    
    # Let me use a different approach for the conflict graph:
    # Two edges (a,b) and (c,d) conflict if they can be in the same Q3.
    # This means there exist 6 more vertices such that the 8 vertices form a Q3
    # with (a,b) and (c,d) as edges.
    
    # For the branch-and-bound, I'll use a greedy + local search hybrid instead,
    # which is more practical given the time limit.
    
    def check_q3_with_edge(adj, u, v):
        """Check if adding edge (u,v) creates a Q3. Returns True if Q3 exists."""
        u_neighbors = [x for x in range(n) if x != u and x != v and adj[u][x]]
        v_neighbors = [x for x in range(n) if x != u and x != v and adj[v][x]]
        
        if len(u_neighbors) < 2 or len(v_neighbors) < 2:
            return False
        
        for a in u_neighbors:
            for b in u_neighbors:
                if a >= b:
                    continue
                for c in v_neighbors:
                    for d in v_neighbors:
                        if c >= d:
                            continue
                        if not (adj[a][c] and adj[b][d]):
                            continue
                        common_ab = [x for x in range(n) if x != u and x != v and x != a and x != b 
                                     and adj[a][x] and adj[b][x]]
                        common_cd = [x for x in range(n) if x != u and x != v and x != c and x != d 
                                     and adj[c][x] and adj[d][x]]
                        for e in common_ab:
                            for f in common_cd:
                                if e == f:
                                    continue
                                if adj[e][f]:
                                    return True
        return False
    
    def verify_q3_free(adj):
        """Verify the graph is Q3-free by checking all 8-subsets."""
        # For each 8-subset, check if it contains a Q3
        # This is expensive, so we only do it on the final answer
        from itertools import combinations
        for subset in combinations(range(n), 8):
            # Check if this 8-set can form a Q3
            # Try all 8! assignments to cube positions, but use symmetry to reduce
            # Actually, just check: is there a bijection f: {0..7} -> subset such that
            # for all i,j with popcount(i^j)==1, f(i)-f(j) is an edge?
            # 8! = 40320, and C(16,8) = 12870, so 12870 * 40320 = 519M, too slow.
            # Skip full verification, rely on incremental check.
            pass
        return True
    
    # Greedy construction with multiple restarts
    best_edge_count = 0
    best_adj = None
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        adj = [[False] * n for _ in range(n)]
        
        # Start with random subset of edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.seed(attempt * 12345 + 67890)
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() > deadline - 5:
                break
            if not check_q3_with_edge(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
        
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = [row[:] for row in adj]
        
        # Local improvement: try adding remaining edges
        if time.time() > deadline - 5:
            break
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for (u, v) in all_edges:
                if time.time() > deadline - 3:
                    break
                if not adj[u][v] and not check_q3_with_edge(adj, u, v):
                    adj[u][v] = True
                    adj[v][u] = True
                    improved = True
                    edge_count = sum(sum(row) for row in adj) // 2
                    if edge_count > best_edge_count:
                        best_edge_count = edge_count
                        best_adj = [row[:] for row in adj]
    
    # Try starting from K3,13 and adding edges
    if time.time() < deadline - 10:
        adj = [[False] * n for _ in range(n)]
        for u in range(3):
            for v in range(3, 16):
                adj[u][v] = True
                adj[v][u] = True
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() > deadline - 5:
                break
            if not check_q3_with_edge(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
        
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = [row[:] for row in adj]
    
    # Final local improvement
    if best_adj and time.time() < deadline - 3:
        adj = [row[:] for row in best_adj]
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(all_edges)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for (u, v) in all_edges:
                if time.time() > deadline - 2:
                    break
                if not adj[u][v] and not check_q3_with_edge(adj, u, v):
                    adj[u][v] = True
                    adj[v][u] = True
                    improved = True
                    edge_count = sum(sum(row) for row in adj) // 2
                    if edge_count > best_edge_count:
                        best_edge_count = edge_count
                        best_adj = [row[:] for row in adj]
    
    if best_adj is None:
        # Fallback: empty graph
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
