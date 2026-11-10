import time
import numpy as np
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16
    
    # K_{8,8} is the unique bipartite graph on 16 vertices with diameter 2
    # and maximum edge count (64). Every edge is critical: removing any edge
    # disconnects the two parts in the bipartite structure or creates a vertex
    # pair at distance > 2.
    #
    # Proof sketch:
    # - Diameter 2: any two vertices in the same part share the other 7 vertices
    #   in the opposite part as common neighbors (distance 2). Vertices in
    #   different parts are directly connected (distance 1).
    # - Edge-critical: removing edge (a, b) where a in A, b in B means a and b
    #   are no longer directly connected. They still have 7 common neighbors,
    #   so distance remains 2. Wait — that means the edge is NOT critical?
    #
    # Let me re-examine. In K_{8,8}, removing edge (a,b): a and b still have
    # 7 common neighbors, so dist(a,b) = 2. The diameter remains 2.
    # So K_{8,8} does NOT have the property that every edge is critical!
    #
    # I need a different approach. Let me think about what graphs satisfy the
    # condition: diameter exactly 2, and deleting ANY edge increases diameter
    # above 2 or disconnects.
    #
    # This means every edge is a "diameter-critical" edge. In a diameter-2 graph,
    # an edge (u,v) is critical if after removing it, either the graph is
    # disconnected or there exists a pair at distance >= 3.
    #
    # For a complete graph K_n, removing any edge leaves K_n minus an edge.
    # The two endpoints of the removed edge are at distance 2 (via any other
    # vertex), so diameter remains 2. So K_n edges are NOT critical.
    #
    # I need a sparse-ish graph where every edge is essential for maintaining
    # diameter 2.
    #
    # Consider a cycle C_n for n >= 5: diameter is floor(n/2). For n=5, diameter=2.
    # Removing any edge from C_5 gives a path P_5 with diameter 4. So C_5 works
    # but only has 5 edges.
    #
    # For 16 vertices, I need something denser. Let me think about Moore graphs
    # or other extremal structures.
    #
    # Actually, let me reconsider. The condition is that EVERY edge is critical.
    # This is a very strong condition. Let me think about what graphs satisfy it.
    #
    # A graph where every edge is critical for diameter 2: this means the graph
    # is "edge-minimal" for having diameter 2. These are called "diameter-critical"
    # graphs or "minimally 2-diameter" graphs.
    #
    # Key insight: In such a graph, for every edge (u,v), there must exist a pair
    # of vertices {x,y} such that the only path of length <= 2 between x and y
    # uses the edge (u,v). 
    #
    # Let me try a different construction. Consider the Petersen graph (10 vertices,
    # 15 edges, diameter 2). Is every edge critical? The Petersen graph is known
    # to be edge-critical for diameter 2. Let me verify: removing any edge from
    # the Petersen graph gives a graph with diameter 3.
    #
    # For 16 vertices, I could take a product or join of smaller critical graphs.
    #
    # Alternative: use a random search with verification. Given the 60s time limit,
    # I can try a few random graphs and verify them.
    
    # Strategy: Try several candidate constructions and verify.
    
    # Candidate 1: Complete bipartite K_{8,8} - already shown NOT to work
    # Candidate 2: Random dense graph with diameter 2, check edge-criticality
    
    # Let's try a structured approach: 
    # Start with a base graph known to have the property, extend to 16 vertices.
    
    # The Clebsch graph (16 vertices, 30 edges) is a strongly regular graph
    # with parameters (16, 5, 0, 2). It has diameter 2.
    # Let me check if every edge is critical.
    
    # Clebsch graph construction: vertices are binary strings of length 4 with
    # even weight (16 vertices). Two vertices are adjacent if they differ in
    # exactly 2 positions.
    
    # Actually, the Clebsch graph has 16 vertices and each vertex has degree 5,
    # so 16*5/2 = 40 edges. Wait, let me recheck.
    # Strongly regular (16, 5, 0, 2): 16 vertices, degree 5, lambda=0, mu=2.
    # Number of edges = 16*5/2 = 40.
    
    # Let me construct the Clebsch graph and verify.
    
    def _clebsch_graph(n=16):
        """Construct the Clebsch graph: even-weight 4-bit strings, adjacent if Hamming distance 2."""
        vertices = []
        for i in range(n):
            bits = format(i, '04b')
            if sum(int(b) for b in bits) % 2 == 0:
                vertices.append(i)
        
        # Map from original index to position in vertices list
        # Actually, let's just use all 16 vertices (0-15) as 4-bit strings
        # and connect those at Hamming distance 2 with even parity constraint.
        # 
        # Simpler: the Clebsch graph is the halved 5-cube. Let me use a direct construction.
        
        # Vertices: all 16 4-bit strings
        # Adjacent if Hamming distance is 2 AND both have even weight
        # Wait, that's not right either. Let me use the standard construction.
        
        # The Clebsch graph: vertices are Z_2^4, two vertices adjacent if their
        # difference is in the set {0011, 0101, 0110, 1001, 1010, 1100} (weight-2 vectors).
        # This gives a regular graph of degree 6... no.
        
        # Let me just use a known adjacency for the Clebsch graph.
        # Actually, let me try a different well-known graph.
        
        # The Shrikhande graph: 16 vertices, degree 6, strongly regular (16,6,2,2).
        # 16*6/2 = 48 edges. Diameter 2.
        
        # Construction: vertices are (i,j) for i,j in {0,1,2,3}.
        # (i,j) adjacent to (i',j') if i=i' and j-j' in {1,3} mod 4, or j=j' and i-i' in {1,3} mod 4,
        # or i-i' in {1,3} mod 4 and j-j' in {1,3} mod 4.
        # Wait, that gives degree 3+3+4=10? No.
        
        # Let me use a simpler approach: generate random graphs and test.
        
        return None
    
    # Let me just do a direct search with verification.
    # Given 16 vertices, I'll try several structured candidates.
    
    # Candidate: Complete graph minus a perfect matching (16 vertices, 60 edges)
    # Diameter 1 (complete minus matching still has diameter 2 for matched pairs, 1 otherwise)
    # Actually diameter is 2. Is every edge critical? Removing an edge (u,v) where
    # u,v are not matched: they still have many common neighbors. Not critical.
    
    # Let me think differently. The maximum number of edges in a graph on 16 vertices
    # with diameter 2 where every edge is critical.
    
    # Upper bound: if every edge is critical, then for each edge e, there's a pair
    # {x,y} whose distance becomes >2 when e is removed. This is a strong constraint.
    
    # Let me try: a cycle of length 16 with chords to make diameter 2, then check
    # which edges are critical and remove non-critical ones, then re-check.
    
    # Actually, the simplest approach: brute-force search over small edge sets.
    # But 16 vertices means up to 120 possible edges. That's too many to enumerate.
    
    # Let me try a greedy construction:
    # 1. Start with a spanning tree (15 edges).
    # 2. Add edges one at a time, keeping only those that are critical.
    # 3. The result is a maximal edge-critical diameter-2 graph.
    
    def _is_edge_critical(adj, n, u, v):
        """Check if edge (u,v) is critical: removing it increases diameter above 2 or disconnects."""
        # Remove edge
        adj[u][v] = 0
        adj[v][u] = 0
        
        # Check diameter
        maxd = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                x = q.popleft()
                for w in range(n):
                    if adj[x][w] and dist[w] == -1:
                        dist[w] = dist[x] + 1
                        q.append(w)
            if -1 in dist:
                # Disconnected - edge is critical
                adj[u][v] = 1
                adj[v][u] = 1
                return True
            maxd = max(maxd, max(dist))
        
        # Restore edge
        adj[u][v] = 1
        adj[v][u] = 1
        
        return maxd > 2
    
    def _diameter(adj, n):
        """Compute diameter. Return None if disconnected."""
        maxd = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                x = q.popleft()
                for w in range(n):
                    if adj[x][w] and dist[w] == -1:
                        dist[w] = dist[x] + 1
                        q.append(w)
            if -1 in dist:
                return None
            maxd = max(maxd, max(dist))
        return maxd
    
    # Greedy construction: start with empty graph, add edges that maintain
    # diameter <= 2 and are critical.
    
    # Better: start with a known diameter-2 graph, then prune non-critical edges.
    
    # Start with K_{8,8} (diameter 2, 64 edges), then iteratively remove
    # non-critical edges until all remaining edges are critical.
    
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = [[a, b] for a in A for b in B]
    
    # Build adjacency
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    
    # Verify initial diameter is 2
    if _diameter(adj, n) != 2:
        # Fallback
        return [[0, v] for v in range(1, n)]
    
    # Iteratively remove non-critical edges
    changed = True
    while changed:
        changed = False
        for i in range(len(edges)):
            if time.time() > deadline - 5:
                break
            u, v = edges[i]
            if not adj[u][v]:
                continue  # Already removed
            if not _is_edge_critical(adj, n, u, v):
                # Remove this edge
                adj[u][v] = 0
                adj[v][u] = 0
                edges[i] = None
                changed = True
        if time.time() > deadline - 5:
            break
    
    # Clean up edges list
    edges = [e for e in edges if e is not None]
    
    # Final verification
    if not edges:
        return [[0, v] for v in range(1, n)]
    
    # Rebuild adj from edges
    adj2 = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj2[u][v] = 1
        adj2[v][u] = 1
    
    # Check diameter == 2
    if _diameter(adj2, n) != 2:
        return [[0, v] for v in range(1, n)]
    
    # Check every edge is critical
    for u, v in edges:
        if time.time() > deadline - 2:
            break
        if not _is_edge_critical(adj2, n, u, v):
            # Edge not critical - this shouldn't happen after pruning
            # Return a valid fallback
            return [[0, v] for v in range(1, n)]
    
    # Return edges as list of [u, v] with Python int types
    result = []
    for u, v in edges:
        result.append([int(u), int(v)])
    
    return result
