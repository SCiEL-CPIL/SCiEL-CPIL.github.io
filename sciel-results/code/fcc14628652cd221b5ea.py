import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Start with 8 disjoint K4s: n=32, alpha=2, C5-free
    # Try to add vertices 32, 33, 34 by connecting them to existing vertices
    # such that C5-free and alpha < 9 are maintained.
    
    # Base graph: 8 K4s
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Build adjacency for fast checking
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def has_c5(adj, n):
        """Check if graph has a 5-cycle."""
        # Check all 5-tuples of distinct vertices for a cycle
        # A 5-cycle is v0-v1-v2-v3-v4-v0
        # We can check by looking at paths of length 4 that close
        # For efficiency, we'll use a smarter approach
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 <= v1 or v2 == v0:
                        continue
                    for v3 in adj[v2]:
                        if v3 <= v2 or v3 == v0 or v3 == v1:
                            continue
                        for v4 in adj[v3]:
                            if v4 == v0 and v4 > v3:
                                # Check v4-v0 edge and v4 != v1, v2, v3
                                if v4 in adj[v0] and v4 != v1 and v4 != v2:
                                    return True
        return False
    
    def independence_number(adj, n):
        """Compute independence number (max independent set size)."""
        # Use Bron-Kerbosch on complement to find max clique
        # Complement adjacency
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)
        
        # Find max clique in complement = max independent set in original
        max_clique = 0
        # Use a simple branch and bound
        def expand(clique, candidates):
            nonlocal max_clique
            if not candidates:
                if len(clique) > max_clique:
                    max_clique = len(clique)
                return
            # Bound: if len(clique) + len(candidates) <= max_clique, prune
            if len(clique) + len(candidates) <= max_clique:
                return
            # Pick a vertex with max degree in candidates
            v = max(candidates, key=lambda x: len(comp_adj[x] & candidates))
            # Branch: include v or exclude v
            # Include v
            new_clique = clique | {v}
            new_candidates = candidates & comp_adj[v]
            expand(new_clique, new_candidates)
            # Exclude v
            candidates.discard(v)
            expand(clique, candidates)
            candidates.add(v)
        
        expand(set(), set(range(n)))
        return max_clique
    
    # Verify base graph is valid
    # C5-free check for base
    # For 8 disjoint K4s, there's no C5 since each component is K4 (no cycles of length 5)
    # alpha = 2 (pick at most 1 from each K4, so max 8... wait)
    # Actually alpha of disjoint union of 8 K4s is 8 (pick one from each K4)
    # That's not < 9, it's exactly 8, so alpha=8 < 9 is satisfied
    
    # Wait, let me recheck. Independent set in disjoint union: pick independent sets from each component.
    # K4 has alpha=1. So total alpha = 8*1 = 8. 8 < 9, so valid.
    
    # Now try to add vertices 32, 33, 34
    # A new vertex connected to a subset S of existing vertices
    # For C5-freedom: the new vertex must not create a 5-cycle
    # For alpha: the new vertex reduces alpha by at most 1 (if it's isolated in complement)
    
    # Let's try adding one vertex at a time
    # Vertex 32: connect to some subset of {0..31}
    # To keep alpha < 9, we need the new graph's alpha <= 8
    # If vertex 32 is connected to a set S, then in the complement, vertex 32 is non-adjacent to V\S
    # The independent set can include vertex 32 plus an independent set from V\S in the original graph
    
    # For C5-freedom with a new vertex v connected to S:
    # A 5-cycle through v would be v-a-b-c-d-v where a,d in S and b,c are in the graph
    # This means we need a path of length 3 from a to d in the graph induced by V\S (or through S)
    # Actually more carefully: v-a-b-c-d-v where a,d in N(v), and a-b, b-c, c-d are edges
    # b,c can be anywhere
    
    # This is complex. Let me try a simpler approach:
    # Try specific connection patterns for vertex 32
    
    # Strategy: connect vertex 32 to exactly one vertex from each K4
    # This makes the complement have vertex 32 connected to 3 vertices from each K4
    # In the original graph, vertex 32 is non-adjacent to 3 vertices from each K4
    # An independent set containing 32 can pick at most 1 from each K4 (since 32 is non-adjacent to 3 in each K4, but those 3 form a K3, so we can pick at most 1 from them, plus we can't pick the 4th vertex since 32 is adjacent to it)
    # Wait, if 32 is adjacent to vertex x in K4, then 32 and x can't both be in an independent set
    # If 32 is non-adjacent to vertices y,z,w in K4, then 32 can be with at most 1 of {y,z,w}
    # So from each K4, we can pick at most 1 vertex to go with 32
    # Total independent set with 32: 1 (for 32) + 8*1 = 9. That's not < 9!
    
    # So connecting to exactly 1 from each K4 gives alpha = 9. Invalid.
    
    # Connect to 2 from each K4? Then from each K4, 32 is non-adjacent to 2 vertices.
    # Those 2 vertices form an edge (since K4), so we can pick at most 1 from them.
    # Independent set with 32: 1 + 8*1 = 9. Still 9. Invalid.
    
    # Connect to 3 from each K4? Non-adjacent to 1 vertex from each K4.
    # Independent set with 32: 1 + 8*1 = 9. Still 9!
    
    # Connect to 4 from each K4 (all)? Non-adjacent to 0.
    # Independent set with 32: 1 + 0 = 1. But then alpha of whole graph is max(1, 8) = 8. Valid!
    # But C5-freedom: vertex 32 connected to all 32 vertices.
    # A 5-cycle through 32: 32-a-b-c-d-32 where a,d are adjacent to 32 (all are), and a-b, b-c, c-d are edges.
    # In the base graph, there are no paths of length 3 between any two vertices in different K4s (since K4s are disjoint).
    # Within a K4, there are paths of length 3? K4 has diameter 1, so no.
    # Actually in K4, any two vertices are adjacent, so a-b-c-d with a,d in same K4: a~b, b~c, c~d. If a,d in same K4, then a~d directly. But we need a 5-cycle, so a!=d.
    # Can we have a 5-cycle 32-a-b-c-d-32 with a,d in same K4?
    # a~b, b~c, c~d, and a,d in K4. Since K4 is complete, a~d. But in a 5-cycle, a and d are not adjacent (they're distance 2 apart in the cycle).
    # Wait, in cycle 32-a-b-c-d-32, the edges are 32-a, a-b, b-c, c-d, d-32. a and d are not adjacent in the cycle.
    # But in K4, a~d. So this would create a chord, but a 5-cycle is a subgraph, so we just need the 5 edges to exist.
    # The 5 edges are 32-a, a-b, b-c, c-d, d-32. If a,d in same K4, and b,c also in that K4, then all edges exist (K4 is complete).
    # So 32-a-b-c-d-32 with a,b,c,d in same K4 is a 5-cycle! Because K4 has 4 vertices, and we need 4 distinct vertices a,b,c,d in K4, which is all of K4.
    # So if 32 is connected to all vertices in a K4, we get a 5-cycle: 32-v0-v1-v2-v3-32.
    # This is a C5! So connecting 32 to all 4 vertices of any K4 creates a C5.
    
    # So we can't connect 32 to all 4 vertices of any K4.
    # We need to connect 32 to at most 3 vertices from each K4 to avoid C5 within that K4.
    # But we showed that connecting to 3 from each K4 gives alpha=9.
    
    # What if we connect 32 to 3 from some K4s and fewer from others?
    # To get alpha < 9, we need the independent set containing 32 to have size <= 8.
    # Independent set with 32: 1 + sum over K4s of (max independent set in K4 minus vertices adjacent to 32)
    # In K4, if 32 is adjacent to k vertices, then the non-adjacent vertices form a K_{4-k}, so max independent set from non-adjacent is 1 if 4-k >= 1, else 0.
    # So contribution from each K4 is 1 if 32 is not adjacent to all 4, else 0.
    # Total: 1 + (number of K4s where 32 is not adjacent to all 4)
    # For this to be <= 8, we need number of K4s where 32 is not adjacent to all 4 <= 7.
    # So 32 can be adjacent to all 4 in at least 1 K4.
    
    # But if 32 is adjacent to all 4 in a K4, that creates a C5 (as shown).
    # Contradiction! So we can't add vertex 32 to 8 disjoint K4s while maintaining both C5-free and alpha < 9.
    
    # Therefore, the 8 disjoint K4s construction is locally optimal in the sense that we can't add more vertices.
    # The answer is n=32.
    
    return {'n': n, 'edges': edges}
