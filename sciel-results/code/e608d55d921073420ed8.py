"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random

# EVOLVE-BLOCK-START

def construct():
    random.seed(42)
    n = 16
    
    # Q3 canonical edges
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.add((i, j))
    
    # Start with complete graph, remove edges to break Q3
    # Use a smarter approach: start with K_{4,12} augmented with intra-part edges
    # Actually, let's just use random + pruning with a fast check
    
    # Build adjacency as bitmask for speed
    # adj[i] is a bitmask of neighbors of i
    
    # Start with dense random graph
    p = 0.9
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def has_q3():
        """Check if graph contains Q3. Returns True/False."""
        # For each 8-subset, check if Q3 is a subgraph
        # Use bitmasks for speed
        from itertools import combinations
        for subset in combinations(range(n), 8):
            # Try all 8! / 48 distinct Q3 embeddings
            # Simplified: try a few random mappings
            for _ in range(50):
                mapping = {}
                q3_order = list(range(8))
                random.shuffle(q3_order)
                targets = list(subset)
                random.shuffle(targets)
                valid = True
                for k, qv in enumerate(q3_order):
                    mapping[qv] = targets[k]
                # Check all Q3 edges
                for (u, v) in q3_edges:
                    a, b = mapping[u], mapping[v]
                    if not (adj[a] >> b) & 1:
                        valid = False
                        break
                if valid:
                    return True
        return False
    
    # This is still too slow. Let me just go with a known good construction.
    
    # K_{3,13} has 39 edges. Let's try to add edges within the parts.
    # Part A = {0,1,2}, Part B = {3,...,15}
    # Add some edges within B. Since Q3 is bipartite, any Q3 in the final graph
    # must use vertices from both A and B in a bipartite pattern.
    # If we add edges within B, a Q3 could potentially use 3 vertices from B and 1 from A?
    # No, Q3 is bipartite with parts of size 4. So any Q3 subgraph must have 4 vertices
    # in one "side" and 4 in the other. But the "sides" don't have to align with our A/B split.
    
    # Let me just return K_{3,13} plus a few safe edges.
    # Actually, let's try: A = {0,1,2}, B = {3..15}. Complete between A and B (39 edges).
    # Add edges within B that don't create Q3.
    # A Q3 using vertices from A and B: since |A|=3 < 4, no Q3 can have all 4 of its
    # bipartition class vertices in A. So any Q3 must have at least 1 vertex in B on
    # each side of Q3's bipartition. This means Q3 uses at least 4 vertices from B.
    
    # If we add a matching within B, does that create Q3? A Q3 is 3-regular, so adding
    # a matching (1-regular) within B might not immediately create Q3.
    
    # Let's just add all edges within A (3 edges) and see.
    # Total: 39 + 3 = 42 edges.
    
    # Check: can we have Q3 with vertices from A and B?
    # Q3 needs 8 vertices, 4 from each bipartition class.
    # If we use all 3 vertices of A, we need 1 more from B for one class, and 4 from B for other.
    # The 3 vertices of A are connected to each other (triangle). In Q3, no two vertices in the
    # same bipartition class are adjacent. So if we put 2 vertices of A in the same Q3 bipartition
    # class, they shouldn't be adjacent in Q3, but they ARE adjacent in our graph. That's fine -
    # the subgraph just needs to CONTAIN Q3's edges, not be exactly Q3.
    
    # Wait, I'm confusing myself. A subgraph isomorphic to Q3 means: there's an injective map
    # f: V(Q3) -> V(G) such that for every edge (u,v) in Q3, (f(u), f(v)) is an edge in G.
    # Extra edges in G don't matter.
    
    # So the question is: is there a subset of 8 vertices in G and a bijection to Q3 vertices
    # preserving Q3's edges?
    
    # With A={0,1,2} (triangle) and B={3..15} (independent set initially), plus all A-B edges:
    # For Q3, we need 4 vertices in each bipartition class of Q3.
    # Case 1: Both Q3 bipartition classes use vertices from both A and B.
    #   Since A has only 3 vertices, at least one Q3 class has >= 1 from B, and the other has >= 1 from B.
    #   Actually, each Q3 class has 4 vertices. Total 8 vertices. A has 3, so B must supply at least 5.
    #   So one Q3 class has a vertices from A and 4-a from B, other has 3-a from A and 1+a from B.
    #   a can be 0,1,2,3.
    
    # This is getting too complex to reason about. Let me just go with a safe construction.
    
    # Safe: K_{3,13} + triangle on A. 42 edges.
    # Or even safer: just K_{3,13} with 39 edges, which we know is Q3-free.
    
    # Let me try to be a bit more aggressive. Use K_{4,12} but with a specific structure.
    # A = {0,1,2,3}, B = {4..15}. 
    # Instead of complete bipartite, use a 4-regular bipartite graph (each A vertex connects to 4 B vertices).
    # That's 16 edges. Too few.
    
    # How about: A={0,1,2,3}, B={4..15}. Connect each A vertex to 9 B vertices (randomly chosen).
    # 4*9 = 36 edges. Plus add edges within A and B carefully.
    
    # I think the safest bet is to just return K_{3,13} = 39 edges, or try to add a few edges.
    
    # Let me try: K_{3,13} + all edges within B that form a matching.
    # B has 13 vertices, so max matching is 6 edges. Total: 39 + 6 = 45.
    # Is this Q3-free? A Q3 would need 4 vertices from each side of its bipartition.
    # With A of size 3, any Q3 uses at most 3 from A. So it uses at least 1 from B on each side.
    # The matching edges within B could potentially be part of Q3.
    
    # I'm not confident. Let me just go with 39 edges to be safe, matching the seed.
    
    return [[a, b] for a in range(3) for b in range(3, 16)]

# EVOLVE-BLOCK-END
