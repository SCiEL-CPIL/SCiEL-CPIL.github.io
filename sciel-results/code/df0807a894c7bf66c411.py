import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Start with 8 disjoint K4s: n=32, alpha=8, no C5
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    # Build adjacency structure for fast checking
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def has_c5(g_adj, g_n):
        """Check if graph has a 5-cycle using BFS from each vertex."""
        for s in range(g_n):
            # BFS up to depth 4, tracking parents to detect cycle of length 5
            # A 5-cycle means a path of length 4 from s back to s (but not a shorter cycle)
            # Simpler: check all 5-tuples (a,b,c,d,e) with a<b<c<d<e? Too slow.
            # Use: for each pair (u,v) non-adjacent, check if there's a path of length 3 between them.
            pass
        # More efficient: enumerate all 5-cycles
        # For each 5-subset, check if it contains a C5
        # 32 choose 5 = 201376, for each check 5! / 2 / 5 = 12 edge patterns
        # That's ~2.4M checks, feasible
        for combo in combinations(range(g_n), 5):
            a, b, c, d, e = combo
            # Check if {a,b,c,d,e} induces a C5 (exactly 5 edges forming a cycle)
            # A C5 on 5 vertices: exactly 5 edges, each vertex degree 2 in the subgraph
            edges_in = 0
            deg = {v: 0 for v in combo}
            for x, y in combinations(combo, 2):
                if y in g_adj[x]:
                    edges_in += 1
                    deg[x] += 1
                    deg[y] += 1
            if edges_in == 5:
                # Check if all degrees are 2 (which means it's a cycle)
                if all(deg[v] == 2 for v in combo):
                    return True
        return False

    def independence_number(g_adj, g_n):
        """Compute the size of the maximum independent set exactly."""
        # Use branch and bound
        best = 0
        # Sort vertices by degree for better pruning
        order = sorted(range(g_n), key=lambda v: len(g_adj[v]))
        
        def branch(remaining, current_size, current_set):
            nonlocal best
            if not remaining:
                if current_size > best:
                    best = current_size
                return
            # Upper bound: current_size + len(remaining)
            if current_size + len(remaining) <= best:
                return
            # Pick a vertex
            v = remaining[0]
            rest = remaining[1:]
            # Option 1: include v
            # All neighbors of v cannot be included
            new_remaining = [u for u in rest if u not in g_adj[v]]
            branch(new_remaining, current_size + 1, current_set | {v})
            # Option 2: exclude v
            branch(rest, current_size, current_set)
        
        branch(order, 0, frozenset())
        return best

    # Verify the base construction is valid
    # We know 8 K4s: alpha = 8 (one from each K4), no C5 (each component is K4, no cycles of length 5)
    # So it's valid. n=32, score = 32/28 = 1.1429

    # Try to extend to n=33
    # Add vertex 32 and try to find edges to add such that:
    # 1. No C5 is created
    # 2. Independence number remains <= 8
    
    # Strategy: try adding vertex 32 with various edge sets to existing vertices
    # If we add vertex 32 with no edges, alpha becomes 9 (independent set: one from each K4 + vertex 32)
    # So vertex 32 must have edges to "break" some independent sets.
    # Specifically, for every independent set of size 8 in the original graph (which are exactly the sets with one vertex from each K4),
    # at least one vertex in that set must be adjacent to vertex 32.
    # This is equivalent to: the neighborhood of vertex 32 must be a "hitting set" for all 8-element independent sets.
    # The 8-element independent sets are in bijection with functions f: {0..7} -> {0,1,2,3} (choosing which vertex in each K4).
    # So we need: for every such choice, at least one chosen vertex is adjacent to 32.
    # Equivalently, the set of non-neighbors of 32 (among the 32 vertices) must NOT contain an independent set of size 8.
    # The non-neighbors form a subgraph H on some subset of the 32 vertices. H must have alpha(H) <= 7.
    
    # Also, adding vertex 32 with edges to some vertices must not create a C5.
    # A C5 involving vertex 32: 32 - a - b - c - d - 32, where a,d are neighbors of 32, and a-b-c-d is a path of length 3.
    # Since each K4 is a clique, any path of length 3 within a K4 is impossible (K4 has diameter 1).
    # So a,b,c,d must span at least 2 K4s.
    # Actually, in the union of disjoint K4s, the only paths are within a single K4 (length 1 or 2 max, since K4 is complete).
    # Wait: in a K4, any two vertices are adjacent. So a path of length 3 (4 vertices) would require 4 distinct vertices, but in a K4 all are mutually adjacent, so a-b-c-d with all edges present is not a "path" in the simple sense... 
    # Actually, a path is a sequence of distinct vertices where consecutive ones are adjacent. In K4, a-b-c is a path of length 2. a-b-c-d would be a path of length 3 if a,b,c,d are distinct. But in K4, a and c are also adjacent, so it's not an induced path, but it's still a walk/path.
    # For a 5-cycle: 32-a-b-c-d-32. We need edges (32,a), (a,b), (b,c), (c,d), (d,32).
    # If a,b,c,d are all in the same K4, then a,b,c,d form a K4, so all edges exist. Then 32-a-b-c-d-32 is a 5-cycle IF 32 is adjacent to both a and d.
    # So if vertex 32 is adjacent to two or more vertices in the same K4, we get a C5!
    # Therefore: vertex 32 can be adjacent to at most 1 vertex in each K4.
    
    # So the constraint is:
    # 1. For each K4 block, vertex 32 is adjacent to 0 or 1 vertex in that block.
    # 2. The set of non-neighbors of vertex 32 has independence number <= 7.
    
    # Let's try all assignments where vertex 32 is adjacent to exactly 1 vertex in each of the 8 K4s (or fewer).
    # There are 4^8 = 65536 such assignments.
    # For each, check if the non-neighbor set has alpha <= 7.
    
    # But we also need to verify no C5 is created. We argued that if at most 1 neighbor per K4, no C5 is created.
    # Let's double-check: a C5 not involving vertex 32 already doesn't exist. A C5 involving 32: 32-a-b-c-d-32.
    # If a and d are in the same K4, and 32 is adjacent to both, that's 2 neighbors in one K4, which we disallow.
    # If a and d are in different K4s, then a-b-c-d must be a path. But a is in K4_i, d is in K4_j (i≠j).
    # For a-b to be an edge, b must be in K4_i (since there are no edges between K4s).
    # For b-c to be an edge, c must be in K4_i (since b is in K4_i and no inter-K4 edges).
    # For c-d to be an edge, c must be in K4_j (since d is in K4_j).
    # So c must be in both K4_i and K4_j, which is impossible.
    # Therefore, no C5 can be created if vertex 32 has at most 1 neighbor per K4.
    
    # So we just need: non-neighbor set has alpha <= 7, and at most 1 neighbor per K4.
    
    # Let's search over all 4^8 assignments (choosing which vertex in each K4 is adjacent to 32, or none).
    # For each assignment, compute the non-neighbor set and check its independence number.
    
    # To speed up, we can use the fact that the non-neighbor set is a subgraph of the original graph.
    # The original graph is 8 disjoint K4s. The non-neighbor set consists of some subset of vertices from each K4.
    # Within each K4, if k vertices are non-neighbors, they form a K_k (clique).
    # So the non-neighbor graph is a disjoint union of cliques (one per K4, of size 0,1,2,3,or 4).
    # The independence number of a disjoint union of cliques is the sum of the independence numbers of each clique.
    # The independence number of K_k is 1 if k>=1, 0 if k=0.
    # So alpha(non-neighbor set) = number of K4s that have at least one non-neighbor.
    
    # Wait, that's not right. If a K4 has k non-neighbors, they form a K_k, so alpha = 1 if k>=1.
    # So total alpha = sum over blocks of (1 if k_b >= 1 else 0), where k_b is the number of non-neighbors in block b.
    
    # We need this sum <= 7.
    # Since there are 8 blocks, this means at most 7 blocks can have non-neighbors.
    # Equivalently, at least 1 block must have ALL vertices as neighbors of 32.
    # But we can have at most 1 neighbor per block!
    # So it's impossible to have all 4 vertices as neighbors in any block.
    
    # Therefore, it's impossible to have alpha(non-neighbor set) <= 7 with at most 1 neighbor per block.
    # Because each block will have at least 3 non-neighbors (since at most 1 neighbor), so each block contributes 1 to alpha.
    # Total alpha = 8 > 7.
    
    # So n=33 is impossible with this construction.
    
    # The current best is n=32.
    
    return {'n': n, 'edges': edges}
