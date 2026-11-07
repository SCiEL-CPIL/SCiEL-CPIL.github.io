# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Identify edge types:
    # 1. Cycle edges in layer 0: (i, i+1) for i in 0..m-1 (with wrap)
    # 2. Cycle edges in layer 1: (m+i, m+i+1) for i in 0..m-1 (with wrap)
    # 3. Matching edges: (i, m+i) for i in 0..m-1
    
    # We need to map edge indices to these types.
    # The graph structure is fixed: two m-cycles and a matching.
    # Let's identify the index of each type of edge.
    
    cycle0_indices = []
    cycle1_indices = []
    match_indices = []
    
    for idx, (u, v) in enumerate(edges):
        # Normalize u, v for checking
        u_min, u_max = min(u, v), max(u, v)
        
        # Check if it's a matching edge: one vertex < m, other >= m, and difference is m
        if (u < m and v >= m) or (u >= m and v < m):
            if abs(u - v) == m:
                match_indices.append(idx)
            else:
                # Should not happen for prism graph, but handle gracefully
                # Actually, in a prism, matching is exactly (i, i+m).
                pass 
        else:
            # It's a cycle edge
            if u < m and v < m:
                cycle0_indices.append(idx)
            elif u >= m and v >= m:
                cycle1_indices.append(idx)
            else:
                # Fallback: if we didn't catch it above (e.g. weird ordering)
                # But standard prism edges are only these types.
                pass

    # Sort indices for deterministic access
    cycle0_indices.sort()
    cycle1_indices.sort()
    match_indices.sort()
    
    # Verify we have the right counts
    assert len(cycle0_indices) == m, f"Expected {m} cycle0 edges, got {len(cycle0_indices)}"
    assert len(cycle1_indices) == m, f"Expected {m} cycle1 edges, got {len(cycle1_indices)}"
    assert len(match_indices) == m, f"Expected {m} matching edges, got {len(match_indices)}"
    
    # Precompute the base adjacency matrix structure for fast updates
    # B is n x n. We will update specific rows/cols.
    # However, for exhaustive search, building the matrix from scratch is O(n^2) or O(E).
    # n is small (8 to 18). E is 3m (12 to 27).
    # np.linalg.eigvalsh is O(n^3). For n=18, this is very fast.
    
    def evaluate_signs(signs):
        """
        Compute spectral radius and check validity.
        signs: array of length num_edges with values -1 or 1.
        """
        # Build B
        B = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            B[u, v] = s
            B[v, u] = s
            
        # Compute spectral radius
        try:
            eigvals = np.linalg.eigvalsh(B)
            radius = np.max(np.abs(eigvals))
        except:
            return float('inf'), False
            
        # Check Connectivity
        # The lift is connected iff the product of signs on any "boundary" cycle is +1.
        # Specifically, consider the cycle: 
        # Start at (0,0) in lift.
        # Traverse cycle 0: (0,0) -> (1,0) ... -> (m-1,0). 
        # Sign product on path 0->1->...->m-1 in layer 0: P0 = prod(signs[cycle0])
        # Wait, the lift edges depend on signs.
        # Let's use the standard criterion for 2-lifts of a graph G.
        # The lift is connected iff the holonomy around every independent cycle is trivial?
        # No, for a 2-lift defined by signs s_e in {+1, -1}:
        # The lift has two components if there is a subset of edges S such that...
        # Actually, simpler:
        # The lift is disconnected iff there exists a vertex partition {V0, V1} of the base vertices
        # such that for every edge (u,v):
        #   if u in V0 and v in V0, sign must be +1 (to stay in same layer? No.)
        
        # Let's use the direct BFS on the lift graph for correctness, it's cheap (2n nodes, 6n edges).
        # Or use the cycle product rule.
        # For a prism graph (two cycles + matching), the fundamental cycles are:
        # 1. The two m-cycles.
        # 2. The "ladder" cycles formed by matching edges.
        
        # A 2-lift is connected if and only if the product of signs around every "odd" cycle 
        # (in the sense of the lift topology) is +1?
        # Actually, the condition is: The lift is connected iff the product of signs 
        # on the edges of any cycle that connects the two "sides" is +1?
        
        # Let's stick to BFS on the lift. It's O(n) and n <= 18.
        # Lift nodes: 0..2n-1. (v, 0) is v, (v, 1) is v+n.
        # Edge (u,v) with sign s:
        #   if s=1: (u,0)-(v,0) and (u,1)-(v,1)
        #   if s=-1: (u,0)-(v,1) and (u,1)-(v,0)
        
        # We can represent the lift adjacency implicitly.
        # visited = [False] * (2*n)
        # queue = [0]
        # visited[0] = True
        # while queue:
        #   u = queue.pop()
        #   # find neighbors of u in lift
        #   # u corresponds to base vertex v = u % n, side = u // n
        #   # For each base edge (v, w) with sign s:
        #   #   neighbor_side = side if s==1 else 1-side
        #   #   neighbor = w + neighbor_side * n
        
        # This is fast.
        
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        
        # Precompute base adjacency list with signs for faster BFS
        # adj_base[v] = list of (w, sign_idx)
        # We already have edges. Let's build a static adj list outside the loop?
        # No, signs change. But we can just iterate over edges.
        # Since n is small, even O(E) per BFS step is fine.
        # Total BFS cost O(n * E) = O(m^2). For m=9, 81 ops. Negligible.
        
        while queue:
            node = queue.pop(0)
            v = node % n
            side = node // n
            
            for idx, (u, w) in enumerate(edges):
                if u == v:
                    w_neighbor = w
                elif w == v:
                    w_neighbor = u
                else:
                    continue
                    
                s = signs[idx]
                if s == 1:
                    new_side = side
                else:
                    new_side = 1 - side
                    
                neighbor = w_neighbor + new_side * n
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
                    
        if count != 2 * n:
            return float('inf'), False
            
        return radius, True

    # Strategy: Exhaustive search with symmetry reduction.
    # The prism graph has symmetry. 
    # However, the spectral radius depends on the specific signs.
    # Total edges = 3m.
    # For m=4, 5, 7, 9:
    # m=4: 12 edges -> 2^12 = 4096
    # m=5: 15 edges -> 2^15 = 32768
    # m=7: 21 edges -> 2^21 = 2,097,152
    # m=9: 27 edges -> 2^27 = 134,217,728 -> TOO MANY.
    
    # We need a smarter search for m=9.
    # The connectivity constraint reduces the space by half.
    # 2^26 = 67 million. Still too many for Python loops in 55s.
    # np.linalg.eigvalsh is the bottleneck.
    
    # Alternative: Local Search / Simulated Annealing from a good start.
    # Or: Exploit the structure.
    # The signed adjacency matrix B is block circulant if we order vertices appropriately?
    # Vertices: 0..m-1 (layer 0), m..2m-1 (layer 1).
    # Edges:
    # - Cycle 0: (i, i+1)
    # - Cycle 1: (m+i, m+i+1)
    # - Match: (i, m+i)
    
    # Let's use a hybrid:
    # 1. For m <= 7, exhaustive search is feasible if optimized?
    #    m=7: 2M states. If each state takes ~10us (eigvalsh + BFS), 2M * 10us = 20s. Feasible.
    #    m=9: 134M states. Too slow.
    
    # For m=9, we need a heuristic or smarter exact method.
    # Note that the graph is small.
    # We can use the fact that B is sparse.
    # But eigvalsh on dense 18x18 is very fast in NumPy (C implementation).
    # Let's benchmark roughly: 18x18 eigvalsh is ~1-2 microseconds? No, probably 5-10us.
    # 134M * 10us = 1340s. Too slow.
    
    # So for m=9, we MUST use a search heuristic or reduce the space.
    # The connectivity constraint:
    # Product of signs on any cycle in the "base" that corresponds to a non-trivial holonomy must be +1?
    # Actually, for a 2-lift to be connected, the product of signs on every "twisted" cycle must be +1.
    # In a prism, the cycles that connect the two layers are the ones that go up and down.
    # Consider the cycle: (i, i+1) in layer 0, (i+1, m+i+1) match, (m+i+1, m+i) in layer 1 (backwards), (m+i, i) match.
    # This is a 4-cycle in the base graph (if we consider the ladder).
    # Let's define the constraint more precisely.
    # The lift is disconnected iff there is a "gauge" transformation that makes all signs +1?
    # No, it's disconnected iff the signs are "frustration-free" in a specific sense?
    # Actually, the lift is connected iff the product of signs around every independent cycle of the base graph is +1?
    # No, that would make it a "trivial" lift (two disjoint copies).
    # Wait.
    # If all signs are +1, the lift is two disjoint copies of G. Disconnected.
    # If we flip one sign, does it become connected?
    # For a tree, flipping any edge makes it connected?
    # For a cycle, flipping one sign makes it connected?
    # Let's check m=3 (Triangular Prism).
    # If all +1: Disconnected.
    # If one -1: Connected?
    # The condition for connectivity of a 2-lift is that the product of signs on every "odd" cycle (cycles that lift to paths connecting the two sheets) is -1?
    # No, standard result: The 2-lift is connected iff the product of signs on the edges of every cycle in the base graph is NOT +1?
    # No.
    # Let's use the BFS check. It's reliable.
    
    # To speed up m=9:
    # We can fix the signs of the matching edges?
    # By gauge invariance, we can fix the signs of a spanning tree to +1.
    # A spanning tree of the prism has 2m-1 edges.
    # Total edges 3m.
    # Remaining edges: 3m - (2m-1) = m-1.
    # So we only need to search 2^(m-1) combinations!
    # For m=9, that is 2^8 = 256 combinations.
    # This is extremely fast.
    
    # Algorithm:
    # 1. Find a spanning tree T of the prism graph.
    # 2. For each assignment of signs to the non-tree edges (chords) in {+1, -1}:
    #    a. Determine the signs of the tree edges required to make the lift connected?
    #       No, the signs of the tree edges are free, but they are constrained by the connectivity condition?
    #       Actually, we can FIX the signs of the tree edges to +1 (gauge transformation).
    #       Then, the signs of the chord edges determine the "holonomy" around the fundamental cycles.
    #       The lift is connected if and only if the holonomy around at least one fundamental cycle is -1?
    #       No, the lift is connected if the holonomy group is transitive.
    #       For a 2-lift, the holonomy group is a subgroup of Z2.
    #       It is transitive (connected) if and only if there is a cycle with holonomy -1.
    #       Wait. If ALL cycles have holonomy +1, the lift is disconnected (two copies).
    #       If AT LEAST ONE cycle has holonomy -1, the lift is connected.
    #       
    #       So:
    #       1. Fix tree edge signs to +1.
    #       2. Iterate over all 2^(m-1) assignments of signs to chord edges.
    #       3. For each assignment, compute the holonomy of the fundamental cycles.
    #       4. If any fundamental cycle has product -1, the lift is connected.
    #       5. If connected, compute the spectral radius.
    #       6. Keep the best one.
    #       7. Also, we must consider that fixing tree edges to +1 is a gauge choice.
    #          Does this cover all connected lifts?
    #          Yes, any connected lift is gauge-equivalent to one where the tree edges are +1?
    #          No, gauge transformation changes signs: s_e -> s_e * g_u * g_v where g_v is +1 or -1.
    #          We can always choose g_v to make tree edges +1.
    #          So yes, searching over chord signs with tree signs +1 covers all gauge-equivalence classes.
    #          But do we need to check all gauge-equivalent instances?
    #          The spectral radius is invariant under gauge transformation (similar matrices).
    #          So we just need to check one representative per class.
    #          The representative with tree edges +1 is unique for a given set of chord signs?
    #          Yes.
    
    # This reduces the search space from 2^(3m) to 2^(m-1).
    # m=9 -> 256 evaluations.
    # m=7 -> 64 evaluations.
    # m=5 -> 16 evaluations.
    # m=4 -> 8 evaluations.
    
    # This is incredibly fast. We can even try multiple spanning trees if needed, but one is enough.
    
    # Step 1: Find a spanning tree.
    # Prism graph: Vertices 0..m-1, m..2m-1.
    # Edges:
    # - (i, i+1) for i in 0..m-1 (0 in cycle)
    # - (m+i, m+i+1) for i in 0..m-1 (1 in cycle)
    # - (i, m+i) for i in 0..m-1 (match)
    
    # A simple spanning tree:
    # Take all matching edges (i, m+i) for i in 0..m-1. This connects all vertices?
    # No, it's a matching. It doesn't connect the cycle vertices.
    # We need to connect the cycle vertices.
    # Take matching edges (i, m+i) for all i.
    # Then take cycle 0 edges (i, i+1) for i in 0..m-2. (Skip the last one to avoid cycle).
    # This forms a tree?
    # Vertices: 0..m-1 connected in a path.
    # Each i connected to m+i.
    # m+i are isolated from each other?
    # Yes, so we have m "leaves" m+i attached to the path 0-1-...-m-1.
    # This is a tree.
    # Number of edges: m (matches) + (m-1) (cycle 0 path) = 2m-1. Correct.
    
    # Tree edges:
    # - Matches: (i, m+i) for i in 0..m-1
    # - Cycle 0: (i, i+1) for i in 0..m-2
    # Non-tree edges (chords):
    # - Cycle 0 last edge: (m-1, 0)
    # - Cycle 1 edges: (m+i, m+i+1) for i in 0..m-1 (all m of them)
    # Total chords: 1 + m = m+1?
    # Wait. Total edges 3m. Tree 2m-1. Chords = 3m - (2m-1) = m+1.
    # My count of chords:
    # Cycle 0 has m edges. We used m-1. So 1 chord.
    # Cycle 1 has m edges. We used 0. So m chords.
    # Matches have m edges. We used m. So 0 chords.
    # Total chords = 1 + m = m+1.
    # Search space: 2^(m+1).
    # For m=9, 2^10 = 1024. Still very fast.
    
    # Let's identify the indices of tree edges and chord edges.
    
    tree_edge_indices = []
    chord_edge_indices = []
    
    for idx, (u, v) in enumerate(edges):
        u_min, u_max = min(u, v), max(u, v)
        is_match = (abs(u - v) == m and (u < m) != (v < m))
        
        if is_match:
            tree_edge_indices.append(idx)
        else:
            # Cycle edge
            if u < m and v < m:
                # Cycle 0
                # Tree edges are (i, i+1) for i=0..m-2.
                # Chord is (m-1, 0).
                # Check if it's the wrap-around edge.
                if (u == m-1 and v == 0) or (u == 0 and v == m-1):
                    chord_edge_indices.append(idx)
                else:
                    tree_edge_indices.append(idx)
            else:
                # Cycle 1 (u>=m, v>=m)
                chord_edge_indices.append(idx)
                
    # Verify counts
    assert len(tree_edge_indices) == 2 * m - 1
    assert len(chord_edge_indices) == m + 1
    
    num_chords = len(chord_edge_indices)
    
    best_radius = float('inf')
    best_signs = None
    
    # Iterate over all 2^(num_chords) combinations
    # num_chords is at most 10. 1024 iterations.
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break
            
        # Assign signs
        # Tree edges: +1
        # Chord edges: determined by mask
        signs = np.ones(num_edges)
        
        for j, idx in enumerate(chord_edge_indices):
            if (mask >> j) & 1:
                signs[idx] = -1.0
            else:
                signs[idx] = 1.0
                
        # Check connectivity via holonomy
        # We need to check if ANY fundamental cycle has product -1.
        # Fundamental cycles correspond to chords.
        # For each chord, the cycle is the chord + the unique path in the tree between its endpoints.
        # We need the product of signs on that cycle.
        # Tree signs are all +1.
        # So the product is just the sign of the chord!
        # Wait.
        # If the tree signs are +1, the product on the cycle formed by chord e is just s_e.
        # So if ANY chord has sign -1, the product is -1, and the lift is connected.
        # If ALL chords have sign +1, the product is +1 for all fundamental cycles.
        # Does that imply all cycles have product +1?
        # Yes, the holonomy is determined by the fundamental cycles.
        # So the lift is disconnected if and only if all chord signs are +1.
        # Therefore, we only need to check masks where at least one bit is 1.
        
        if mask == 0:
            continue  # Disconnected
            
        # Compute spectral radius
        radius, valid = evaluate_signs(signs)
        
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # Fallback if no valid solution found (should not happen)
    if best_signs is None:
        # Random search
        rng = np.random.default_rng(3)
        for _ in range(1000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            radius, valid = evaluate_signs(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                
    if best_signs is None:
        # Last resort: All +1 except one chord -1
        signs = np.ones(num_edges)
        signs[chord_edge_indices[0]] = -1.0
        return signs
        
    return best_signs
# EVOLVE-BLOCK-END
