import random
import itertools

def construct():
    n = 20
    # 1. Create complete graph edges
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edges.append([u, v, 0]) # Placeholder color
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx
            idx += 1
            
    # 2. Assign a Proper Edge Coloring
    # For K_n, we can use the "round-robin" tournament schedule or similar.
    # For n=20 (even), we can use 19 colors.
    # Simple method: 
    # Fix vertex 0. 
    # Actually, easier: 
    # Use the fact that K_{2m} has a 1-factorization.
    # We can just assign colors based on (u+v) % (n-1) or similar, but need to handle vertex 0 specially or use a known construction.
    # Let's use a standard proper edge coloring for K_n.
    # For even n, colors 0..n-2 are used.
    # Color of (u,v) is (u+v) % (n-1) if u,v != n-1?
    # No, standard construction:
    # Vertices 0..n-2 form a cycle. Vertex n-1 is center.
    # This is tedious to code perfectly from scratch.
    # Alternative: Random proper edge coloring?
    # We can just use random colors and fix conflicts locally? No, that's what we want to avoid.
    
    # Let's use a simple heuristic for proper coloring:
    # Sort edges. Assign colors such that no two adjacent edges share a color.
    # This is edge coloring of a graph.
    # We can use a greedy algorithm for edge coloring.
    
    colors = [0] * len(edges)
    # Greedy edge coloring
    # To make it deterministic and fast, we can just use a random permutation of edges and assign the smallest available color.
    # But we have 20 colors available. K_20 needs 19.
    
    # Let's just use the formula:
    # For K_n with n even:
    # Label vertices 0..n-1.
    # For u < v:
    # If u==n-1 or v==n-1:
    #    color = (min(u,v) + 1) % (n-1) ? No.
    
    # Let's use a library or a robust simple method.
    # Since n=20 is small, we can use a backtracking or just a known pattern.
    # Pattern for K_{2k}:
    # Vertices 0..2k-2 and 2k-1 (center).
    # Edges from center to i have color i.
    # Edges between i, j (i,j < 2k-1) have color (i+j) % (2k-1).
    
    k = n // 2
    center = n - 1
    for i in range(len(edges)):
        u, v, _ = edges[i]
        if u == center:
            u = v
        if v == center:
            v = u
        if u == center or v == center:
            # One is center, one is u
            other = u if v == center else v
            colors[i] = other % (n - 1)
        else:
            # Neither is center
            colors[i] = (u + v) % (n - 1)
            
    # Update edges with colors
    for i in range(len(edges)):
        edges[i][2] = colors[i]

    # 3. Precompute 4-cycles
    # We only care about cycles where opposite edges have the same color.
    # Cycle (a,b,c,d): edges (a,b), (b,c), (c,d), (d,a).
    # Opposite pairs: (a,b) & (c,d), (b,c) & (d,a).
    # In a proper coloring, adjacent edges have different colors.
    # So we only check if color(a,b) == color(c,d) OR color(b,c) == color(d,a).
    
    active = [True] * len(edges)
    
    # Precompute cycle edge indices
    # To speed up, we can just iterate all 4-combinations.
    # 4845 combinations * 3 cycles = ~14k cycles.
    # We can store them.
    
    cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx[(a,b)]
        e2 = edge_to_idx[(b,c)]
        e3 = edge_to_idx[(c,d)]
        e4 = edge_to_idx[(d,a)]
        # Check if this cycle is "potentially bad"
        # It is bad if color(e1)==color(e3) or color(e2)==color(e4)
        # Note: We only care if they are active.
        cycles.append([e1, e2, e3, e4])
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx[(a,b)]
        e2 = edge_to_idx[(b,d)]
        e3 = edge_to_idx[(d,c)]
        e4 = edge_to_idx[(c,a)]
        cycles.append([e1, e2, e3, e4])
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx[(a,c)]
        e2 = edge_to_idx[(c,b)]
        e3 = edge_to_idx[(b,d)]
        e4 = edge_to_idx[(d,a)]
        cycles.append([e1, e2, e3, e4])

    # 4. Greedy Removal
    # Count how many bad cycles each edge is part of.
    # We will iterate until no bad cycles remain.
    
    # To be efficient, we maintain a list of active bad cycles.
    # But updating counts is O(Cycles) per removal.
    # 14k cycles * 190 edges max removals = 2.6M operations. Fast.
    
    # Let's just scan all cycles in each iteration.
    # If we remove 50 edges, 50 * 14k = 700k checks. Very fast.
    
    for _ in range(200): # Max removals
        found_bad = False
        # Calculate conflict counts for all edges
        conflict_count = [0] * len(edges)
        
        # Find all currently bad cycles
        bad_cycles = []
        for cycle in cycles:
            e1, e2, e3, e4 = cycle
            if not (active[e1] and active[e2] and active[e3] and active[e4]):
                continue
            
            c1 = edges[e1][2]
            c2 = edges[e2][2]
            c3 = edges[e3][2]
            c4 = edges[e4][2]
            
            # Proper coloring ensures c1!=c2, c2!=c3, c3!=c4, c4!=c1
            # Bad if c1==c3 or c2==c4
            if c1 == c3 or c2 == c4:
                bad_cycles.append(cycle)
                conflict_count[e1] += 1
                conflict_count[e2] += 1
                conflict_count[e3] += 1
                conflict_count[e4] += 1
                
        if not bad_cycles:
            break
            
        # Find the edge with max conflict count
        max_count = -1
        edge_to_remove = -1
        for i in range(len(edges)):
            if active[i] and conflict_count[i] > max_count:
                max_count = conflict_count[i]
                edge_to_remove = i
                
        if edge_to_remove == -1:
            break
            
        active[edge_to_remove] = False
        found_bad = True
        
    # Construct result
    result = []
    for i, (u, v, c) in enumerate(edges):
        if active[i]:
            result.append([u, v, c])
            
    return result
