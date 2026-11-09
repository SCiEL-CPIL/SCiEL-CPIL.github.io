"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices (0..8) that is 3-chromatic
    with the minimum possible number of edges (7).
    
    The hypergraph is the Fano plane (PG(2,2)) embedded on vertices 0-6,
    with vertices 7 and 8 isolated.
    
    Strategy:
    1. Programmatically generate the 7 edges of the Fano plane using the 
       property of the finite field F_2^3 (XOR operation).
    2. Apply a known valid 3-coloring for the Fano plane vertices.
    3. Return the structure with strict type enforcement.
    """
    
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit

    # 1. Generate Fano plane edges
    # The Fano plane points can be represented by non-zero vectors in F_2^3.
    # We map these to integers 1..7. A line consists of three distinct points 
    # a, b, c such that a + b + c = 0 in F_2^3, which is equivalent to 
    # a XOR b XOR c = 0, or c = a XOR b.
    
    fano_edges_int = set()
    # Iterate over pairs (i, j) with 1 <= i < j <= 7
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = i ^ j
            # To avoid duplicates and ensure we only count each line once,
            # we require k > j. This ensures i < j < k.
            if k > j:
                fano_edges_int.add((i, j, k))
                
    # Convert to 0-indexed vertices 0..6
    # Vertex v corresponds to integer v+1
    edges = []
    for e in fano_edges_int:
        v0 = e[0] - 1
        v1 = e[1] - 1
        v2 = e[2] - 1
        # Sort vertices within the edge for canonical form (though order doesn't strictly matter for hypergraph validity, it helps for determinism)
        edge_list = sorted([v0, v1, v2])
        edges.append(edge_list)
        
    # Sort the list of edges for deterministic output
    edges.sort()

    # 2. Determine a valid 3-coloring
    # We use the verified coloring from the current program which is known to work
    # for the Fano plane structure.
    # Vertices 0..6 are colored to ensure no edge is monochromatic.
    # Vertices 7,8 are isolated, so their colors are arbitrary (set to 0).
    
    # Mapping check for the generated edges against this coloring:
    # Edges generated:
    # {0, 1, 2} -> c: 0, 1, 2 (OK)
    # {0, 3, 4} -> c: 0, 1, 2 (OK)
    # {0, 5, 6} -> c: 0, 0, 1 (OK)
    # {1, 3, 5} -> c: 1, 1, 0 (OK)
    # {1, 4, 6} -> c: 1, 2, 1 (OK)
    # {2, 3, 6} -> c: 2, 1, 1 (OK)
    # {2, 4, 5} -> c: 2, 2, 0 (OK)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # 3. Validation and Type Enforcement
    # Check deadline
    if time.time() > deadline:
        # This is extremely unlikely to trigger for this small graph, 
        # but included per directive.
        pass

    # Verify no monochromatic edges
    for e in edges:
        c0 = colors[e[0]]
        c1 = colors[e[1]]
        c2 = colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Monochromatic edge detected in generated Fano plane")

    # Ensure types are strictly Python int and list
    edges_out = []
    for e in edges:
        edges_out.append([int(e[0]), int(e[1]), int(e[2])])
        
    colors_out = [int(c) for c in colors]
    
    return {
        'edges': edges_out,
        'colors': colors_out
    }

# EVOLVE-BLOCK-END
