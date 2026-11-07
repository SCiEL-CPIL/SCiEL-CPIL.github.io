"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import itertools
import time

def construct():
    """Return a 3-uniform hypergraph on 9 vertices with chromatic number exactly 3.
    
    Uses the Fano plane over GF(7) constructed from the multiplicative group structure.
    The Fano plane has 7 edges on vertices 0-6, with vertices 7 and 8 isolated.
    """
    # Start with deadline guard
    deadline = time.time() + 55.0
    
    # Build Fano plane edges using the structure of Z_7
    # The Fano plane can be constructed as lines {x, x+1, x+3} mod 7 for each x
    # This gives 7 distinct edges (each line counted once)
    
    # Generate all edges of form {i, (i+1)%7, (i+3)%7} for i in 0..6
    # But we need to deduplicate since each line is generated multiple times
    edge_set = set()
    for i in range(7):
        edge = tuple(sorted([(i + 0) % 7, (i + 1) % 7, (i + 3) % 7]))
        edge_set.add(edge)
    
    # The above generates 7 distinct edges (the Fano plane)
    # Convert to list of lists
    edges = [list(e) for e in sorted(edge_set)]
    
    # Verify we have exactly 7 edges
    assert len(edges) == 7, f"Expected 7 edges, got {len(edges)}"
    
    # Assign a valid 3-coloring
    # For the Fano plane, we need each edge to have all three colors
    # One valid coloring: vertices 0,5,6 -> color 0; vertices 1,3,4 -> color 1; vertex 2 -> color 2
    # Let's verify and find a proper coloring
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify the coloring is valid (each edge is nonmonochromatic)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        # Each edge must have at least 2 different colors
        assert len({c0, c1, c2}) >= 2, f"Edge {e} is monochromatic"
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
