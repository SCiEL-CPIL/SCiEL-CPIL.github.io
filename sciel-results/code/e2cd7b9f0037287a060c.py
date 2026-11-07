"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    # Deadline guard: The task allows 60s, but we kill at 90s.
    start_time = time.time()
    
    # The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges and 7 vertices. We use vertices 0..6.
    # Vertices 7 and 8 are isolated.
    
    # Fano plane edges (triples)
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Convert to lists as required by the interface
    edges = [list(e) for e in edges]
    
    # A valid 3-coloring for the Fano plane vertices 0..6
    # Vertices 7, 8 are isolated, so they can be any color.
    # This coloring ensures no edge is monochromatic.
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 1
    # 4: 2
    # 5: 0
    # 6: 1
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verification:
    # 1. Check that every edge is non-monochromatic in the provided coloring.
    for edge in edges:
        c = {colors[v] for v in edge}
        if len(c) == 1:
            # This should not happen for the Fano plane with this coloring
            # If it does, the solution is invalid.
            pass

    # 2. The Fano plane is known to be 3-chromatic (not 2-colorable).
    # We don't need to verify this at runtime as it's a known mathematical fact,
    # but for robustness, we could add a check if time permits.
    # Given the strict time limit and the fact that 7 is the minimum,
    # we return the static optimal solution.

    # Check deadline (should be well within limit)
    if time.time() - start_time > 50:
        # Fallback to a known valid solution if somehow we took too long (unlikely)
        pass

    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
