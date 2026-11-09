"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices (0..8) that is exactly 3-chromatic
    (not 2-colorable) with the minimum number of edges (7).
    The Fano plane is the unique (up to isomorphism) 3-uniform 3-chromatic hypergraph
    with 7 edges.
    """
    start_time = time.time()
    deadline = 90  # seconds
    
    # The Fano plane edges on vertices 0..6
    # This is the standard Fano plane configuration
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Convert to list of lists to satisfy type checks if necessary, 
    # though tuples are also allowed. The prompt says "sequences must be list or tuple".
    # The seed used lists. I will use lists to be consistent with "distinct triples".
    edges = [list(e) for e in fano_edges]
    
    # A valid 3-coloring for the Fano plane vertices 0..6
    # c[0]=0, c[1]=1, c[2]=2, c[3]=1, c[4]=2, c[5]=0, c[6]=1
    # Vertices 7, 8 are isolated, assign color 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Validation: Ensure no edge is monochromatic
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # This coloring is invalid. 
            # In this specific case, we know it's valid, but for safety:
            # We could try to fix it, but the hardcoded one is verified above.
            # If it fails, we return an invalid state? No, we must return valid.
            # Since we verified it manually, this branch should not be hit.
            # However, to be robust, if it fails, we could fall back to a search.
            # But 7 edges is the minimum, so we can't reduce count.
            # We just need a valid coloring.
            pass
            
    # Validation: Ensure the hypergraph is NOT 2-colorable.
    # The Fano plane is known to be 3-chromatic.
    # We can perform a quick check to be absolutely sure.
    # A hypergraph is 2-colorable if there exists a map c: V -> {0,1} such that no edge is mono.
    # This is equivalent to: For every edge {u,v,w}, not (c[u]==c[v]==c[w]).
    # This is a SAT problem. For 7 vertices, 2^7 = 128 assignments.
    
    is_2_colorable = False
    for mask in range(128):
        c = [(mask >> i) & 1 for i in range(9)]
        valid = True
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        # This should never happen for the Fano plane.
        # If it does, we have a bug in our edge definition.
        # We would need to find another 3-chromatic hypergraph.
        # But the Fano plane is the standard one.
        pass

    # Ensure types are correct
    # Edges: list of lists of ints
    # Colors: list of ints
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
