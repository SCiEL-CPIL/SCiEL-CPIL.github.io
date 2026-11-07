import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 58s safety margin (evaluator kills at 90s)
    
    # The Fano plane (7 edges, 7 vertices) is the unique minimal 3-uniform
    # 3-chromatic hypergraph. We embed it on vertices 0..6 and leave 7, 8 isolated.
    # This gives exactly 7 edges, which is the theoretical minimum.
    
    # Fano plane edges on vertices 0-6
    # Standard Fano plane: lines of the projective plane of order 2
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verify: each pair of vertices appears in exactly one edge (Fano plane property)
    # Verify: no edge is monochromatic under the 3-coloring below
    
    # 3-coloring: vertices 0-6 get colors cycling 0,1,2,0,1,2,0
    # Vertices 7, 8 are isolated, assign color 0
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify no edge is monochromatic
    for e in fano_edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # Should not happen, but if it does, try a different coloring
            raise ValueError("Monochromatic edge found")
    
    # Verify the hypergraph is not 2-colorable (i.e., no 2-coloring avoids monochromatic edges)
    # For the Fano plane, this is a well-known fact, but let's verify for safety
    # A 2-coloring assigns each vertex color 0 or 1. An edge is monochromatic if all 3 vertices have the same color.
    # The Fano plane is not 2-colorable: for any 2-coloring of its 7 vertices, at least one line is monochromatic.
    
    # Brute-force check (2^7 = 128 colorings, trivial)
    def is_two_colorable(edges, n):
        for mask in range(1 << n):
            coloring = [(mask >> i) & 1 for i in range(n)]
            valid = True
            for e in edges:
                if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    # Only check on vertices 0-6 (the Fano plane part)
    if is_two_colorable(fano_edges, 7):
        # This should never happen for the Fano plane, but if it does, we have a problem
        raise ValueError("Fano plane appears to be 2-colorable, which is impossible")
    
    # Ensure all types are exactly Python int
    final_edges = []
    for e in fano_edges:
        final_edges.append(tuple(int(v) for v in e))
    
    final_colors = tuple(int(c) for c in colors)
    
    # Deadline guard
    if time.time() > deadline:
        pass  # We're already done, this is just a safety check
    
    return {
        'edges': final_edges,
        'colors': final_colors
    }
