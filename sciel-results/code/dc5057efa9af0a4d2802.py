import itertools

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges.
    # It is known to be 3-chromatic (not 2-colorable, but 3-colorable).
    # We use vertices 0..6 for the Fano plane and 7,8 are isolated.
    # This gives a 3-chromatic hypergraph on 9 vertices with 7 edges.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Find a valid 3-coloring for the Fano plane part.
    # Brute force 3^7 = 2187 colorings, which is very fast.
    found_coloring = None
    for c in itertools.product([0, 1, 2], repeat=7):
        valid = True
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                valid = False
                break
        if valid:
            # Extend to 9 vertices, assigning 0 to vertices 7 and 8
            found_coloring = list(c) + [0, 0]
            break
            
    if found_coloring is None:
        # Fallback, though Fano plane is always 3-colorable
        found_coloring = [0, 1, 2, 0, 1, 2, 0, 0, 0]
        
    return {'edges': [list(e) for e in edges], 'colors': found_coloring}
