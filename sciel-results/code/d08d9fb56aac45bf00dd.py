import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # The Fano plane is the minimal 3-chromatic 3-uniform hypergraph.
    # It has 7 edges. It is defined on 7 vertices (0-6).
    # Vertices 7 and 8 are isolated.
    # Edges are triples of vertices.
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]

    # A valid 3-coloring for the Fano plane.
    # v0: 0, v1: 1, v2: 2
    # v3: 0, v4: 2, v5: 1
    # v6: 0
    # Vertices 7, 8: 0 (isolated, any color works)
    colors = [0, 1, 2, 0, 2, 1, 0, 0, 0]

    # Verification: Check if the provided 3-coloring is valid (no monochromatic edges).
    for e in edges:
        c0 = colors[e[0]]
        c1 = colors[e[1]]
        c2 = colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Invalid 3-coloring provided")

    # Verification: Check if the hypergraph is NOT 2-colorable (i.e., 3-chromatic).
    # We check all 2^7 = 128 possible 2-colorings of the 7 active vertices.
    # If any 2-coloring makes all edges non-monochromatic, the graph is 2-colorable.
    is_2_colorable = False
    for mask in range(128):
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break

    if is_2_colorable:
        raise ValueError("Hypergraph is 2-colorable, not 3-chromatic")

    # Ensure types are exactly Python int and list
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1]), int(e[2])])

    final_colors = [int(c) for c in colors]

    return {'edges': final_edges, 'colors': final_colors}
