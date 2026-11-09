"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Construct a 3-uniform hypergraph on 9 vertices that is exactly 3-chromatic
    with the minimum number of edges (7).
    
    We generate the Fano plane on vertices 0-6 using GF(2)^3 arithmetic.
    Vertices 7 and 8 are added with color 0 and no edges (or edges that don't reduce chromatic number).
    We then find a valid 3-coloring for the Fano plane vertices.
    """
    # Generate Fano plane edges on vertices 0..6
    # Vertices correspond to non-zero vectors in GF(2)^3
    # Map index to vector
    vecs = []
    for i in range(1, 8):
        v = (i & 1, (i >> 1) & 1, (i >> 2) & 1)
        vecs.append(v)
    
    edges = []
    # Find all triples (i, j, k) such that vecs[i] + vecs[j] + vecs[k] == (0,0,0)
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                # Sum in GF(2)^3
                s0 = vecs[i][0] ^ vecs[j][0] ^ vecs[k][0]
                s1 = vecs[i][1] ^ vecs[j][1] ^ vecs[k][1]
                s2 = vecs[i][2] ^ vecs[j][2] ^ vecs[k][2]
                if s0 == 0 and s1 == 0 and s2 == 0:
                    edges.append([i, j, k])
    
    # Find a valid 3-coloring for vertices 0..6
    # Backtracking search
    def is_valid_coloring(colors, edges):
        for e in edges:
            c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
            if c0 == c1 == c2:
                return False
        return True

    def backtrack(idx, colors, edges):
        if idx == 7:
            return is_valid_coloring(colors, edges)
        for c in range(3):
            colors[idx] = c
            if backtrack(idx + 1, colors, edges):
                return True
        colors[idx] = -1
        return False

    colors_fano = [-1] * 7
    if not backtrack(0, colors_fano, edges):
        # Fallback to known coloring if backtracking fails (should not happen)
        colors_fano = [0, 0, 1, 1, 2, 2, 0]
    
    # Extend colors to 9 vertices
    # Vertices 7 and 8 can be any color, let's use 0
    colors = colors_fano + [0, 0]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
