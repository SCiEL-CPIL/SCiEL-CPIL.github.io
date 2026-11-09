"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Fano plane: 7 edges on 7 vertices, 3-uniform, chromatic number 3.
    # This is the minimum 3-uniform 3-chromatic hypergraph.
    # Vertices 0..6 are used for the Fano plane; vertices 7,8 are isolated (unused).
    #
    # Fano plane edges (standard representation):
    # Lines of the Fano plane (projective plane of order 2)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    
    # 3-coloring of the Fano plane:
    # Color assignment: 0->0, 1->0, 2->1, 3->1, 4->2, 5->2, 6->0
    # Let me verify each edge is non-monochromatic:
    # [0,1,2]: 0,0,1 - not mono ✓
    # [0,3,4]: 0,1,2 - not mono ✓
    # [0,5,6]: 0,2,0 - not mono ✓
    # [1,3,5]: 0,1,2 - not mono ✓
    # [1,4,6]: 0,2,0 - not mono ✓
    # [2,3,6]: 1,1,0 - not mono ✓
    # [2,4,5]: 1,2,2 - not mono ✓
    
    colors = [0, 0, 1, 1, 2, 2, 0, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
