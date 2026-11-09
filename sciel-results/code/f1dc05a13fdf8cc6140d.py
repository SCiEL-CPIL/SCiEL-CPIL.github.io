"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the smallest 3-uniform hypergraph that is not 2-colorable.
    # It has 7 edges. This is the minimum possible number of edges for a 
    # 3-uniform hypergraph to be 3-chromatic.
    # We use vertices 0-6 for the Fano plane. Vertices 7-8 are isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices (0-6).
    # We need to ensure no edge is monochromatic.
    # Let's verify the coloring:
    # c(0)=0, c(1)=1, c(2)=2 -> Edge [0,1,2] has colors 0,1,2 (OK)
    # c(3)=1, c(4)=2, c(5)=0 -> Edge [0,3,4] has colors 0,1,2 (OK)
    # c(6)=1 -> Edge [0,5,6] has colors 0,0,1 (OK, not mono)
    # Edge [1,3,5] has colors 1,1,0 (OK)
    # Edge [1,4,6] has colors 1,2,1 (OK)
    # Edge [2,3,6] has colors 2,1,1 (OK)
    # Edge [2,4,5] has colors 2,2,0 (OK)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure all types are strictly Python native
    edges_out = [tuple(int(v) for v in e) for e in edges]
    colors_out = tuple(int(c) for c in colors)
    
    return {
        'edges': edges_out,
        'colors': colors_out
    }

# EVOLVE-BLOCK-END
