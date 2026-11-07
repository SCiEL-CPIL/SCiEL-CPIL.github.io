"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 16
    half = n // 2
    
    # Generate all pairs (u, v) where u in [0, half) and v in [half, n)
    # This creates the complete bipartite graph K_{8,8}
    
    # Create meshgrid for the two partitions
    # u_coords will have shape (8, 8) with rows repeating 0..7
    # v_coords will have shape (8, 8) with columns repeating 8..15
    u_coords, v_coords = np.meshgrid(
        np.arange(half), 
        np.arange(half, n), 
        indexing='ij'
    )
    
    # Flatten to 1D arrays of coordinates
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    
    # Stack into shape (64, 2)
    edges_np = np.stack([u_flat, v_flat], axis=-1)
    
    # Convert to list of lists of Python ints to satisfy type checking
    # np.int64 is not Python int, so we must convert
    edges = [[int(u), int(v)] for u, v in edges_np]
    
    return edges
# EVOLVE-BLOCK-END
