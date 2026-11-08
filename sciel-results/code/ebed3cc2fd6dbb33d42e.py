import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication M_{2,3,7}.
    The rank is n*m*p = 42.
    This implementation uses vectorized numpy operations to generate the factor matrices.
    Includes a deadline guard as per directive.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate the multi-dimensional grid of indices
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    
    # Flatten the grids
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Column indices
    col_indices = np.arange(rank)
    
    # Row indices
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Populate
    U[u_rows, col_indices] = 1
    V[v_rows, col_indices] = 1
    W[w_rows, col_indices] = 1
    
    # Check time, though this operation is instantaneous
    if time.time() > deadline:
        # Fallback or return partial? 
        # Since it's deterministic and fast, this won't trigger.
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
