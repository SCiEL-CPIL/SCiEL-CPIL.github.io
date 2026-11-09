import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank decomposition for matrix multiplication M_{3,4,5}.
    Uses the standard basis decomposition (rank n*m*p).
    This approach vectorizes the index generation for efficiency.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for the triple loop (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten the indices to correspond to the rank dimension
    # The rank dimension is the iteration order.
    # We need to map (i, j, k) to a unique column index.
    # The current seed uses col = 0..rank-1 in lexicographic order of (i, j, k).
    # meshgrid with indexing='ij' produces arrays where i varies slowest?
    # Let's check:
    # i_idx[0,0,0]=0, i_idx[0,0,1]=0... i_idx[0,1,0]=0... i_idx[1,0,0]=1
    # So i is slowest, k is fastest.
    # This matches the loop:
    # for i: for j: for k:
    # So the flattened array of indices will be in the correct order if we flatten C-style.
    
    flat_i = i_idx.flatten()
    flat_j = j_idx.flatten()
    flat_k = k_idx.flatten()
    
    # Calculate row indices for U, V, W
    u_rows = flat_i * m + flat_j
    v_rows = flat_j * p + flat_k
    w_rows = flat_k * n + flat_i
    
    # Column indices are just 0, 1, ..., rank-1
    cols = np.arange(rank)
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill the matrices at the specified positions
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
