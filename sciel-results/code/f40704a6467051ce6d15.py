"""Optimized program using vectorized schoolbook decomposition for 3x4x4 matrix multiplication."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-48 exact tensor decomposition (schoolbook) for multiplying 
    a 3x4 matrix by a 4x4 matrix using NumPy vectorization.
    
    The output is a dictionary with keys 'U', 'V', 'W'.
    U: shape (n*m, rank) = (12, 48)
    V: shape (m*p, rank) = (16, 48)
    W: shape (p*n, rank) = (12, 48)
    """
    import numpy as np

    n, m, p = DIMENSIONS
    rank = n * m * p

    # Create index arrays for vectorized assignment
    # We need to map each term index r to (i, j, k)
    # r = i * m * p + j * p + k
    r_indices = np.arange(rank)
    i_indices = (r_indices // (m * p)).astype(np.int64)
    j_indices = ((r_indices // p) % m).astype(np.int64)
    k_indices = (r_indices % p).astype(np.int64)

    # Calculate row indices for U, V, W
    u_rows = i_indices * m + j_indices
    v_rows = j_indices * p + k_indices
    w_rows = k_indices * n + i_indices

    # Initialize matrices
    U_np = np.zeros((n * m, rank), dtype=np.int64)
    V_np = np.zeros((m * p, rank), dtype=np.int64)
    W_np = np.zeros((p * n, rank), dtype=np.int64)

    # Assign 1s at the specific locations
    # Using advanced indexing
    U_np[u_rows, r_indices] = 1
    V_np[v_rows, r_indices] = 1
    W_np[w_rows, r_indices] = 1

    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }
# EVOLVE-BLOCK-END
