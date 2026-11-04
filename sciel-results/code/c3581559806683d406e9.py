import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-30 tensor decomposition for matrix multiplication
    using a vectorized Kronecker-like index structure.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for each term in the decomposition
    # Each column c corresponds to a unique triplet (i, j, k)
    idx = np.arange(rank)
    
    # Decode c into i, j, k based on the mapping c = i*(m*p) + j*p + k
    i = idx // (m * p)
    j = (idx // p) % m
    k = idx % p
    
    # Calculate row indices for U, V, W
    # U is indexed by (i, j) -> i * m + j
    # V is indexed by (j, k) -> j * p + k
    # W is indexed by (k, i) -> k * n + i
    u_rows = i * m + j
    v_rows = j * p + k
    w_rows = k * n + i
    
    # Initialize U, V, W as NumPy arrays for efficient construction
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the 1s using advanced indexing
    # For each column c (which is idx), set the corresponding row to 1
    cols = np.arange(rank)
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists to match the original interface return type
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
