import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m by m x p.
    
    Implements the schoolbook algorithm (rank = n*m*p) which is optimal for 
    the given dimensions (2x4x5) as the tensor rank is n*m*p.
    Uses vectorized NumPy operations for efficient construction.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We iterate over the "middle" dimension j (from 0 to m-1)
    # and the output dimensions i (0 to n-1) and k (0 to p-1).
    # Each combination (i, j, k) corresponds to one rank-1 term.
    
    # Create index arrays for vectorization
    # i ranges 0..n-1, j ranges 0..m-1, k ranges 0..p-1
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten the indices to get the column index in the factor matrices
    # The order must be consistent. Let's use i*m*j + ... no, just linear index.
    # The term index r corresponds to (i, j, k).
    # We can map (i, j, k) to r = i * m * p + j * p + k.
    r_idx = i_idx * (m * p) + j_idx * p + k_idx
    r_flat = r_idx.flatten()
    
    # Calculate row indices for U, V, W
    # U is indexed by (i, j) -> i * m + j
    u_rows = (i_idx * m + j_idx).flatten()
    
    # V is indexed by (j, k) -> j * p + k
    v_rows = (j_idx * p + k_idx).flatten()
    
    # W is indexed by (k, i) -> k * n + i (Trace convention W[k*n+i, r])
    w_rows = (k_idx * n + i_idx).flatten()
    
    # Set the values to 1
    U[u_rows, r_flat] = 1
    V[v_rows, r_flat] = 1
    W[w_rows, r_flat] = 1
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
