"""Rewritten program for 5x5x5 matrix multiplication tensor decomposition."""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the factor matrices for the schoolbook decomposition of 5x5x5 matrix multiplication.
    Uses NumPy for efficient construction.
    Rank = 125.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for the three dimensions of the tensor
    # The tensor T_{ijk} is 1 if i==j and j==k? No, T_{i j k} corresponds to A_{ij} B_{jk} = C_{ik}.
    # So the non-zero entries are at (i, j, k) for all i, j, k.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (as per contract)
    
    # Generate all combinations of i, j, k
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten them to get the column indices for U, V, W
    # Each (i,j,k) triplet corresponds to one rank-1 term
    # There are n*m*p terms total.
    
    u_rows = (i_idx * m + j_idx).flatten()
    v_rows = (j_idx * p + k_idx).flatten()
    w_rows = (k_idx * n + i_idx).flatten()
    
    # Create sparse-like matrices by scattering 1s
    # Since we need dense lists of lists, we create zero arrays and fill them
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # The column index for the r-th term is simply the flattened index of (i,j,k)
    col_idx = np.arange(rank)
    
    U[u_rows, col_idx] = 1
    V[v_rows, col_idx] = 1
    W[w_rows, col_idx] = 1
    
    # Convert to list of lists to match the expected return type structure if strict, 
    # though numpy arrays are often accepted if the interface allows. 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The seed returned lists. I will return lists to be safe.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
