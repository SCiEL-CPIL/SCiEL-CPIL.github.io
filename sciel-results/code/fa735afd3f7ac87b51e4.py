import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    This implementation uses a systematic index mapping to generate the schoolbook
    decomposition, which serves as the baseline minimal rank candidate for these dimensions.
    """
    n, m, p = DIMENSIONS
    
    # The schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros using NumPy for efficient memory layout
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate the matrices based on the tensor decomposition identity:
    # T_{ijk} = sum_t U_{it} V_{jt} W_{kt}
    # For the schoolbook method, each term t isolates a single (i, j, k) contribution.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Calculate the term index t
                t = i * m * p + j * p + k
                
                # Set the non-zero entries
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                
    # Return as lists to match the expected interface format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
