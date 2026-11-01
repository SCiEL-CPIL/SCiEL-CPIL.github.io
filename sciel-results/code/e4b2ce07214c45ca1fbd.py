import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook decomposition for 2x2 * 2x13 matrix multiplication.
    Returns rank 52.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = n * m * p

    # Initialize with zeros
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Convert to list of lists as required by interface?
    # The interface says: 'factor matrices of shapes ... with integer/half-integer entries'
    # Usually numpy arrays are accepted if they can be indexed, but the seed used lists.
    # Let's stick to lists to be safe with the "integer/half-integer entries" check if it expects scalars.
    # However, using numpy to build and then converting is a valid structural change.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
