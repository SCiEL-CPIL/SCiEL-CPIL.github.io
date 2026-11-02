import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
    # is defined by the trilinear form <A, B, C> = Tr(C^T (A B)) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
    # This corresponds to the tensor T_{i,j,k} = 1 for all i, j, k.
    # The rank of the all-ones tensor is 1.
    
    # However, we must be careful. Is the "official" tensor for matmul the all-ones tensor?
    # Yes, in the context of bilinear complexity, the rank is the rank of this tensor.
    # The all-ones tensor of size n x m x p has rank 1.
    
    rank = 1
    
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
