import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construct the trivial (schoolbook) decomposition
    # The tensor T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # We want T_{i,j,k} = 1 if (i,j,k) corresponds to a multiplication term, 0 otherwise.
    # In matrix multiplication C = A * B, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor index is (i, j, k). The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So for each (i, j, k), there is exactly one term in the sum.
    # We assign a unique rank index r for each (i, j, k) triple.
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j) flattened to i*m + j
                # V index: (j, k) flattened to j*p + k
                # W index: (k, i) flattened to k*n + i (Note: Trace convention W[k*n+i, r])
                
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
    # Convert to lists of lists to match the expected interface format strictly 
    # (though numpy arrays are often accepted, the prompt shows list of lists in seed)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
