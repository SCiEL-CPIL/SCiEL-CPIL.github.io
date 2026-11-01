import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # The decomposition is:
    # C_{ik} = sum_{r=1}^{rank} U_{i, r} V_{j, r} W_{k, r} ? No.
    # The tensor is T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}
    # But the indices in the problem description are:
    # U: (n*m, rank) -> index i*m + j
    # V: (m*p, rank) -> index j*p + k
    # W: (p*n, rank) -> index k*n + i
    
    # The term in the sum for a specific (i, j, k) is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # We need this product to be 1 if the term (i,j,k) is the one assigned to rank r, and 0 otherwise.
    # Specifically, for each triplet (i, j, k), we assign a unique rank index r.
    # Then U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # All other entries for that rank are 0.
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
