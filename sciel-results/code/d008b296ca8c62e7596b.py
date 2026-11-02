# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    n, m, p = 2, 3, 10
    rank = n * m * p
    
    # Create identity-like structure for the tensor
    # The tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # For schoolbook, each term is unique.
    # U is (n*m) x rank
    # V is (m*p) x rank
    # W is (p*n) x rank
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
