"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal tensor decomposition for the bilinear form
    Tr(A*B) where A is 2x4 and B is 4x4.
    
    The tensor T is defined by T[i,j,k] = 1 if k == i, else 0.
    This corresponds to sum_{i,j} A[i,j] * B[j,i].
    
    The decomposition uses rank R = n * m.
    Each term r corresponds to a pair (i, j).
    U[im+j, r] = 1 if r == im+j else 0.
    V[jp+k, r] = 1 if r == im+j and k == i else 0.
    W[kn+i, r] = 1 if r == im+j and k == i else 0.
    """
    n, m, p = DIMENSIONS
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            
            # U component: picks out the specific (i,j) entry of A
            U[i * m + j][r] = 1
            
            # V component: picks out the specific (j,i) entry of B (since k must be i)
            # V row index is j * p + k. We set k = i.
            V[j * p + i][r] = 1
            
            # W component: picks out the specific (i,i) entry in the trace index space?
            # W row index is k * n + i. We set k = i.
            W[i * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
