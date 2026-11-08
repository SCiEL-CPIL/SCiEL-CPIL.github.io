"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for multiplying (n x m) by (m x p) where the tensor is 
    # essentially n copies of the identity structure on the m x p indices (if n>1 
    # and they are independent) is not quite right. 
    # The tensor T_ijk = delta_jk.
    # Slice i is I_m. Rank(I_m) = m.
    # So rank >= m.
    # We found a decomposition of rank m.
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        # U[i, c] = 1 for all i
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = 1
        
        # V[j, c] is non-zero only if j=c
        # V index is j*p + k. We need V[j, c] = delta_jc.
        # In the tensor sum, V contributes V[j, c] and W contributes W[k, c].
        # Wait, V is (m x p) -> (m*p) vector.
        # V[j*p + k, c].
        # We established V[j*p+k, c] = delta_jc * delta_kc.
        # So V[c*p + c, c] = 1.
        V[c * p + c][c] = 1
        
        # W[k, c] = delta_kc.
        # W index is k*n + i.
        # W[k*n + i, c] = delta_kc.
        # So for k=c, W[c*n + i, c] = 1 for all i.
        for i in range(n):
            W[c * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
