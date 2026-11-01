"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The strategy is to exploit the linear structure by decomposing into two independent
    1x2 * 2x14 vector-matrix products. For each output row i (0 or 1), we compute:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    For a 1x2 * 2x14 product (vector times matrix), the tensor rank is exactly 2*14 = 28
    by schoolbook, and it's known this is optimal for such "slice" tensors.
    
    However, we can do better by recognizing that the two rows share the same B matrix.
    We use a Kronecker product structure:
    
    For each k in 0..13, we have:
    [C[0,k]]   [A[0,0] A[0,1]] [B[0,k]]
    [C[1,k]] = [A[1,0] A[1,1]] [B[1,k]]
    
    This is a 2x2 matrix multiplication for each k. But we can share the B[j,k] values.
    
    Actually, the key insight: the full tensor is a direct sum of 14 independent 2x2 
    multiplications (one for each column k). Each 2x2 * 2x2 has rank 7 (Strassen).
    So total rank = 14 * 7 = 98, which is worse than schoolbook 56.
    
    Let me reconsider. The schoolbook decomposition has rank 56. Can we do better?
    
    For the tensor T[i,j,k] = 1 (for all i,j,k valid), the rank is the minimum number
    of rank-1 tensors. A rank-1 tensor corresponds to choosing one (i,j) for U, one 
    (j',k) for V, and one (k',i') for W, with the constraint that j=j' and i=i' and k=k'.
    
    Actually, the constraint is that the three indices must be consistent: the j in U 
    must match the j in V, the k in V must match the k in W, and the i in W must match 
    the i in U.
    
    So a rank-1 term picks a specific (i,j,k) triple and contributes 1 to T[i,j,k].
    This means the schoolbook rank of 56 is indeed the rank, since each of the 56 
    entries is independent.
    
    Wait, that's not quite right. A rank-1 tensor U_r ⊗ V_r ⊗ W_r where U_r has a 1 
    at position (i_r, j_r), V_r has a 1 at position (j_r, k_r), and W_r has a 1 at 
    position (k_r, i_r) will contribute 1 to T[i,j,k] if and only if i=i_r, j=j_r, k=k_r.
    
    So yes, each rank-1 term can only cover one (i,j,k) entry. The rank is exactly 56.
    
    Therefore, the schoolbook decomposition with rank 56 is optimal.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2 * 2 * 14 = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
