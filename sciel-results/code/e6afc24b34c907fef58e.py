"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use block-wise Kronecker product decomposition
    # Treat 3x5 as 1x3 block row of 1x5 vectors
    # Treat 5x6 as 5x1 block column of 1x6 vectors
    # Base tensor for 1x5 by 5x6 has rank 5
    # Total rank = 3 * 5 = 15
    
    rank = n * m  # 3 * 5 = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each block row i (0..n-1), we have a base decomposition of rank m=5
    # The base 1x5 by 5x6 tensor: C_k = sum_j A_j * B_jk
    # This is just the identity-like decomposition where each (j,k) pair contributes
    # But we can do better: the 1x5 by 5x6 multiplication has rank 5
    # Because C_k = sum_j A_j * B_jk, and we can use rank-1 terms
    
    # Actually, the minimal rank for 1x5 by 5x6 is 5 (since it's like matrix-vector product)
    # For each block row i, we use rank 5 decomposition
    # Total rank = n * m = 3 * 5 = 15
    
    col = 0
    for i in range(n):
        for j in range(m):
            # This is the j-th component of the 1x5 vector
            u_idx = i * m + j
            U[u_idx][col] = 1
            
            # For the V side, we need to extract the j-th row of B
            # B is 5x6, so row j of B is B[j, :]
            # For each k, V[j*p + k][col] should be B[j,k] but we're decomposing
            # The rank-1 term: A_j * (row j of B)
            # So for this term, V should have 1s at positions corresponding to row j of B
            for k in range(p):
                v_idx = j * p + k
                V[v_idx][col] = 1
                
            # For W, we need to place the result
            # C_k = sum_j A_j * B_jk
            # For term (i,j), W should contribute to all k with coefficient B_jk
            # But in a rank-1 decomposition, W is fixed per term
            # So we need W[k*n + i] to accumulate A_j * B_jk
            # This means W[w_idx] = B_jk for each k
            for k in range(p):
                w_idx = k * n + i
                W[w_idx][col] = 1  # This is wrong, we need B_jk values
                
            col += 1
    
    # The above is incorrect because we're using 1s everywhere
    # Let me reconsider the decomposition
    
    # The correct approach:
    # C = A * B where A is 3x5, B is 5x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Rank-1 decomposition: C = sum_r U_r * V_r * W_r
    # where U_r[i*m+j], V_r[j*p+k], W_r[k*n+i]
    # 
    # For each r, we have:
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = A[i,j] * B[j,k]
    # 
    # One simple decomposition: for each (i,j,k), use a rank-1 term
    # U[i*m+j] = A[i,j], V[j*p+k] = B[j,k], W[k*n+i] = 1
    # But this gives rank 3*5*6 = 90
    # 
    # Better: group by (i,j) and use the fact that B[j,:] is fixed
    # For fixed (i,j): C[i,k] += A[i,j] * B[j,k]
    # This is a rank-1 outer product of A[i,j] (scalar) with B[j,:] (vector of length p)
    # So for each (i,j), we can use one rank-1 term if we allow W to vary
    # But W is shared across all i for a given k
    # 
    # Actually, let's use the standard decomposition:
    # For each j (0..4), we have a term that handles all i and k
    # U[i*m+j] = A[i,j] for all i
    # V[j*p+k] = B[j,k] for all k  
    # W[k*n+i] = 1 for all i,k
    # But this doesn't work because W would need to be 1 for all (k,i) pairs
    # and we'd be summing over j: sum_j A[i,j]*B[j,k]*1 = C[i,k] ✓
    # 
    # So the decomposition is:
    # rank = m = 5
    # For each j in 0..4:
    #   U[i*m+j] = A[i,j] for i in 0..2
    #   V[j*p+k] = B[j,k] for k in 0..5
    #   W[k*n+i] = 1 for all k,i
    # 
    # But wait, this is a decomposition of the tensor, not using specific A and B values
    # The tensor decomposition must work for ALL A and B
    # So U, V, W must be constant matrices
    # 
    # The correct constant decomposition:
    # For each j, the term is:
    # U[i*m+j] = 1
    # V[j*p+k] = 1  
    # W[k*n+i] = 1
    # This gives sum_j 1*1*1 = 5 for each C[i,k], which is wrong
    # 
    # I need to think about this differently.
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want to decompose this as sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    # 
    # The trivial decomposition has rank 90 with one term per (i,j,k)
    # 
    # A better decomposition: for each j, we can combine all i and k
    # Term j: U[i*m+j] = 1 for all i, V[j*p+k] = 1 for all k, W[k*n+i] = 1 for all i,k
    # But this gives 5 for each entry, not 1
    # 
    # We need W to be selective. If we use W[k*n+i] = 1/(n*p) = 1/18, then
    # sum_j 1*1*(1/18) = 5/18, still not 1
    # 
    # The issue is that the tensor is not separable in a simple way.
    # 
    # Let me just use the trivial rank-90 decomposition for now, which is correct.
    
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
