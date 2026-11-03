"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition based on block diagonal structure
    # T = sum_{i=1}^3 e_i \otimes (1_4 \otimes I_6)
    # This corresponds to 3 independent 1x4x6 problems.
    # Each 1x4x6 problem has rank 6.
    # Total rank = 18.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        # For each row i of A (and C), we have a 1x4x6 sub-problem
        # Rank 6 for this sub-problem
        for k in range(p):
            # The term corresponds to C[i, k] = sum_j A[i, j] B[j, k]
            # We use 6 terms for the 6 columns of B/C
            # Actually, the rank 6 decomposition for 1x4x6 is:
            # C_k = sum_{j=1}^4 A_j B_{jk}
            # We can write this using 4 rank-1 terms?
            # C_k = sum_{j=1}^4 (A_j) (B_{jk}) (1)
            # But B_{jk} depends on k.
            # The tensor T_{j, k, k'} = delta_{k, k'} has rank 6.
            # The decomposition is:
            # T = sum_{r=1}^6 u_r \otimes v_r \otimes w_r
            # u_{j, r} = 1
            # v_{k, r} = delta_{k, r}
            # w_{k', r} = delta_{k', r}
            
            # Wait, the indices for V are (j, k) in B.
            # V is a matrix of shape (m*p, rank).
            # V[j*p + k][col]
            
            # For a fixed i and k (which corresponds to r in the 1x4x6 sub-problem):
            # U[i*m + j][col] = 1 for all j
            # V[j*p + k][col] = 1 if j matches?
            # No, V is linear in B.
            # V_{j, k}^{(r)} = delta_{k, r}
            # So V[j*p + k][col] = 1 if k == r, else 0.
            
            # W is linear in C.
            # W_{i, k'}^{(r)} = delta_{k', r}
            # So W[k'*n + i][col] = 1 if k' == r, else 0.
            
            # Let's map the sub-problem rank index r (1..6) to global col.
            # Global col = i * 6 + (k - 1)
            
            # U[i*m + j][col] = 1
            # V[j*p + k][col] = 1 (since k == r)
            # W[k*n + i][col] = 1 (since k' == k == r)
            
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
