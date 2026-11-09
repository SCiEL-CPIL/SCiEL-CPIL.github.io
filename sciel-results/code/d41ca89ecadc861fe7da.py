"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 6 x 20
    V = [[0] * rank for _ in range(m * p)]  # 30 x 20
    W = [[0] * rank for _ in range(p * n)]  # 20 x 20

    # Construction: For each k in 0..9, we need to compute C[:,k] = A @ B[:,k]
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We use 2 terms per k (total 20 terms) with clever sharing.
    # Term 2k:   A[0,0]*B[0,k] + A[1,0]*B[0,k] + A[0,1]*B[1,k]
    #             -> This doesn't work as a single rank-1 term.
    
    # Each rank-1 term is: (linear in A) * (linear in B) -> (linear in C)
    # So U[row] gives coefficient for A[row], V[col] for B[col], W[out] for C[out]
    
    # Let me define 20 terms:
    # For k = 0..9:
    #   Term 2k:   U: A[0,0]=1, A[1,0]=1, A[0,1]=1  V: B[0,k]=1, B[1,k]=1  W: C[0,k]=1, C[1,k]=1
    #   This gives: (A[0,0]+A[1,0]+A[0,1]) * (B[0,k]+B[1,k]) -> too many cross terms.
    
    # I need each term to be a SINGLE product of a linear form in A and a linear form in B,
    # contributing to a single (or linear combination of) C entries.
    
    # Rank-1 term: (sum_i u_i A_i) * (sum_j v_j B_j) = sum_{i,j} u_i v_j A_i B_j
    # This contributes to C entries via the W coefficients.
    
    # For exact decomposition, the sum over all terms must equal the tensor.
    
    # Let me try: for each k, use 2 terms that together give all 3 products for each of the 2 rows.
    # That's 6 products per k, but with 2 terms we can only get 2*2=4 independent products if each term
    # is a product of 2 A-factors and 2 B-factors... no, each term is one A-linear * one B-linear.
    
    # Each rank-1 term: (a0*A00 + a1*A01 + a2*A02 + a3*A10 + a4*A11 + a5*A12) * 
    #                   (b0*B0k + b1*B1k + b2*B2k)
    # This gives 6*3 = 18 cross terms, but we only want 6 specific ones.
    
    # This approach is too constrained. Let me just use the naive rank 60 which is guaranteed correct.
    
    # Actually, let me try a simpler optimization: rank 30 by sharing across the 2 rows.
    # For each j in 0..2 and k in 0..9:
    #   Term: A[:,j] * B[j,k] contributes to C[:,k]
    #   That's 2*3*10 = 60 still.
    
    # Let me just return the naive rank 60 which is definitely correct.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
