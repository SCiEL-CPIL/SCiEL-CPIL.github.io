"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    # Peeling strategy: recursively peel off one column of B at a time.
    # For each peeled column k, we compute C[:,k] = A @ B[:,k]
    # which is a 2x3 @ 3x1 multiplication.
    # 
    # Key insight: For 2x3 @ 3x1, we can use rank 3 (not 6) by noting that
    # each term can handle one B[j,k] multiplied with the appropriate A row.
    # Actually, the minimal rank for 2x3 @ 3x1 is 3:
    #   t_j = A_col_j @ B_j  (but this doesn't work directly)
    #
    # Let me reconsider: C[0,k] = sum_j A[0,j]*B[j,k], C[1,k] = sum_j A[1,j]*B[j,k]
    # We need: sum_c u_c[i,j]*v_c[j',k']*w_c[i',k'] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    #
    # For fixed k (peeled column), we need to decompose the 6x3x2 sub-tensor.
    # Using 3 terms per peeled column (one per j):
    #   Term (j,k): u has 1 at (0,j) and (1,j), v has 1 at (j,k), w has 1 at (0,k) and (1,k)
    #   This gives: [A[0,j]+A[1,j]] * B[j,k] * [C[0,k]+C[1,k]] which is WRONG.
    #
    # Correct approach: use 2 terms per (j,k) for the two rows of A:
    #   Term 1 (i=0,j,k): u=e_{(0,j)}, v=e_{(j,k)}, w=e_{(0,k)}
    #   Term 2 (i=1,j,k): u=e_{(1,j)}, v=e_{(j,k)}, w=e_{(1,k)}
    # Total: 2*3*7 = 42 terms (schoolbook)
    #
    # To improve, note that v_c can be shared: for fixed (j,k), both terms use same v.
    # But in tensor decomposition, each term is independent, so we can't "share" v.
    #
    # Alternative: use the fact that we can combine the two A-rows in u.
    # Let's try: for each j,k, use ONE term with:
    #   u = (1 at (0,j), 1 at (1,j)), v = (1 at (j,k)), w = (1 at (0,k), 1 at (1,k))
    # This gives contribution: (A[0,j]+A[1,j])*B[j,k] to BOTH C[0,k] and C[1,k]
    # That's not correct - we need A[0,j]*B[j,k] to C[0,k] and A[1,j]*B[j,k] to C[1,k]
    #
    # So we truly need separate terms. Rank 42 is the schoolbook.
    #
    # Let me try a different approach: use rank 21 by combining across k.
    # For each (i,j), use one term that handles all k:
    #   u = e_{(i,j)}, v = (1 at (j,0), 1 at (j,1), ..., 1 at (j,6)), w = (1 at (i,0), ..., 1 at (i,6))
    # This gives: A[i,j] * (sum_k B[j,k]) to (sum_k C[i,k]) -- WRONG.
    #
    # The issue is that w must have delta_{k,k'} structure.
    #
    # I'll use the schoolbook decomposition which is known to be correct.
    rank = n * m * p
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
