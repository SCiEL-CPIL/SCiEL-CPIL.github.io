"""Program for matrix multiplication tensor rank minimization (3x4 by 4x6)."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Rank 7 decomposition for 3x4 * 4x6 matrix multiplication
    # Using a structured approach based on bilinear forms
    
    # Let's use a known approach: for 3x4*4x6, we can try to find a decomposition
    # by exploiting the structure. The trivial rank is 3*4*6 = 72.
    # A lower bound is max(n*p, m*p, n*m) = max(18, 24, 12) = 24 for the trivial case,
    # but we can do better with non-trivial decompositions.
    
    # For small cases, we can use known results or construct explicit decompositions.
    # Let's try a decomposition that achieves rank 24 (which is m*p = 24).
    
    # Actually, let me reconsider. The rank of matrix multiplication C = A * B where
    # A is 3x4 and B is 4x6. The tensor has dimensions (3*4) x (4*6) x (6*3) = 12 x 24 x 18.
    
    # A simple decomposition: for each entry C[i,k] = sum_j A[i,j] * B[j,k]
    # This gives rank = 3*4*6 = 72 with the trivial decomposition.
    
    # Let's use a more efficient approach. We can group columns of B.
    # C[:, 0:2] = A * B[:, 0:2]  -- 3x4 * 4x2 = 3x2, rank contribution 3*4*2 = 24
    # C[:, 2:4] = A * B[:, 2:4]  -- 3x4 * 4x2 = 3x2, rank contribution 3*4*2 = 24
    # C[:, 4:6] = A * B[:, 4:6]  -- 3x4 * 4x2 = 3x2, rank contribution 3*4*2 = 24
    # Total: 72. Same as trivial.
    
    # Let me try a different approach: use the fact that we can share some computations.
    # Actually, for exact decomposition, the rank is at least the max of the dimensions
    # of the individual matrix multiplications when done naively.
    
    # Let's just use the trivial decomposition with rank = n*m*p = 72.
    # But wait, we can do better. Let's use rank = m*p = 24 by fixing i and varying j,k
    # in a structured way.
    
    # Actually, the standard result is that the rank of matrix multiplication of m x n by n x p
    # is at most m*n*p and at least max(m*p, m*n, n*p).
    # For 3x4*4x6: at least max(18, 12, 24) = 24.
    
    # Let's construct a rank-24 decomposition:
    # For each j (0..3) and k (0..5), we have a rank-1 term for each i (0..2).
    # That's 3*4*6 = 72 terms. To get rank 24, we need to share across i.
    
    # One approach: C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as: for each j,k, the vector (A[0,j]B[j,k], A[1,j]B[j,k], A[2,j]B[j,k])
    # is a single term that contributes to all three rows of C in column k.
    
    # So for each (j,k) pair, we have one rank-1 term where:
    # U has entries A[i,j] for i=0,1,2 at position (i*m + j)
    # V has entry B[j,k] at position (j*p + k)
    # W has entry 1 at positions (k*n + i) for i=0,1,2... wait, that doesn't work for rank-1.
    
    # For rank-1, we need U, V, W to be vectors such that U ⊗ V ⊗ W matches one slice.
    # The tensor T has entries T[i,j, k,l] = delta... no.
    # The matrix multiplication tensor T_{i,j,k,l} = 1 if j is the summation index, so
    # T[i,j,k,l] = delta_{j,k'} where... 
    
    # Let me be more careful. The tensor for C = A*B is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has indices (i, j) for A, (j, k) for B, (i, k) for C
    # T[i,j, j',k, i',k'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # This is a 6-index tensor that factors as a product of deltas.
    
    # The rank-1 decomposition: each rank-1 term is u_{i,j} v_{j',k} w_{i',k'}
    # with the constraint that sum_r u^{(r)}_{i,j} v^{(r)}_{j',k} w^{(r)}_{i',k'} = T
    # = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    
    # Trivial rank-72: u^{(i,j,k)}_{i',j'} = delta_{i,i'}delta_{j,j'}, 
    #                  v^{(i,j,k)}_{j'',k'} = delta_{j,j''}delta_{k,k'},
    #                  w^{(i,j,k)}_{i'',k''} = delta_{i,i''}delta_{k,k''}
    
    # For rank 24, let's try: for each (j,k), define a term where
    # u^{(j,k)}_{i,j'} = delta_{j,j'} (same for all i)
    # v^{(j,k)}_{j'',k'} = delta_{j,j''}delta_{k,k'}
    # w^{(j,k)}_{i',k'} = delta_{k,k'} (same for all i')
    # Then sum over j,k of u*v*w gives: sum_{j,k} delta_{j,j'}delta_{k,k'} delta_{j,j''}delta_{k,k''} delta_{k,k''}
    # = sum_{j,k} delta_{j,j'}delta_{j,j''}delta_{k,k'}delta_{k,k''}delta_{k,k''}
    # This doesn't seem right.
    
    # Let me just go with the trivial rank-72 decomposition.
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
