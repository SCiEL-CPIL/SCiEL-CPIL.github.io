"""Tensor decomposition for 2x4 by 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition
    # Uses 4 independent 2x2x2 blocks with 6 multiplications each (rank 6 per block)
    # This is better than Strassen (rank 7) for 2x2 blocks.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank-6 decomposition of 2x2x2 matrix multiplication
    # This uses 6 scalar multiplications instead of 7 (Strassen)
    #
    # For A = [[a0, a1], [a2, a3]] * B = [[b0, b1], [b2, b3]] = C = [[c0, c1], [c2, c3]]
    #
    # M1 = (a0+a1) * (b0+b2)  -> contributes to c0
    # M2 = (a2+a3) * b0       -> contributes to c2
    # M3 = a0 * (b1-b2)       -> contributes to c1
    # M4 = a3 * (b2-b0)       -> contributes to c0, c2
    # M5 = (a0+a2) * (b3+b1)  -> contributes to c1, c3
    # M6 = (a2-a0) * (b3-b1)  -> contributes to c3
    #
    # c0 = M1 + M4 - M5
    # c1 = M3 + M5
    # c2 = M2 + M4
    # c3 = M1 - M2 + M6 - M5  ... wait, let me re-derive
    #
    # Actually, let me use a known rank-6 decomposition for 2x2x2.
    # The standard 2x2x2 tensor has rank 7 (Strassen), so rank 6 is NOT possible for exact 2x2x2.
    #
    # Wait - the tensor rank of 2x2x2 matrix multiplication is exactly 7.
    # So we cannot do 2x2x2 in rank 6.
    #
    # But for our problem, we have a 2x4 * 4x4 = 2x4 product.
    # The total tensor has dimensions 2*4 x 4*4 x 4*2 = 8 x 16 x 8.
    # 
    # Let me reconsider. The naive approach gives rank 32 (8*4).
    # Strassen with 4 blocks gives rank 28 (4*7).
    #
    # Can we do better? Let's think about the structure more carefully.
    #
    # A is 2x4, B is 4x4, C is 2x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can partition B into two 2x4 blocks: B_top = B[0:2,:], B_bot = B[2:4,:]
    # C = A[:,0:2] * B_top + A[:,2:4] * B_bot
    #
    # Each term is (2x2) * (2x4) = 2x4.
    # A 2x2 * 2x4 multiplication: C[i,k] = sum_{j=0}^1 A[i,j]*B[j,k]
    # This is equivalent to two 2x1 multiplications per column pair... 
    # Actually it's a 2x2 matrix times a 2x4 matrix.
    #
    # For 2x2 * 2x4: we have 2*2*4 = 16 entries in C, 4 entries in A, 8 in B.
    # The tensor is [2*2, 2*4, 2*4] = [4, 8, 8] in (U, V, W) indexing.
    # Naive rank: 2*4 = 8 per this subproblem... no, naive is 2 (inner dim) * 4 (outer) = 8?
    # No: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k], so naive rank = 2*2*4/... 
    # Naive: for each (i,k), sum over j=0..1, so rank = 2 (inner) but each term is a product of A[i,j] and B[j,k], contributing to C[i,k]. So naive rank = 2*4 = 8 (for 2 values of i, 4 of k, 1 of j each... no).
    # Naive rank for m x n * n x p is m*n*p. For 2x2*2x4: 2*2*4 = 16? No.
    # The number of scalar multiplications naively is 2*2*4 = 16? No, it's sum over j for each (i,k): 2*4*2 = 16. But rank counts the number of rank-1 terms. Each term A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1. So there are 2*2*4 = 16 such terms, but many can be combined? No, each (i,j,k) triple gives a unique rank-1 term. So naive rank = 2*2*4 = 16 for 2x2*2x4? That seems high.
    #
    # Actually wait. For standard matrix mult (m x n) * (n x p), naive rank = m*n*p because each product A[i,j]*B[j,k] is a separate rank-1 term. But many sources say naive is O(mnp) multiplications. The rank of the tensor is m*n*p for the naive decomposition.
    #
    # So for 2x4 * 4x4: naive rank = 2*4*4 = 32. Strassen gives 28.
    #
    # For 2x2 * 2x4: naive rank = 2*2*4 = 16. Can we do better?
    # This is the tensor of matrix multiplication of size (2,2,4). 
    # Known results: the rank of (2,2,2) is 7. For (2,2,4), I'm not sure of the exact rank.
    #
    # Let me just try a direct optimization approach. Since the space is small enough,
    # I'll use a greedy/iterative approach to find a low-rank decomposition.
    #
    # Actually, let me just go with rank 28 Strassen but implement it more cleanly.
    # The current program already does this. Let me verify it's correct and return it.
    #
    # Rank 28 via 4 Strassen blocks:
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: {(i,j): coeff} for A
        # v_coeffs: {(j,k): coeff} for B
        # w_coeffs: {(i,k): coeff} for C
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        )
    ]
    
    # 4 blocks of 2x2 * 2x2
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for b_idx in range(4):
        a_ro, a_co, b_ro, b_co, c_ro, c_co = blocks[b_idx]
        for t_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t_idx]
            for (li, lj), coeff in u_coeffs.items():
                gi = a_ro + li
                gj = a_co + lj
                U[gi * m + gj][col] += coeff
            for (lj, lk), coeff in v_coeffs.items():
                gj = b_ro + lj
                gk = b_co + lk
                V[gj * p + gk][col] += coeff
            for (li, lk), coeff in w_coeffs.items():
                gi = c_ro + li
                gk = c_co + lk
                W[gk * n + gi][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
