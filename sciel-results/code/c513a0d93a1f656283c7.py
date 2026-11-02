"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # We seek a rank-minimal exact tensor decomposition for (2x2) * (2x14) -> (2x14).
    # The multiplication tensor T_{i,k,j,l} = delta_{i,j} (output index (i,k), input (j,l)).
    # We view this as a linear map from the 28-dim space of B (indexed by (j,l)) to the 28-dim space of C (indexed by (i,k)),
    # for each fixed A (2x2). Actually the bilinear map is (A, B) -> C.
    # The tensor is T in V_{2x2}^* \otimes V_{2x14}^* \otimes V_{2x14}.
    # We can write C_{i,k} = sum_{j,l} A_{i,j} B_{j,l} delta_{k,l}.
    # This is a bilinear map. The rank of this tensor is the minimum number of rank-1 tensors needed.
    #
    # Key insight: The map B -> C for fixed A is linear and given by (I_14 \otimes A) acting on vec(B).
    # But the full tensor T is the "multiplication tensor". Its rank is related to the rank of the linear map
    # from A \otimes B to C, which is the Kronecker product structure.
    #
    # Actually, for matrix multiplication (n x m) * (m x p), the tensor rank is known to be at most n*m*p (schoolbook).
    # For 2x2 * 2x2, Strassen gives 7. For 2x2 * 2xp, we can use a block approach.
    #
    # Let's think of B as 7 blocks of 2x2: B = [B_0, B_1, ..., B_6] where each B_b is 2x2.
    # Then C = A * B = [A*B_0, A*B_1, ..., A*B_6].
    # Each A*B_b is a 2x2 matrix product. Using Strassen for each 2x2*2x2 product takes 7 multiplications per block.
    # So total rank = 7 blocks * 7 Strassen multiplications = 49.
    #
    # But wait, can we do better by sharing computations across blocks?
    # The Strassen algorithm for 2x2 * 2x2 uses 7 scalar multiplications. The inputs are linear combinations of entries of A and B.
    # For 7 blocks, we have 7 different B blocks. The A is shared.
    # In Strassen, the 7 "multiplications" are of the form M_k = (linear combo of A) * (linear combo of B).
    # Since A is the same for all blocks, the linear combos of A are the same across blocks.
    # So we can compute the 7 linear combos of A once, and then for each block, compute the 7 linear combos of B_b,
    # and then do 7 multiplications per block. That's still 7*7 = 49 multiplications.
    #
    # However, the tensor rank might be lower if we can share the B-side linear combinations across blocks in a smarter way.
    # But each B_b is independent (different entries of B), so the linear combinations for different blocks involve different entries of B.
    # So there's no sharing possible on the B side between blocks.
    #
    # Therefore, the rank is 49 via this block-Strassen approach.
    #
    # Let's verify: schoolbook rank is 2*2*14 = 56. Block-Strassen gives 49.
    # Is 49 minimal? The lower bound for matrix multiplication tensor rank is not well-known in general,
    # but for 2x2 * 2xp, the block-Strassen approach is likely optimal or near-optimal.
    #
    # Let's implement the block-Strassen decomposition.
    #
    # Strassen for 2x2 * 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For our case, A is 2x2, and B is 2x14, split into 7 blocks of 2x2.
    # For each block b (0..6), B_b is 2x2.
    # The 7 Strassen multiplications for block b are:
    #   M1_b = (a11 + a22) * (B_b[0,0] + B_b[1,1])
    #   M2_b = (a21 + a22) * B_b[0,0]
    #   M3_b = a11 * (B_b[0,1] - B_b[1,1])
    #   M4_b = a22 * (B_b[1,0] - B_b[0,0])
    #   M5_b = (a11 + a12) * B_b[1,1]
    #   M6_b = (a21 - a11) * (B_b[0,0] + B_b[0,1])
    #   M7_b = (a12 - a22) * (B_b[1,0] + B_b[1,1])
    #
    # So total 7*7 = 49 multiplications.
    #
    # Now we need to express this as U, V, W matrices.
    # U has shape (n*m, rank) = (4, 49)
    # V has shape (m*p, rank) = (28, 49)
    # W has shape (p*n, rank) = (28, 49)
    #
    # The tensor decomposition is: T_{i,k,j,l} = sum_{r=1}^{rank} U_{i*m+j, r} * V_{j*p+l, r} * W_{k*n+i, r}
    # Wait, the trace convention is W[k*n+i, r]. Let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # So the output index is (k, i) where k is the column of C (0..p-1) and i is the row of C (0..n-1).
    # The input indices are: U is indexed by (i, j) where i is row of A, j is column of A (which is row of B).
    # V is indexed by (j, l) where j is row of B, l is column of B.
    # So the bilinear form is: C_{i,k} = sum_{j,l} A_{i,j} B_{j,l} delta_{k,l}.
    # And the tensor decomposition is: T_{i,k,j,l} = sum_r U_{i,j,r} V_{j,l,r} W_{k,i,r}
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    #
    # Let's index U by (i, j) -> row i*m + j.
    # V by (j, l) -> row j*p + l.
    # W by (k, i) -> row k*n + i.
    #
    # For the block-Strassen approach:
    # For each block b (0..6) and each Strassen step s (0..6), we have one multiplication term.
    # The term contributes to C_{i,k} where k = 2*b + k_local (k_local is 0 or 1 within the block).
    # And i is 0 or 1.
    #
    # Let's define for each (b, s):
    #   The A-side linear combination: alpha_s (a vector of length 4, indexed by (i,j))
    #   The B-side linear combination: beta_{b,s} (a vector of length 28, indexed by (j,l))
    #   The C-side coefficient: gamma_{b,s} (a vector of length 28, indexed by (k,i))
    #
    # Then U_{i,j, r(b,s)} = alpha_s[i,j]
    # V_{j,l, r(b,s)} = beta_{b,s}[j,l]
    # W_{k,i, r(b,s)} = gamma_{b,s}[k,i]
    #
    # Let's work out the Strassen coefficients.
    # For a 2x2 * 2x2 product, the 7 multiplications are:
    # s=0: M1 = (a11+a22)(b11+b22)
    #   alpha_0 = [1, 0, 0, 1]  (a11, a12, a21, a22)
    #   beta_0  = [1, 0, 0, 1]  (b11, b12, b21, b22)
    #   gamma_0: C11 += 1, C12 += 0, C21 += 0, C22 += 1
    #             So gamma_0[0,0]=1, gamma_0[0,1]=0, gamma_0[1,0]=0, gamma_0[1,1]=1
    #             In (k,i) indexing: k=0,i=0 -> 1; k=0,i=1 -> 0; k=1,i=0 -> 0; k=1,i=1 -> 1
    #             So gamma_0 = [1, 0, 0, 1] in (k,i) order: (0,0),(0,1),(1,0),(1,1)
    #
    # s=1: M2 = (a21+a22)b11
    #   alpha_1 = [0, 0, 1, 1]
    #   beta_1  = [1, 0, 0, 0]
    #   gamma_1: C21 += 1, C22 += -1
    #             gamma_1[0,0]=0, gamma_1[0,1]=0, gamma_1[1,0]=1, gamma_1[1,1]=-1
    #
    # s=2: M3 = a11(b12-b22)
    #   alpha_2 = [1, 0, 0, 0]
    #   beta_2  = [0, 1, 0, -1]
    #   gamma_2: C12 += 1, C22 += 1
    #             gamma_2[0,0]=0, gamma_2[0,1]=1, gamma_2[1,0]=0, gamma_2[1,1]=1
    #
    # s=3: M4 = a22(b21-b11)
    #   alpha_3 = [0, 0, 0, 1]
    #   beta_3  = [-1, 0, 1, 0]
    #   gamma_3: C11 += -1, C21 += 1
    #             gamma_3[0,0]=-1, gamma_3[0,1]=0, gamma_3[1,0]=1, gamma_3[1,1]=0
    #
    # s=4: M5 = (a11+a12)b22
    #   alpha_4 = [1, 1, 0, 0]
    #   beta_4  = [0, 0, 0, 1]
    #   gamma_4: C11 += 1, C22 += -1
    #             gamma_4[0,0]=1, gamma_4[0,1]=0, gamma_4[1,0]=0, gamma_4[1,1]=-1
    #
    # s=5: M6 = (a21-a11)(b11+b12)
    #   alpha_5 = [-1, 0, 1, 0]
    #   beta_5  = [1, 1, 0, 0]
    #   gamma_5: C22 += 1
    #             gamma_5[0,0]=0, gamma_5[0,1]=0, gamma_5[1,0]=0, gamma_5[1,1]=1
    #
    # s=6: M7 = (a12-a22)(b21+b22)
    #   alpha_6 = [0, 1, 0, -1]
    #   beta_6  = [0, 0, 1, 1]
    #   gamma_6: C11 += -1, C12 += 1
    #             gamma_6[0,0]=-1, gamma_6[0,1]=1, gamma_6[1,0]=0, gamma_6[1,1]=0
    #
    # Now for block b, the B entries are B_b[0,0], B_b[0,1], B_b[1,0], B_b[1,1].
    # In the full B (2x14), these are at positions:
    #   B[0, 2*b+0], B[0, 2*b+1], B[1, 2*b+0], B[1, 2*b+1]
    # So for V, the row index is j*p + l = j*14 + l.
    # For block b, the relevant V entries are:
    #   j=0, l=2*b -> row 0*14 + 2*b = 2*b
    #   j=0, l=2*b+1 -> row 2*b+1
    #   j=1, l=2*b -> row 1*14 + 2*b = 14+2*b
    #   j=1, l=2*b+1 -> row 14+2*b+1
    #
    # For W, the output C entries for block b are at positions:
    #   C[0, 2*b], C[0, 2*b+1], C[1, 2*b], C[1, 2*b+1]
    # In (k,i) indexing: k=2*b or 2*b+1, i=0 or 1.
    # W row index = k*n + i = k*2 + i.
    # So for block b:
    #   k=2*b, i=0 -> row 4*b
    #   k=2*b, i=1 -> row 4*b+1
    #   k=2*b+1, i=0 -> row 4*b+2
    #   k=2*b+1, i=1 -> row 4*b+3
    #
    # Now let's build U, V, W.
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    # Strassen alpha vectors (for A, indexed by (i,j): (0,0),(0,1),(1,0),(1,1))
    alphas = [
        [1, 0, 0, 1],   # s=0
        [0, 0, 1, 1],   # s=1
        [1, 0, 0, 0],   # s=2
        [0, 0, 0, 1],   # s=3
        [1, 1, 0, 0],   # s=4
        [-1, 0, 1, 0],  # s=5
        [0, 1, 0, -1],  # s=6
    ]

    # Strassen beta vectors (for B, indexed by (j,l): (0,0),(0,1),(1,0),(1,1))
    betas = [
        [1, 0, 0, 1],   # s=0
        [1, 0, 0, 0],   # s=1
        [0, 1, 0, -1],  # s=2
        [-1, 0, 1, 0],  # s=3
        [0, 0, 0, 1],   # s=4
        [1, 1, 0, 0],   # s=5
        [0, 0, 1, 1],   # s=6
    ]

    # Strassen gamma vectors (for C, indexed by (k,i): (0,0),(0,1),(1,0),(1,1))
    gammas = [
        [1, 0, 0, 1],    # s=0
        [0, 0, 1, -1],   # s=1
        [0, 1, 0, 1],    # s=2
        [-1, 0, 1, 0],   # s=3
        [1, 0, 0, -1],   # s=4
        [0, 0, 0, 1],    # s=5
        [-1, 1, 0, 0],   # s=6
    ]

    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            # U: alpha_s for A
            for i in range(2):
                for j in range(2):
                    U[i * m + j][r] = alphas[s][i * 2 + j]
            # V: beta_s for block b of B
            for j in range(2):
                for l_local in range(2):
                    l = 2 * b + l_local
                    V[j * p + l][r] = betas[s][j * 2 + l_local]
            # W: gamma_s for block b of C
            for k_local in range(2):
                for i in range(2):
                    k = 2 * b + k_local
                    W[k * n + i][r] = gammas[s][k_local * 2 + i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
