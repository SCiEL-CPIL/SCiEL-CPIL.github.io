import numpy as np

def construct():
    n, m, p = 4, 4, 5

    # Build the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    # Conventions:
    #   U[a*2+b, t]  (A factor, a=row, b=col)
    #   V[b*2+c, t]  (B factor, b=row, c=col)
    #   W[c*2+a, t]  (C factor, c=row, a=col)
    # such that sum_t U[...,t] V[...,t] W[...,t] reproduces the 2x2x2 tensor.

    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # Term 0: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # Term 1: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # Term 2: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # Term 3: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # Term 4: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # Term 5: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # Term 6: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 4x4x4 multiplication tensor: rank 49 = 7 * 7 via block Strassen.
    # View 4x4 matrices as 2x2 blocks of 2x2 blocks.
    # The 49 terms are indexed by (outer_s, inner_t) for s, t in 0..6.
    # For a fixed outer term s, the 4x4 A and B are linear combinations of 2x2 blocks.
    # We need to express the 4x4x4 tensor as sum over s,t of (rank-1 in 4x4x4).

    # Actually, the standard recursive Strassen for 4x4x4 works as follows:
    # The 4x4x4 tensor T4 can be written as sum_{s=0}^{6} T_s where each T_s
    # corresponds to one of the 7 Strassen terms at the block level.
    # Each T_s is a 4x4x4 tensor that is itself a product of 2x2 matrices of 2x2 blocks.
    # To decompose T_s into rank-1 terms, we apply the 2x2x2 Strassen decomposition
    # to the 2x2 blocks.

    # More concretely, for 4x4 matrices A, B:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # where each Aij, Bjk is 2x2.
    #
    # Strassen at block level:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 - M5
    # C01 = M2 + M4
    # C10 = M1 + M6
    # C11 = M0 - M1 + M2 + M6
    #
    # Each M_s is a product of two 2x2 matrices (of 2x2 blocks).
    # To get the 49 rank-1 terms, we decompose each M_s using the 2x2x2 Strassen factors.
    #
    # For term s, we have:
    #   X_s = the 2x2 matrix (of 2x2 blocks) that multiplies A
    #   Y_s = the 2x2 matrix (of 2x2 blocks) that multiplies B
    #   M_s = X_s * Y_s
    #
    # The rank-1 decomposition of M_s as a 4x4x4 tensor uses the 7 inner Strassen terms.
    # The 49 terms are:
    #   For each s in 0..6 and t in 0..6:
    #     U[4x4 index, (s,t)] = X_s[row, col] * U2[row*2+col, t]
    #     V[4x4 index, (s,t)] = Y_s[row, col] * V2[row*2+col, t]
    #     W[4x4 index, (s,t)] = W2[row*2+col, t] * coeff
    #
    # Wait, this is getting complicated. Let me use a different approach.
    #
    # The 4x4x4 multiplication tensor can be directly constructed as the Kronecker product
    # of the 2x2x2 tensor with itself? No, that would give rank 49, but the structure
    # is not exactly Kronecker.
    #
    # Actually, the recursive Strassen for 4x4x4 gives rank 49. Let me construct it explicitly.
    #
    # For the 4x4x4 tensor, the indices are (i, j, k) with i,j in 0..3 for A, j,k in 0..3 for B, k,i in 0..3 for C.
    # Wait, A is 4x4, B is 4x4, C = A*B is 4x4.
    # The tensor is T[i, j, k] = 1 if C[i,k] = sum_j A[i,j] B[j,k].
    #
    # The 49 rank-1 terms come from the recursive Strassen. Let me just hardcode a known 49-rank decomposition.
    #
    # Alternatively, I can use the fact that the 4x4x4 tensor is the "direct sum" of 7 copies of the 2x2x2 tensor,
    # each embedded appropriately.
    #
    # Let me use a simpler approach: use the schoolbook for 4x4x4 (rank 64) and then add the 5th column of B
    # using schoolbook (rank 16), total rank 80. This is what the current program does.
    #
    # To improve, I need a better decomposition. The known rank for 4x4x4 is 49 (Strassen).
    # For 4x4x5, the rank is at most 49 + 16 = 65, but it might be lower.
    #
    # Let me try to construct the 49-rank decomposition for 4x4x4 explicitly.
    #
    # The 49 terms are indexed by (s, t) where s, t in 0..6.
    # For each (s, t), we have a rank-1 term u_s,t ⊗ v_s,t ⊗ w_s,t.
    #
    # The outer Strassen term s determines which blocks of A and B are combined.
    # The inner Strassen term t determines how the 2x2 blocks are multiplied.
    #
    # Let me define the 7 outer terms:
    # s=0: X = [[1,0],[0,1]], Y = [[1,0],[0,1]]
    # s=1: X = [[0,0],[1,1]], Y = [[1,0],[0,0]]
    # s=2: X = [[1,0],[0,0]], Y = [[0,1],[0,-1]]
    # s=3: X = [[0,0],[0,1]], Y = [[0,0],[1,-1]]
    # s=4: X = [[1,1],[0,0]], Y = [[0,0],[0,1]]
    # s=5: X = [[-1,0],[1,0]], Y = [[1,1],[0,0]]
    # s=6: X = [[0,1],[-1,0]], Y = [[0,0],[1,1]]
    #
    # For each s, X and Y are 2x2 matrices of 2x2 blocks.
    # The product X_s * Y_s is a 2x2 matrix of 2x2 blocks.
    # We decompose this using the 7 inner Strassen terms.
    #
    # For inner term t, the 2x2x2 Strassen factors U2, V2, W2 are used.
    # The global U, V, W for term (s, t) are:
    #   U[global_idx, (s,t)] = X_s[bi, bj] * U2[bi*2+bj, t]  (if X_s[bi,bj] is nonzero)
    #   V[global_idx, (s,t)] = Y_s[bj, bk] * V2[bj*2+bk, t]
    #   W[global_idx, (s,t)] = W2[bk*2+bi, t] * (coefficient from C reconstruction)
    #
    # This is still complex. Let me just use the known fact that the rank of 4x4x4 is 49
    # and construct a valid 49-rank decomposition.
    #
    # Given the time constraints, I will use a simpler approach:
    # 1. Use Strassen for 4x4x4 (rank 49).
    # 2. Use schoolbook for the 5th column of B (rank 16).
    # 3. Total rank = 65.
    #
    # But I need to construct the 49-rank decomposition for 4x4x4.
    #
    # Let me use a different strategy: use the 2x2x2 Strassen decomposition
    # and apply it to 2x2 blocks of the 4x4 matrices.
    #
    # The 4x4x4 tensor can be viewed as a 2x2x2 tensor where each entry is a 2x2x2 tensor.
    # So the rank is 7 * 7 = 49.
    #
    # The 49 rank-1 terms are:
    #   For s, t in 0..6:
    #     u = U2_outer[s] ⊗ U2_inner[t]
    #     v = V2_outer[s] ⊗ V2_inner[t]
    #     w = W2_outer[s] ⊗ W2_inner[t]
    #
    # Where U2_outer, V2_outer, W2_outer are the 2x2x2 Strassen factors for the block level.
    #
    # Let me define the outer Strassen factors (for 2x2 blocks of 2x2 matrices):
    # The outer 2x2 matrices have entries that are 2x2 matrices.
    # The Strassen factors for the outer level are the same as the inner level,
    # but applied to the block indices.
    #
    # So the 49 terms are:
    #   For s, t in 0..6:
    #     U[row, col, (s,t)] = U2[row_block, col_block, s] * U2[row_inner, col_inner, t]
    #     V[row, col, (s,t)] = V2[row_block, col_block, s] * V2[row_inner, col_inner, t]
    #     W[row, col, (s,t)] = W2[row_block, col_block, s] * W2[row_inner, col_inner, t]
    #
    # Where row_block, col_block are the block indices (0 or 1), and
    # row_inner, col_inner are the inner indices (0 or 1).
    #
    # The global index for U is (row_block*2 + row_inner) * 4 + (col_block*2 + col_inner).
    #
    # Let me implement this.

    # The 49-rank decomposition for 4x4x4:
    U49 = np.zeros((16, 49))
    V49 = np.zeros((16, 49))
    W49 = np.zeros((16, 49))

    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for rb in range(2):
                for cb in range(2):
                    for ri in range(2):
                        for ci in range(2):
                            u_val = U2[rb * 2 + ri, s] * U2[rb * 2 + ri, t]
                            if u_val != 0:
                                U49[(rb * 2 + ri) * 4 + (cb * 2 + ci), r] = u_val
            for rb in range(2):
                for cb in range(2):
                    for ri in range(2):
                        for ci in range(2):
                            v_val = V2[rb * 2 + ri, s] * V2[cb * 2 + ci, t]
                            if v_val != 0:
                                V49[(rb * 2 + ri) * 4 + (cb * 2 + ci), r] = v_val
            for rb in range(2):
                for cb in range(2):
                    for ri in range(2):
                        for ci in range(2):
                            w_val = W2[rb * 2 + ri, s] * W2[cb * 2 + ci, t]
                            if w_val != 0:
                                W49[(rb * 2 + ri) * 4 + (cb * 2 + ci), r] = w_val

    # Now, for 4x4x5, we add the 5th column of B.
    # The 5th column of B is a 4x1 matrix.
    # We use schoolbook for the 4x4x1 multiplication: rank 16.
    #
    # The total rank is 49 + 16 = 65.

    total_rank = 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))

    # Copy the 49-rank decomposition for 4x4x4.
    U[:, :49] = U49
    V[:16, :49] = V49
    W[:16, :49] = W49

    # Add the schoolbook terms for the 5th column of B.
    for i in range(4):
        for j in range(4):
            col_idx = 49 + i * 4 + j
            U[i * 4 + j, col_idx] = 1
            V[j * 5 + 4, col_idx] = 1
            W[4 * 4 + i, col_idx] = 1

    # Convert to lists for the return value.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
