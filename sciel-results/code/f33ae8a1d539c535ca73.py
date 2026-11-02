import numpy as np
import time
import itertools

def construct():
    start = time.time()
    deadline = start + 125  # safety margin for 150s limit

    n, m, p = 3, 4, 4
    # We want to multiply A (n x m) by B (m x p) to get C (n x p)
    # Tensor T_{i,j,k} where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # i = i_r * m + i_c (row i_r of A, col i_c)
    # j = j_r * p + j_c (row j_r of B, col j_c)
    # k = k_c * n + k_r (col k_c of C, row k_r)
    # T_{i,j,k} = 1 iff i_c == j_r AND k_r == i_r AND k_c == j_c

    # Known lower bound: rank >= ceil(n*m*p / min(n,m,p)) is not tight
    # Schoolbook rank = n*m*p = 48
    # Strassen-like blocks: split A into 3x2 blocks (2 blocks), B into 2x2 blocks (2 blocks)
    # 3x2x2 multiplication rank is at most 7 (Strassen for 2x2x2 is 7, plus 4 for row 2 = 11)
    # Actually for 3x2x2: rank is exactly 11? Let's check.
    # The 3x2x2 tensor: A is 3x2, B is 2x2, C is 3x2
    # T_{i,j,k} where i in [0,6), j in [0,4), k in [0,6)
    # i = i_r*2 + i_c (i_r in {0,1,2}, i_c in {0,1})
    # j = j_r*2 + j_c (j_r in {0,1}, j_c in {0,1})
    # k = k_c*2 + k_r (k_c in {0,1}, k_r in {0,1,2})
    # T_{i,j,k} = 1 iff i_c == j_r AND k_r == i_r AND k_c == j_c
    
    # For 2x2x2, rank is 7 (Strassen)
    # For the extra row (i_r=2), we need 4 more terms (one for each (i_c, j_c, k_c) combo where k_r=2)
    # So 7+4=11 for 3x2x2

    # Total: 4 blocks of 3x2x2, each rank 11, total rank 44
    # Can we do better?
    # 3x4x4: split A into 3x2 (2 blocks), B into 2x2 (2 blocks) -> 4 blocks of 3x2x2
    # Each block needs rank at least... the 3x2x2 tensor has border rank >= 9? 
    # Actually, let's just try to find a good decomposition.

    # Let's build the 3x2x2 decomposition with rank 11
    # U_loc: 6 x 11, V_loc: 4 x 11, W_loc: 6 x 11
    # i index: i_r*2 + i_c, i_r in {0,1,2}, i_c in {0,1}
    # j index: j_r*2 + j_c, j_r in {0,1}, j_c in {0,1}
    # k index: k_c*2 + k_r, k_c in {0,1}, k_r in {0,1,2}

    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))

    # Strassen for 2x2x2 (i_r in {0,1}, j_r in {0,1}, k_r in {0,1})
    # A_00=A[0,0], A_01=A[0,1], A_10=A[1,0], A_11=A[1,1]
    # B_00=B[0,0], B_01=B[0,1], B_10=B[1,0], B_11=B[1,1]
    # C_00=C[0,0], C_01=C[0,1], C_10=C[1,0], C_11=C[1,1]

    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B10)
    # M6 = (A01-A11)*(B11+B10)
    # C00 = M0+M3-M5-M6
    # C01 = M2+M4+M6
    # C10 = M1+M5
    # C11 = M0+M2-M1+M4

    # i indices: A00=0, A01=1, A10=2, A11=3
    # j indices: B00=0, B01=1, B10=2, B11=3
    # k indices: C00=0, C01=1, C10=2, C11=3

    # M0: (A00+A11)*(B00+B11)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[2, 0] = 1  # C00 and C10? Wait, C10 = M1+M5, not M0
    # C00 = M0+M3-M5-M6, C01 = M2+M4+M6, C10 = M1+M5, C11 = M0+M2-M1+M4
    # So M0 contributes to C00 and C11
    W_loc[0, 0] = 1; W_loc[3, 0] = 1  # C00=0, C11=3

    # M1: (A10+A11)*B00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1  # C10=2, C11=3

    # M2: A00*(B01-B11)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1  # C01=1, C11=3

    # M3: A11*(B10-B00)
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1  # C00=0, C10=2

    # M4: (A00+A01)*B11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[1, 4] = 1; W_loc[3, 4] = 1  # C01=1, C11=3

    # M5: (A10-A00)*(B00+B10)
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[2, 5] = 1
    W_loc[0, 5] = -1; W_loc[2, 5] = 1  # C00=0, C10=2

    # M6: (A01-A11)*(B11+B10)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[3, 6] = 1; V_loc[2, 6] = 1
    W_loc[0, 6] = -1; W_loc[1, 6] = 1  # C00=0, C01=1

    # Extra terms for i_r=2 (row 2 of A)
    # A[2,0]=i_idx 4, A[2,1]=i_idx 5
    # C[2,0]=k_idx 4, C[2,1]=k_idx 5
    # For (i_c, j_r, j_c) where i_r=2, k_r=2:
    # T_{i,j,k} = 1 iff i_c == j_r AND k_r == i_r AND k_c == j_c
    # i_c=0, j_r=0: A[2,0]*B[0,j_c] -> C[2,j_c]
    #   j_c=0: A[2,0]*B[0,0] -> C[2,0]
    #   j_c=1: A[2,0]*B[0,1] -> C[2,1]
    # i_c=1, j_r=1: A[2,1]*B[1,j_c] -> C[2,j_c]
    #   j_c=0: A[2,1]*B[1,0] -> C[2,0]
    #   j_c=1: A[2,1]*B[1,1] -> C[2,1]

    # Term 7: A[2,0]*B[0,0] -> C[2,0]
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    # Term 8: A[2,0]*B[0,1] -> C[2,1]
    U_loc[4, 8] = 1; V_loc[1, 8] = 1; W_loc[5, 8] = 1
    # Term 9: A[2,1]*B[1,0] -> C[2,0]
    U_loc[5, 9] = 1; V_loc[2, 9] = 1; W_loc[4, 9] = 1
    # Term 10: A[2,1]*B[1,1] -> C[2,1]
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1

    # Now build full 3x4x4 decomposition
    # A: 3x4, split into two 3x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    # B: 4x4, split into two 2x2 blocks: B_top (rows 0-1), B_bottom (rows 2-3)
    # C: 3x4, split into two 3x2 blocks: C_left (cols 0-1), C_right (cols 2-3)

    # Block (a_block, b_block) where a_block in {0,1} (left/right of A), b_block in {0,1} (top/bottom of B)
    # a_block=0: A cols 0-1, a_block=1: A cols 2-3
    # b_block=0: B rows 0-1, b_block=1: B rows 2-3
    # C block: if a_block=0, C cols 0-1 (left); if a_block=1, C cols 2-3 (right)
    # But also depends on b_block: 
    # C[i, j] = sum_{k=0}^{3} A[i,k]*B[k,j]
    # If a_block=0 (A cols 0-1) and b_block=0 (B rows 0-1): C[i, j] gets A[i,0-1]*B[0-1,j]
    #   This contributes to C[i, j] for all j
    #   If j is in 0-1: C_left
    #   If j is in 2-3: C_right
    # So each (a_block, b_block) pair contributes to both C_left and C_right?
    # No, wait. The 3x2x2 multiplication computes C[i, j] for i in {0,1,2}, j in {0,1}
    # where j is the local column index. The global column depends on the block structure.

    # Let me reconsider. We have 4 blocks:
    # Block 00: A[:, 0:2] * B[0:2, :] -> contributes to C[:, :]
    # Block 01: A[:, 2:4] * B[0:2, :] -> contributes to C[:, :]
    # Block 10: A[:, 0:2] * B[2:4, :] -> contributes to C[:, :]
    # Block 11: A[:, 2:4] * B[2:4, :] -> contributes to C[:, :]

    # But each block is 3x2 * 2x4 -> 3x4
    # We can split the 2x4 B part into two 2x2 parts
    # So block 00: A[:, 0:2] * B[0:2, 0:2] -> 3x2 (C[:, 0:2])
    #              A[:, 0:2] * B[0:2, 2:4] -> 3x2 (C[:, 2:4])
    # etc.

    # Total 8 sub-blocks of 3x2x2, each rank 11, total rank 88. That's worse than schoolbook 48!

    # Better approach: keep B as 2x4 in each block
    # 3x2 * 2x4 -> 3x4
    # This is a 3x2x4 multiplication. What's its rank?
    # T_{i,j,k} where i in [0,6), j in [0,8), k in [0,12)
    # i = i_r*2 + i_c, j = j_r*4 + j_c, k = k_c*3 + k_r
    # T = 1 iff i_c == j_r AND k_r == i_r AND k_c == j_c

    # This is more complex. Let's just use the 44-rank solution from the seed.

    U = np.zeros((n * m, 44))
    V = np.zeros((m * p, 44))
    W = np.zeros((p * n, 44))

    def add_block(block_idx, a_col_start, b_row_start, c_col_start):
        col_off = block_idx * 11
        for i_r in range(3):
            for i_c in range(2):
                a_idx = i_r * m + (a_col_start + i_c)
                U[a_idx, col_off:col_off+11] += U_loc[i_r * 2 + i_c, :]
        for j_r in range(2):
            for j_c in range(2):
                b_idx = (b_row_start + j_r) * p + j_c
                V[b_idx, col_off:col_off+11] += V_loc[j_r * 2 + j_c, :]
        for k_r in range(3):
            for k_c in range(2):
                c_idx = (c_col_start + k_c) * n + k_r
                W[c_idx, col_off:col_off+11] += W_loc[k_c * 2 + k_r, :]

    # 4 blocks: (a_block, b_block) where a_block=0/1 (A left/right), b_block=0/1 (B top/bottom)
    # Each block: A[:, a_block*2:(a_block+1)*2] * B[b_block*2:(b_block+1)*2, :]
    # C[:, c_block*2:(c_block+1)*2] where c_block depends on... 
    # Actually, C[i,j] = sum_{k} A[i,k] B[k,j]
    # For fixed a_block and b_block, we compute partial C[i,j] = A[i, a_block*2 + i_c] * B[b_block*2 + j_r, j]
    # where i_c in {0,1}, j_r in {0,1}, and we need i_c == j_r
    # The result goes to C[i, j] for all j in {0,1,2,3}
    # But our 3x2x2 decomposition only handles j in {0,1} (local)
    # So we need to split j into two halves: j in {0,1} and j in {2,3}

    # This means 8 sub-blocks total, each 3x2x2, rank 11, total 88.
    # But the seed uses 44, so it must be using 4 blocks of 3x2x2 somehow.

    # Let me re-read the seed. The seed uses 4 add_block calls with:
    # add_block(0, 0, 0, 0, 0, 0, 0) -> a_row=0, a_col=0, b_row=0, b_col=0, c_row=0, c_col=0
    # add_block(1, 0, 2, 2, 0, 0, 0)
    # add_block(2, 0, 0, 0, 2, 0, 2)
    # add_block(3, 0, 2, 2, 2, 0, 2)

    # I think the seed is using a different decomposition. Let me just use the seed's approach.

    # Actually, let's just return the seed's solution as-is, since it scores 1.0909
    # and try to improve from there.

    # For now, build the 44-rank solution using the seed's block structure
    # The seed's add_block has 6 parameters: block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start
    # This suggests a more general block structure.

    # Let me just hardcode the 44-rank solution from the seed and verify it works.

    # Rebuild using the seed's logic
    U = np.zeros((12, 44))
    V = np.zeros((16, 44))
    W = np.zeros((12, 44))

    def add_block2(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_off = block_idx * 11
        for r in range(3):
            for c in range(2):
                la = r * 2 + c
                ga = (a_row_start + r) * m + (a_col_start + c)
                U[ga, col_off:col_off+11] += U_loc[la, :]
        for r in range(2):
            for c in range(2):
                lb = r * 2 + c
                gb = (b_row_start + r) * p + (b_col_start + c)
                V[gb, col_off:col_off+11] += V_loc[lb, :]
        for r in range(3):
            for c in range(2):
                lc = r * 2 + c
                gc = (c_col_start + c) * n + (c_row_start + r)
                W[gc, col_off:col_off+11] += W_loc[lc, :]

    add_block2(0, 0, 0, 0, 0, 0, 0)
    add_block2(1, 0, 2, 2, 0, 0, 0)
    add_block2(2, 0, 0, 0, 2, 0, 2)
    add_block2(3, 0, 2, 2, 2, 0, 2)

    # Verify
    T_exp = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        ar, ac = divmod(a, m)
        for b in range(m*p):
            br, bc = divmod(b, p)
            if ac == br:
                ck = bc * n + ar
                T_exp[a, b, ck] = 1.0

    T_test = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    if not np.allclose(T_test, T_exp, atol=1e-9):
        # Fallback to schoolbook
        U_s = np.zeros((n*m, n*m*p))
        V_s = np.zeros((m*p, n*m*p))
        W_s = np.zeros((p*n, n*m*p))
        idx = 0
        for a in range(n*m):
            ar, ac = divmod(a, m)
            for b in range(m*p):
                br, bc = divmod(b, p)
                if ac == br:
                    ck = bc * n + ar
                    U_s[a, idx] = 1
                    V_s[b, idx] = 1
                    W_s[ck, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Try to reduce rank
    def verify(U_mat, V_mat, W_mat):
        T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
        return np.allclose(T, T_exp, atol=1e-9)

    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    r_cur = 44
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for col in range(r_cur - 1, -1, -1):
            if time.time() > deadline:
                break
            U_t = np.delete(U_cur, col, axis=1)
            V_t = np.delete(V_cur, col, axis=1)
            W_t = np.delete(W_cur, col, axis=1)
            if verify(U_t, V_t, W_t):
                U_cur, V_cur, W_cur = U_t, V_t, W_t
                r_cur -= 1
                improved = True
                break

    return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
