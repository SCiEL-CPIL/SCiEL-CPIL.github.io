import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7

    # Rank-minimal block decomposition:
    # Partition rows into [2,1], middle columns into [2,2,1], output columns into [2,2,2,1]
    # Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for smaller blocks.
    #
    # Row partition: A[0:2,:] and A[2:3,:]
    # Middle partition: A[:,0:2], A[:,2:4], A[:,4:5]
    # Output partition: C[:,0:2], C[:,2:4], C[:,4:6], C[:,6:7]
    #
    # For each (row_block, mid_block, out_block) we need a decomposition.
    #
    # Let's enumerate all combinations:
    # row_block sizes: [2, 1]
    # mid_block sizes: [2, 2, 1]
    # out_block sizes: [2, 2, 2, 1]
    #
    # Total rank calculation:
    # For (2,2,2): Strassen rank 7
    # For (2,2,1): schoolbook 2*2*1 = 4
    # For (2,1,2): schoolbook 2*1*2 = 4
    # For (2,1,1): schoolbook 2*1*1 = 2
    # For (1,2,2): schoolbook 1*2*2 = 4
    # For (1,2,1): schoolbook 1*2*1 = 2
    # For (1,1,2): schoolbook 1*1*2 = 2
    # For (1,1,1): schoolbook 1*1*1 = 1
    #
    # Let's compute total:
    # Rows 0-1 (size 2):
    #   Mids 0-1 (size 2): Outs 0-1 (size 2): 7, Outs 2-3 (size 2): 7, Outs 4-5 (size 2): 7, Outs 6 (size 1): 4
    #   Mids 2-3 (size 2): Outs 0-1 (size 2): 7, Outs 2-3 (size 2): 7, Outs 4-5 (size 2): 7, Outs 6 (size 1): 4
    #   Mid 4 (size 1): Outs 0-1 (size 2): 4, Outs 2-3 (size 2): 4, Outs 4-5 (size 2): 4, Outs 6 (size 1): 2
    #   Subtotal for rows 0-1: 7+7+7+4 + 7+7+7+4 + 4+4+4+2 = 25 + 25 + 14 = 64
    #
    # Row 2 (size 1):
    #   Mids 0-1 (size 2): Outs 0-1 (size 2): 4, Outs 2-3 (size 2): 4, Outs 4-5 (size 2): 4, Outs 6 (size 1): 2
    #   Mids 2-3 (size 2): Outs 0-1 (size 2): 4, Outs 2-3 (size 2): 4, Outs 4-5 (size 2): 4, Outs 6 (size 1): 2
    #   Mid 4 (size 1): Outs 0-1 (size 2): 2, Outs 2-3 (size 2): 2, Outs 4-5 (size 2): 2, Outs 6 (size 1): 1
    #   Subtotal for row 2: 4+4+4+2 + 4+4+4+2 + 2+2+2+1 = 14 + 14 + 7 = 35
    #
    # Total rank = 64 + 35 = 99

    rank_total = 99

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    def add_strassen(rank_offset, i0, j0, k0):
        # M222 Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        u_loc = [[0] * 7 for _ in range(4)]
        v_loc = [[0] * 7 for _ in range(4)]
        w_loc = [[0] * 7 for _ in range(4)]

        # M1 = (A11+A22)*(B11+B22)
        u_loc[0][0] = 1; u_loc[3][0] = 1
        v_loc[0][0] = 1; v_loc[3][0] = 1
        w_loc[0][0] = 1; w_loc[3][0] = 1

        # M2 = (A21+A22)*B11
        u_loc[2][1] = 1; u_loc[3][1] = 1
        v_loc[0][1] = 1
        w_loc[1][1] = 1; w_loc[3][1] = -1

        # M3 = A11*(B21-B22)
        u_loc[0][2] = 1
        v_loc[1][2] = 1; v_loc[3][2] = -1
        w_loc[2][2] = 1; w_loc[3][2] = 1

        # M4 = A22*(B11-B21)
        u_loc[3][3] = 1
        v_loc[2][3] = 1; v_loc[0][3] = -1
        w_loc[0][3] = 1; w_loc[1][3] = 1

        # M5 = (A11+A12)*B22
        u_loc[0][4] = 1; u_loc[1][4] = 1
        v_loc[3][4] = 1
        w_loc[0][4] = -1; w_loc[2][4] = 1

        # M6 = (A21-A11)*(B11+B12)
        u_loc[2][5] = 1; u_loc[0][5] = -1
        v_loc[0][5] = 1; v_loc[1][5] = 1
        w_loc[3][5] = 1

        # M7 = (A12-A22)*(B21+B22)
        u_loc[1][6] = 1; u_loc[3][6] = -1
        v_loc[2][6] = 1; v_loc[3][6] = 1
        w_loc[0][6] = 1

        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l * 2 + j_l][r]
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l * 2 + k_l][r]
        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l * 2 + i_l][r]

    def add_schoolbook(rank_offset, i0, j0, k0, ni, nj, nk):
        # Schoolbook for A[i0:i0+ni, j0:j0+nj] * B[j0:j0+nj, k0:k0+nk]
        for i_l in range(ni):
            for j_l in range(nj):
                for k_l in range(nk):
                    r = (i_l * nj + j_l) * nk + k_l
                    u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                    U[u_g_idx][rank_offset + r] += 1
                    v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                    V[v_g_idx][rank_offset + r] += 1
                    w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                    W[w_g_idx][rank_offset + r] += 1

    # ---- assemble blocks ----
    # Row block 0-1 (size 2)
    #   Mid block 0-1 (size 2)
    add_strassen(0, 0, 0, 0)        # rank 7
    add_strassen(7, 0, 0, 2)        # rank 7
    add_strassen(14, 0, 0, 4)       # rank 7
    add_schoolbook(21, 0, 0, 6, 2, 2, 1)  # rank 4
    #   Mid block 2-3 (size 2)
    add_strassen(25, 0, 2, 0)       # rank 7
    add_strassen(32, 0, 2, 2)       # rank 7
    add_strassen(39, 0, 2, 4)       # rank 7
    add_schoolbook(46, 0, 2, 6, 2, 2, 1)  # rank 4
    #   Mid block 4 (size 1)
    add_schoolbook(50, 0, 4, 0, 2, 1, 2)  # rank 4
    add_schoolbook(54, 0, 4, 2, 2, 1, 2)  # rank 4
    add_schoolbook(58, 0, 4, 4, 2, 1, 2)  # rank 4
    add_schoolbook(62, 0, 4, 6, 2, 1, 1)  # rank 2

    # Row block 2 (size 1)
    #   Mid block 0-1 (size 2)
    add_schoolbook(64, 2, 0, 0, 1, 2, 2)  # rank 4
    add_schoolbook(68, 2, 0, 2, 1, 2, 2)  # rank 4
    add_schoolbook(72, 2, 0, 4, 1, 2, 2)  # rank 4
    add_schoolbook(76, 2, 0, 6, 1, 2, 1)  # rank 2
    #   Mid block 2-3 (size 2)
    add_schoolbook(78, 2, 2, 0, 1, 2, 2)  # rank 4
    add_schoolbook(82, 2, 2, 2, 1, 2, 2)  # rank 4
    add_schoolbook(86, 2, 2, 4, 1, 2, 2)  # rank 4
    add_schoolbook(90, 2, 2, 6, 1, 2, 1)  # rank 2
    #   Mid block 4 (size 1)
    add_schoolbook(92, 2, 4, 0, 1, 1, 2)  # rank 2
    add_schoolbook(94, 2, 4, 2, 1, 1, 2)  # rank 2
    add_schoolbook(96, 2, 4, 4, 1, 1, 2)  # rank 2
    add_schoolbook(98, 2, 4, 6, 1, 1, 1)  # rank 1

    # ---- verification ----
    try:
        if time.time() - START_TIME < TIME_LIMIT:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)

            # Verify: sum_r U[i,j,r] V[j,k,r] W[k,i,r] should equal 1 for all i,j,k
            U_3d = U_np.reshape(n, m, rank_total)
            V_3d = V_np.reshape(m, p, rank_total)
            W_3d = W_np.reshape(p, n, rank_total)

            resid = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

            if not np.allclose(resid, 1.0, atol=1e-9):
                # Fallback: full schoolbook
                rank_fb = n * m * p
                U_fb = [[0] * rank_fb for _ in range(n * m)]
                V_fb = [[0] * rank_fb for _ in range(m * p)]
                W_fb = [[0] * rank_fb for _ in range(p * n)]
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j][idx] = 1
                            V_fb[j * p + k][idx] = 1
                            W_fb[k * n + i][idx] = 1
                            idx += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}

            # Drop exactly-zero rank columns
            col_norm = (np.sum(np.abs(U_np), axis=0) +
                        np.sum(np.abs(V_np), axis=0) +
                        np.sum(np.abs(W_np), axis=0))
            keep = [r for r in range(rank_total) if col_norm[r] > 1e-12]
            if len(keep) < rank_total:
                U = [[U_np[i, r] for r in keep] for i in range(n * m)]
                V = [[V_np[i, r] for r in keep] for i in range(m * p)]
                W = [[W_np[i, r] for r in keep] for i in range(p * n)]
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
