"""Matrix multiplication 2,4,6 with block-Strassen decomposition (rank 42)."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Strassen 2x2 by 2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h+g)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[p, q], [r, s]]
    # p = M1 + M4 - M5 + M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M2 + M3 + M6

    # U vectors for A (a,b,c,d)
    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]

    # V vectors for B (e,f,g,h)
    V_strassen = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 1, 1, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 1, 1, 1]   # M7
    ]

    # W vectors for C (p,q,r,s)
    # p = M1 + M4 - M5 + M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M2 + M3 + M6
    W_strassen = []
    for k in range(7):
        w = [0, 0, 0, 0]
        if k == 0:   # M1: p=1, s=1
            w[0] += 1
            w[3] += 1
        elif k == 1: # M2: r=1, s=-1
            w[2] += 1
            w[3] -= 1
        elif k == 2: # M3: q=1, s=1
            w[1] += 1
            w[3] += 1
        elif k == 3: # M4: p=1, r=1
            w[0] += 1
            w[2] += 1
        elif k == 4: # M5: p=-1, q=1
            w[0] -= 1
            w[1] += 1
        elif k == 5: # M6: s=1
            w[3] += 1
        elif k == 6: # M7: p=1
            w[0] += 1
        W_strassen.append(w)

    # Build 2x4 by 4x6 decomposition
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x6: rows j=0,1,2,3; cols k=0,1,2,3,4,5
    # C is 2x6: rows i=0,1; cols k=0,1,2,3,4,5

    # We partition A into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
    # We partition B into three 2x2 blocks: B_left (rows 0,1), B_mid (rows 2,3)
    # Wait, B is 4x6. We need to partition its rows into groups of 2 and cols into groups of 2.
    # Rows: (0,1), (2,3) -> 2 groups
    # Cols: (0,1), (2,3), (4,5) -> 3 groups
    # So we have 2 * 3 = 6 blocks of 2x2 multiplication.
    # Each block uses 7 Strassen terms. Total rank = 6 * 7 = 42.

    U = [[0] * 42 for _ in range(n * m)]  # 8 x 42
    V = [[0] * 42 for _ in range(m * p)]  # 24 x 42
    W = [[0] * 42 for _ in range(p * n)]  # 12 x 42

    block_idx = 0
    for i_block in range(2):  # rows of A: (0,1)
        for j_block in range(2):  # rows of B / cols of A: (0,1) and (2,3)
            for k_block in range(3):  # cols of B: (0,1), (2,3), (4,5)
                # For this block, we use Strassen decomposition
                # A_block is 2x2: A[i_block*2: i_block*2+2, j_block*2: j_block*2+2]
                # B_block is 2x2: B[j_block*2: j_block*2+2, k_block*2: k_block*2+2]
                # C_block is 2x2: C[i_block*2: i_block*2+2, k_block*2: k_block*2+2]

                for s in range(7):  # 7 Strassen terms
                    col = block_idx * 7 + s

                    # U entries: A has indices (i, j) where i in {i_block*2, i_block*2+1}, j in {j_block*2, j_block*2+1}
                    # U is indexed by (i*m + j) = (i*4 + j)
                    for di in range(2):
                        for dj in range(2):
                            i = i_block * 2 + di
                            j = j_block * 2 + dj
                            u_val = U_strassen[s][di * 2 + dj]  # Wait, U_strassen is for 2x2 A
                            # U_strassen[s] = [a, b, c, d] where A = [[a, b], [c, d]]
                            # So index di*2+dj maps to (di, dj)
                            # Actually U_strassen[s] is a flat vector of 4 elements: [A00, A01, A10, A11]
                            # So U_strassen[s][di*2 + dj] is correct
                            U[i * m + j][col] += u_val

                    # V entries: B has indices (j, k) where j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
                    # V is indexed by (j*p + k) = (j*6 + k)
                    for dj in range(2):
                        for dk in range(2):
                            j = j_block * 2 + dj
                            k = k_block * 2 + dk
                            v_val = V_strassen[s][dj * 2 + dk]  # V_strassen[s] = [B00, B01, B10, B11]
                            V[j * p + k][col] += v_val

                    # W entries: C has indices (i, k) where i in {i_block*2, i_block*2+1}, k in {k_block*2, k_block*2+1}
                    # W is indexed by (k*n + i) = (k*2 + i) per the trace convention W[k*n+i, r]
                    for di in range(2):
                        for dk in range(2):
                            i = i_block * 2 + di
                            k = k_block * 2 + dk
                            w_val = W_strassen[s][di * 2 + dk]  # W_strassen[s] = [C00, C01, C10, C11]
                            W[k * n + i][col] += w_val

                block_idx += 1

    # Verify the decomposition
    def _verify(U, V, W, n, m, p):
        rank = len(U[0]) if len(U) > 0 else 0
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        T = np.einsum('ijc,abc,dec->ijabde', U_reshaped, V_reshaped, W_reshaped)
        idx_i = np.arange(n)[:, None, None, None, None, None]
        idx_j = np.arange(m)[None, :, None, None, None, None]
        idx_a = np.arange(m)[None, None, :, None, None, None]
        idx_b = np.arange(p)[None, None, None, :, None, None]
        idx_d = np.arange(p)[None, None, None, None, :, None]
        idx_e = np.arange(n)[None, None, None, None, None, :]
        target = (idx_i == idx_e) & (idx_j == idx_a) & (idx_b == idx_d)
        target = target.astype(float)
        return np.allclose(T, target, atol=1e-5)

    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
