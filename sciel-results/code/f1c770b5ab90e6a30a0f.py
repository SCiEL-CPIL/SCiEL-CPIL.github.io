"""Rewritten program: rank-minimal decomposition for 2x4 * 4x5 matrix multiplication.

Strategy:
- Split the inner dimension m=4 into two blocks of size 2.
- For each block, split the output dimension p=5 into:
  * k in {0,1}: use Strassen's 2x2x2 decomposition (rank 7)
  * k in {2,3,4}: use the 2x2x3 decomposition (rank 12)
- Total rank = 2 * (7 + 12) = 38.

This is a structural improvement over the schoolbook rank of 40.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    # Schoolbook fallback (rank 40)
    def schoolbook():
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def verify(U, V, W):
        """Check that U*V*W reconstructs the matmul tensor exactly."""
        r = U.shape[1]
        # Check all valid (i,j,k) positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = float(np.dot(U[idx_u, :], V[idx_v, :]))
                    val = val * W[idx_w, :]  # this is wrong, need elementwise
                    # Correct: val = sum_c U[idx_u,c]*V[idx_v,c]*W[idx_w,c]
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check zero positions
        for idx_u in range(n * m):
            i, j = divmod(idx_u, m)
            for idx_v in range(m * p):
                j2, k = divmod(idx_v, p)
                if j2 != j:
                    continue
                for idx_w in range(p * n):
                    k2, i2 = divmod(idx_w, n)
                    if k2 != k or i2 != i:
                        val = 0.0
                        for c in range(r):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        if abs(val) > 1e-9:
                            return False
        return True

    # Strassen 2x2x2 decomposition
    def strassen_222(j_offset, k_offset):
        """Add 7 rank-1 terms for a 2x2x2 sub-block.
        Local i in {0,1}, local j in {0,1}, local k in {0,1}.
        Global U index: i*4 + (j_offset + j_local)
        Global V index: (j_offset + j_local)*5 + (k_offset + k_local)
        Global W index: (k_offset + k_local)*2 + i
        """
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append((
            [0, 0], [1, 1],       # U: A00=1, A11=1
            [0, 0], [1, 1],       # V: B00=1, B11=1
            [(0, 0), (1, 1)]      # W: C00=1, C11=1
        ))
        # M2 = (A10+A11)B00
        terms.append((
            [0, 1], [1, 1],       # U: A10=1, A11=1
            [0, 0],               # V: B00=1
            [(1, 0), (1, 1)]      # W: C10=1, C11=-1
        ))
        # M3 = A00(B01-B11)
        terms.append((
            [0, 0],               # U: A00=1
            [0, 1], [1, -1],      # V: B01=1, B11=-1
            [(1, 0), (1, 1)]      # W: C01=1, C11=1
        ))
        # M4 = A11(B10-B00)
        terms.append((
            [1, 1],               # U: A11=1
            [1, 0], [0, -1],      # V: B10=1, B00=-1
            [(0, 0), (0, 1)]      # W: C00=1, C10=1
        ))
        # M5 = (A00+A01)B11
        terms.append((
            [0, 0], [0, 1],       # U: A00=1, A01=1
            [1, 1],               # V: B11=1
            [(0, 0), (1, 0)]      # W: C00=-1, C01=1
        ))
        # M6 = (A10-A00)(B00+B01)
        terms.append((
            [1, 0], [0, -1],      # U: A10=1, A00=-1
            [0, 0], [0, 1],       # V: B00=1, B01=1
            [(1, 1)]              # W: C11=1
        ))
        # M7 = (A01-A11)(B10+B11)
        terms.append((
            [0, 1], [1, -1],      # U: A01=1, A11=-1
            [1, 0], [1, 1],       # V: B10=1, B11=1
            [(0, 0)]              # W: C00=1
        ))

        # W coefficients
        w_coeffs = [
            [(0, 0, 1.0), (1, 1, 1.0)],       # M1
            [(1, 0, 1.0), (1, 1, -1.0)],      # M2
            [(1, 0, 1.0), (1, 1, 1.0)],       # M3
            [(0, 0, 1.0), (0, 1, 1.0)],       # M4
            [(0, 0, -1.0), (1, 0, 1.0)],      # M5
            [(1, 1, 1.0)],                     # M6
            [(0, 0, 1.0)],                     # M7
        ]

        U_mat = np.zeros((n * m, 7))
        V_mat = np.zeros((m * p, 7))
        W_mat = np.zeros((p * n, 7))

        for t in range(7):
            # U
            for entry in terms[t][0:2:2]:  # this is getting complex
                pass

        # Let me rebuild this more carefully
        U_mat = np.zeros((n * m, 7))
        V_mat = np.zeros((m * p, 7))
        W_mat = np.zeros((p * n, 7))

        # Term 0: M1
        U_mat[0*4+(0+j_offset), 0] = 1.0
        U_mat[1*4+(1+j_offset), 0] = 1.0
        V_mat[(0+j_offset)*5+(0+k_offset), 0] = 1.0
        V_mat[(1+j_offset)*5+(1+k_offset), 0] = 1.0
        W_mat[(0+k_offset)*2+0, 0] = 1.0
        W_mat[(1+k_offset)*2+1, 0] = 1.0

        # Term 1: M2
        U_mat[1*4+(0+j_offset), 1] = 1.0
        U_mat[1*4+(1+j_offset), 1] = 1.0
        V_mat[(0+j_offset)*5+(0+k_offset), 1] = 1.0
        W_mat[(0+k_offset)*2+1, 1] = 1.0
        W_mat[(1+k_offset)*2+1, 1] = -1.0

        # Term 2: M3
        U_mat[0*4+(0+j_offset), 2] = 1.0
        V_mat[(0+j_offset)*5+(1+k_offset), 2] = 1.0
        V_mat[(1+j_offset)*5+(1+k_offset), 2] = -1.0
        W_mat[(1+k_offset)*2+0, 2] = 1.0
        W_mat[(1+k_offset)*2+1, 2] = 1.0

        # Term 3: M4
        U_mat[1*4+(1+j_offset), 3] = 1.0
        V_mat[(1+j_offset)*5+(0+k_offset), 3] = 1.0
        V_mat[(0+j_offset)*5+(0+k_offset), 3] = -1.0
        W_mat[(0+k_offset)*2+0, 3] = 1.0
        W_mat[(0+k_offset)*2+1, 3] = 1.0

        # Term 4: M5
        U_mat[0*4+(0+j_offset), 4] = 1.0
        U_mat[0*4+(1+j_offset), 4] = 1.0
        V_mat[(1+j_offset)*5+(1+k_offset), 4] = 1.0
        W_mat[(0+k_offset)*2+0, 4] = -1.0
        W_mat[(1+k_offset)*2+0, 4] = 1.0

        # Term 5: M6
        U_mat[1*4+(0+j_offset), 5] = 1.0
        U_mat[0*4+(0+j_offset), 5] = -1.0
        V_mat[(0+j_offset)*5+(0+k_offset), 5] = 1.0
        V_mat[(0+j_offset)*5+(1+k_offset), 5] = 1.0
        W_mat[(1+k_offset)*2+1, 5] = 1.0

        # Term 6: M7
        U_mat[0*4+(1+j_offset), 6] = 1.0
        U_mat[1*4+(1+j_offset), 6] = -1.0
        V_mat[(1+j_offset)*5+(0+k_offset), 6] = 1.0
        V_mat[(1+j_offset)*5+(1+k_offset), 6] = 1.0
        W_mat[(0+k_offset)*2+0, 6] = 1.0

        return U_mat, V_mat, W_mat

    # 2x2x3 decomposition (rank 12)
    def mu_223(j_offset, k_offset):
        """Add 12 rank-1 terms for a 2x2x3 sub-block.
        Local i in {0,1}, local j in {0,1}, local k in {0,1,2}.
        """
        U_mat = np.zeros((n * m, 12))
        V_mat = np.zeros((m * p, 12))
        W_mat = np.zeros((p * n, 12))

        col = 0
        for lk in range(3):
            gk = lk + k_offset
            # T1: A00 * B0k * C0k
            U_mat[0*4+(0+j_offset), col] = 1.0
            V_mat[(0+j_offset)*5+gk, col] = 1.0
            W_mat[gk*2+0, col] = 1.0
            col += 1
            # T2: A01 * B1k * C0k
            U_mat[0*4+(1+j_offset), col] = 1.0
            V_mat[(1+j_offset)*5+gk, col] = 1.0
            W_mat[gk*2+0, col] = 1.0
            col += 1
            # T3: A10 * B0k * C1k
            U_mat[1*4+(0+j_offset), col] = 1.0
            V_mat[(0+j_offset)*5+gk, col] = 1.0
            W_mat[gk*2+1, col] = 1.0
            col += 1
            # T4: A11 * B1k * C1k
            U_mat[1*4+(1+j_offset), col] = 1.0
            V_mat[(1+j_offset)*5+gk, col] = 1.0
            W_mat[gk*2+1, col] = 1.0
            col += 1

        return U_mat, V_mat, W_mat

    # Build the full decomposition
    # Block 1: j in {0,1}
    U1_strassen, V1_strassen, W1_strassen = strassen_222(0, 0)
    U1_mu223, V1_mu223, W1_mu223 = mu_223(0, 2)
    U_block1 = np.hstack([U1_strassen, U1_mu223])
    V_block1 = np.hstack([V1_strassen, V1_mu223])
    W_block1 = np.hstack([W1_strassen, W1_mu223])

    # Block 2: j in {2,3}
    U2_strassen, V2_strassen, W2_strassen = strassen_222(2, 0)
    U2_mu223, V2_mu223, W2_mu223 = mu_223(2, 2)
    U_block2 = np.hstack([U2_strassen, U2_mu223])
    V_block2 = np.hstack([V2_strassen, V2_mu223])
    W_block2 = np.hstack([W2_strassen, W2_mu223])

    U_full = np.hstack([U_block1, U_block2])
    V_full = np.hstack([V_block1, V_block2])
    W_full = np.hstack([W_block1, W_block2])

    # Verify
    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback
    U_s, V_s, W_s = schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
