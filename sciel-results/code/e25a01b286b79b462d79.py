import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6

    # Build target tensor once (used only for verification)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    deadline = time.time() + 110.0

    # Primary: rank-189 via Strassen(2x2, rank 7) x schoolbook(3x3, rank 27)
    # 6x6 = 2x2 blocks of 3x3 sub-blocks
    rank = 7 * 27  # 189
    bs = 3
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen's 7 intermediate products for 2x2 block multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6

    # Each term: list of (block_row, block_col, coeff) for A, B, and C blocks
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M0 = (A00+A11)(B00+B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1 = (A10+A11)B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2 = A00(B01-B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3 = A11(B10-B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4 = (A00+A01)B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5 = (A10-A00)(B00+B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6 = (A01-A11)(B10+B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    # Reconstruction coefficients: C_block = sum_s gamma_s * M_s
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6
    recon = [
        {0: 1.0, 3: 1.0, 4: -1.0, 6: -1.0},  # C00
        {2: 1.0, 4: 1.0},                      # C01
        {1: 1.0, 3: 1.0},                      # C10
        {0: 1.0, 1: -1.0, 2: 1.0, 6: 1.0},    # C11
    ]

    r_idx = 0
    for s in range(7):
        s_def = strassen_terms[s]
        # For each (u, v, j) in the 3x3 sub-block indices:
        # u = row within sub-block (for A rows and C rows)
        # v = col within sub-block (for A cols / B rows)  
        # j = col within sub-block (for B cols and C cols)
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # U: A contributions
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        U[i_glob * m + k_glob, r_idx] += coeff_a
                    # V: B contributions
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        V[k_glob * p + j_glob, r_idx] += coeff_b
                    # W: C contributions
                    for c_block_idx in range(4):
                        gamma = recon[c_block_idx].get(s, 0.0)
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            W[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    # Verify
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Final fallback: schoolbook rank 216
        U_std = np.zeros((n * m, n * m * p), dtype=float)
        V_std = np.zeros((m * p, n * m * p), dtype=float)
        W_std = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_std[i * m + k, idx] = 1.0
                    V_std[k * p + j, idx] = 1.0
                    W_std[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Helper functions removed - primary construction is now self-contained in construct()
