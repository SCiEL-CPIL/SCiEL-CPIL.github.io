import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1

    # Trivial decomposition (rank 216) as fallback
    rank_triv = n * m * p
    U_triv = np.zeros((n * m, rank_triv), dtype=np.int64)
    V_triv = np.zeros((m * p, rank_triv), dtype=np.int64)
    W_triv = np.zeros((p * n, rank_triv), dtype=np.int64)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_triv[i * m + k, idx] = 1
                V_triv[k * p + j, idx] = 1
                W_triv[j * n + i, idx] = 1
                idx += 1

    # Block-Strassen with block size 2: 3x3 blocks, rank 189
    bs = 2
    nb = n // bs  # 3

    # Strassen definitions for 2x2 blocks
    # Each term is a list of (block_idx, coeff)
    # block_idx is 0..3 for 2x2 block
    # A blocks: 0,1,2,3 corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    # B blocks: 0,1,2,3 corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    strassen_A = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    strassen_B = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    
    # Reconstruction for C blocks
    # C00 = M1 - M4 + M6 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    recon_C = [
        [(0, 1), (3, -1), (5, 1), (6, -1)],
        [(4, 1), (6, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, -1), (1, 1), (4, -1)]
    ]

    rank_bs = 189
    U2 = np.zeros((n * m, rank_bs), dtype=np.int64)
    V2 = np.zeros((m * p, rank_bs), dtype=np.int64)
    W2 = np.zeros((p * n, rank_bs), dtype=np.int64)

    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    # Update U
                    for block_idx, coeff_a in strassen_A[s]:
                        ar = block_idx // 2
                        ac = block_idx % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U2[i_glob * m + k_glob, r_idx] += coeff_a
                    
                    # Update V
                    for block_idx, coeff_b in strassen_B[s]:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V2[k_glob * p + j_glob, r_idx] += coeff_b
                    
                    # Update W
                    for c_block_idx in range(4):
                        gamma = 0
                        for (s_idx, g) in recon_C[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0:
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W2[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    # Verify Block-Strassen
    T_recon = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    if not np.array_equal(T_recon, T):
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
