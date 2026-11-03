import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    n, m, p = 6, 6, 6
    # Target tensor T[u, v, w] where u=i*m+k, v=k*p+j, w=j*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    # Block-Strassen with block size 2: 3x3 grid of 2x2 blocks
    # Each 2x2*2x2 product uses Strassen (rank 7)
    # Total: 27 * 7 = 189
    
    bs = 2
    nb = n // bs  # 3 blocks
    
    # Strassen 2x2 definitions (7 multiplications)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    strassen_defs = [
        {'A': {0: 1.0, 3: 1.0}, 'B': {0: 1.0, 3: 1.0}},
        {'A': {2: 1.0, 3: 1.0}, 'B': {0: 1.0}},
        {'A': {0: 1.0}, 'B': {1: 1.0, 3: -1.0}},
        {'A': {3: 1.0}, 'B': {2: 1.0, 0: -1.0}},
        {'A': {0: 1.0, 1: 1.0}, 'B': {3: 1.0}},
        {'A': {2: 1.0, 0: -1.0}, 'B': {0: 1.0, 1: 1.0}},
        {'A': {1: 1.0, 3: -1.0}, 'B': {2: 1.0, 3: 1.0}},
    ]
    
    # Reconstruction: C_block = sum_s gamma[s] * M_s
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    recon = [
        {0: 1.0, 3: 1.0, 4: -1.0, 6: 1.0},
        {2: 1.0, 4: 1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 1: -1.0, 2: 1.0, 5: 1.0},
    ]
    
    rank = nb * nb * nb * 7  # 27 * 7 = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                if time.time() > deadline - 5.0:
                    # Not enough time to finish safely; fall through to verification/fallback
                    # But since this is a deterministic fixed-size construction, it should never
                    # actually hit the deadline. This is just a safety guard.
                    pass
                for s in range(7):
                    s_def = strassen_defs[s]
                    # U: A entries
                    for blk, coeff in s_def['A'].items():
                        ar = blk // 2
                        ac = blk % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U[i_glob * m + k_glob, r_idx] += coeff

                    # V: B entries
                    for blk, coeff in s_def['B'].items():
                        br = blk // 2
                        bc = blk % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V[k_glob * p + j_glob, r_idx] += coeff

                    # W: C reconstruction
                    for c_blk in range(4):
                        gamma = recon[c_blk].get(s, 0.0)
                        if gamma != 0.0:
                            cr = c_blk // 2
                            cc = c_blk % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W[j_glob * n + i_glob, r_idx] += gamma

                    r_idx += 1

    # Verify
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.allclose(T_recon, T, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: trivial rank-216 decomposition
    rank_t = n * m * p
    U_t = np.zeros((n * m, rank_t), dtype=float)
    V_t = np.zeros((m * p, rank_t), dtype=float)
    W_t = np.zeros((p * n, rank_t), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_t[i * m + k, idx] = 1.0
                V_t[k * p + j, idx] = 1.0
                W_t[j * n + i, idx] = 1.0
                idx += 1
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
