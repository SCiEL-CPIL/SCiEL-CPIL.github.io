import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 115.0

    # Laderman's rank-16 algorithm for 4x4 matrix multiplication
    # Partition 4x4 into 2x2 blocks of size 2x2
    # 16 rank-1 terms (scalar multiplications in the block algebra)
    
    # Laderman's algorithm definitions:
    # M1  = A11 * (B11 - B22)
    # M2  = (A11 + A12) * B22
    # M3  = (A21 + A22) * B11
    # M4  = A22 * (B21 - B11)
    # M5  = (A11 - A21) * (B11 + B12)
    # M6  = (A12 - A22) * (B21 + B22)
    # M7  = (A11 + A22 + A12 + A21) * (B11 + B22 - B12 - B21)  -- wait, this is not standard
    # Let me use the standard Laderman 16-term algorithm:
    
    # Standard Laderman (from literature):
    # M1  = A11 * (B11 - B22)
    # M2  = (A11 + A12) * B22
    # M3  = (A21 + A22) * B11
    # M4  = A22 * (B21 - B11)
    # M5  = (A11 - A21) * (B11 + B12)
    # M6  = (A12 - A22) * (B21 + B22)
    # M7  = (A11 + A22) * (B12 + B21)
    # M8  = (A12 + A21) * (B11 + B22)
    # M9  = (A11 + A12 + A21 + A22) * (B11 + B12 + B21 + B22)
    # ... Actually, let me just use the known rank-16 Laderman coefficients.
    
    # The 16 multiplications in Laderman's algorithm:
    # Using 2x2 block notation where A = [[A11, A12], [A21, A22]], each block is 2x2:
    
    laderman_terms = [
        # Each term: (A_block_indices with coeffs, B_block_indices with coeffs, C_block_indices with coeffs)
        # A blocks: 0=A11, 1=A12, 2=A21, 3=A22
        # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
        # C blocks: 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1 = A11 * (B11 - B22)
        ([(0, 1.0)], [(0, 1.0), (3, -1.0)], [(0, 1.0)]),
        # M2 = (A11 + A12) * B22
        ([(0, 1.0), (1, 1.0)], [(3, 1.0)], [(1, 1.0)]),
        # M3 = (A21 + A22) * B11
        ([(2, 1.0), (3, 1.0)], [(0, 1.0)], [(2, 1.0)]),
        # M4 = A22 * (B21 - B11)
        ([(3, 1.0)], [(2, 1.0), (0, -1.0)], [(3, 1.0)]),
        # M5 = (A11 - A21) * (B11 + B12)
        ([(0, 1.0), (2, -1.0)], [(0, 1.0), (1, 1.0)], [(0, 1.0)]),
        # M6 = (A12 - A22) * (B21 + B22)
        ([(1, 1.0), (3, -1.0)], [(2, 1.0), (3, 1.0)], [(1, 1.0)]),
        # M7 = (A11 + A22) * (B12 + B21)
        ([(0, 1.0), (3, 1.0)], [(1, 1.0), (2, 1.0)], [(2, 1.0)]),
        # M8 = (A12 + A21) * (B11 + B22)
        ([(1, 1.0), (2, 1.0)], [(0, 1.0), (3, 1.0)], [(3, 1.0)]),
        # M9 = (A11 + A12 + A21 + A22) * (B11 + B12 + B21 + B22)
        ([(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)], [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)], [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]),
        # M10 = (A11 - A12 + A21 - A22) * (B11 - B12 + B21 - B22)
        ([(0, 1.0), (1, -1.0), (2, 1.0), (3, -1.0)], [(0, 1.0), (1, -1.0), (2, 1.0), (3, -1.0)], [(0, -1.0), (1, -1.0), (2, -1.0), (3, -1.0)]),
        # M11 = (A11 + A12 - A21 - A22) * (B11 + B12 - B21 - B22)
        ([(0, 1.0), (1, 1.0), (2, -1.0), (3, -1.0)], [(0, 1.0), (1, 1.0), (2, -1.0), (3, -1.0)], [(0, -1.0), (1, 1.0), (2, -1.0), (3, 1.0)]),
        # M12 = (A11 - A12 - A21 + A22) * (B11 - B12 - B21 + B22)
        ([(0, 1.0), (1, -1.0), (2, -1.0), (3, 1.0)], [(0, 1.0), (1, -1.0), (2, -1.0), (3, 1.0)], [(0, 1.0), (1, -1.0), (2, 1.0), (3, -1.0)]),
        # M13 = A11 * B11
        ([(0, 1.0)], [(0, 1.0)], [(0, -1.0), (1, -1.0), (2, -1.0), (3, -1.0)]),
        # M14 = A12 * B12
        ([(1, 1.0)], [(1, 1.0)], [(0, -1.0), (1, -1.0), (2, -1.0), (3, -1.0)]),
        # M15 = A21 * B21
        ([(2, 1.0)], [(2, 1.0)], [(0, -1.0), (1, -1.0), (2, -1.0), (3, -1.0)]),
        # M16 = A22 * B22
        ([(3, 1.0)], [(3, 1.0)], [(0, -1.0), (1, -1.0), (2, -1.0), (3, -1.0)]),
    ]
    
    # Actually, I'm not confident in these Laderman coefficients. Let me use a verified approach.
    # Let me use Strassen's 7-term algorithm for 2x2 blocks instead, applied at multiple levels.
    
    # Strategy: Use Strassen 2x2 of 3x3 blocks (rank 7 * 27 = 189) as baseline,
    # but try to improve by using a better 3x3 algorithm.
    
    # Actually, let me just implement the straightforward Strassen 2x2 of 3x3 with rank 189,
    # since I know that's correct. Then I'll verify it works.
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 definitions for 3x3 blocks
    strassen_defs = [
        # M1 = (A11 + A22) * (B11 + B22)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M2 = (A21 + A22) * B11
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M3 = A11 * (B12 - B22)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M4 = A22 * (B21 - B11)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M5 = (A11 + A12) * B22
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M6 = (A21 - A11) * (B11 + B12)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M7 = (A12 - A22) * (B21 + B22)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    recon_coeffs = [
        # C11 = M1 + M4 - M5 + M7
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C12 = M3 + M5
        [(2, 1.0), (4, 1.0)],
        # C21 = M2 + M4
        [(1, 1.0), (3, 1.0)],
        # C22 = M1 - M2 + M3 + M6
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    bs = 3
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row, r_idx] += gamma
                    r_idx += 1
    
    # Verify
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback to standard
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
