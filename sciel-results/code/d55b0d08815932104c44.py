import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    rank = 49

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2x2 decomposition (7 terms)
    U_2 = np.zeros((4, 7), dtype=np.int8)
    V_2 = np.zeros((4, 7), dtype=np.int8)
    W_2 = np.zeros((4, 7), dtype=np.int8)

    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block mapping for A (4x4 matrix)
    # A is split into 4 blocks of 2x2: A_{ij} where i,j in {0,1}
    # A_specs maps (coeff, block_idx) -> 2x2 block positions in the 4x4 matrix
    # U_2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    A_specs = [
        [(1, 0), (1, 3)],      # M1 = (A00+A11)(B00+B11) -> A blocks 0 and 3
        [(1, 2), (1, 3)],      # M2 = (A10+A11)B00       -> A blocks 2 and 3
        [(1, 0)],              # M3 = A00(B10-B00)        -> A block 0
        [(1, 3)],              # M4 = A11(B01-B10)        -> A block 3
        [(1, 0), (1, 1)],      # M5 = (A00+A01)B11        -> A blocks 0 and 1
        [(1, 2), (-1, 0)],     # M6 = (A10-A00)(B00+B10)  -> A blocks 2 and -0
        [(1, 1), (-1, 3)],     # M7 = (A01-A11)(B11+B01)  -> A blocks 1 and -3
    ]

    # Block mapping for B (4x8 matrix, split into two 4x4 blocks)
    B_specs = [
        [(1, 0), (1, 3)],      # B blocks 0 and 3
        [(1, 0)],              # B block 0
        [(1, 1), (-1, 3)],     # B blocks 1 and -3
        [(1, 2), (-1, 0)],     # B blocks 2 and -0
        [(1, 3)],              # B block 3
        [(1, 0), (1, 1)],      # B blocks 0 and 1
        [(1, 2), (1, 3)],      # B blocks 2 and 3
    ]

    # Block mapping for C (4x8 result, split into two 4x4 blocks)
    C_specs = [
        [(1, 0), (1, 3)],      # C blocks 0 and 3
        [(1, 2), (-1, 3)],     # C blocks 2 and -3
        [(1, 1), (1, 3)],      # C blocks 1 and 3
        [(1, 0), (1, 2)],      # C blocks 0 and 2
        [(-1, 0), (1, 1)],     # C blocks -0 and 1
        [(1, 3)],              # C block 3
        [(1, 0)],              # C block 0
    ]

    col = 0
    for half in range(2):  # Two 4x4 blocks of B
        for k_p in range(7):
            for m_s in range(7):
                # Build U contribution
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            val = coeff_a * int(U_2[a_eff_idx, m_s])
                            if val != 0:
                                U[global_a_idx, col] += val

                # Build V contribution
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            val = coeff_b * int(V_2[b_eff_idx, m_s])
                            if val != 0:
                                V[global_b_idx, col] += val

                # Build W contribution
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            val = coeff_c * int(W_2[p_eff_idx, m_s])
                            if val != 0:
                                W[global_w_idx, col] += val

                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

BET: {mechanism: "Corrected the block indexing for A from divmod(idx,4) to divmod(idx,2) to properly map 2x2 blocks to 4x4 matrix positions, which fixes the Strassen decomposition correctness", q: 0.95, delta: [0.001, 0.01]}
