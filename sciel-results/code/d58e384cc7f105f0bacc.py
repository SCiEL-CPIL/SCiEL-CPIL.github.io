"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for 6x6x6 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock limit

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # Each pattern: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict (i,j) -> coeff for A[i,j]
    # B_coeffs: dict (j,k) -> coeff for B[j,k]
    # C_coeffs: dict (i,k) -> coeff for C[i,k]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    laderman_23_patterns = [
        # 1: a00*(b00+b10+b20) -> c00
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        # 2: a10*(b10+b20) -> c10
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        # 3: a20*(b20) -> c20
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        # 4: (a11+a21)*(b01+b11+b21) -> c10, c20
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,0):1,(2,0):-1}),
        # 5: a11*(b21) -> c10
        ({(1,1):1}, {(2,1):1}, {(1,0):-1}),
        # 6: a21*(b21) -> c20
        ({(2,1):1}, {(2,1):1}, {(2,0):1}),
        # 7: (a12+a22)*(b02+b12+b22) -> c10, c20
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,0):-1,(2,0):1}),
        # 8: a12*(b22) -> c10
        ({(1,2):1}, {(2,2):1}, {(1,0):1}),
        # 9: a22*(b22) -> c20
        ({(2,2):1}, {(2,2):1}, {(2,0):-1}),
        # 10: a01*(b00+b10+b20) -> c01
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        # 11: a02*(b00+b10+b20) -> c02
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        # 12: a11*(b01+b11+b21) -> c11
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        # 13: a12*(b02+b12+b22) -> c12
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        # 14: a21*(b01+b11+b21) -> c21
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        # 15: a22*(b02+b12+b22) -> c22
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        # 16: a00*(b11+b21) -> c01
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        # 17: a00*(b12+b22) -> c02
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        # 18: a11*(b22) -> c12
        ({(1,1):1}, {(2,2):1}, {(1,2):-1}),
        # 19: a12*(b21) -> c11
        ({(1,2):1}, {(2,1):1}, {(1,1):-1}),
        # 20: a21*(b22) -> c22
        ({(2,1):1}, {(2,2):1}, {(2,2):-1}),
        # 21: a22*(b21) -> c21
        ({(2,2):1}, {(2,1):1}, {(2,1):-1}),
        # 22: (a11+a22)*(b01+b12) -> c11, c22
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):1,(2,2):1}),
        # 23: (a12+a21)*(b02+b11) -> c12, c21
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):1,(2,1):1}),
    ]

    def verify_decomposition(U_arr, V_arr, W_arr):
        """Verify a decomposition is correct using multiple test matrices."""
        rank_used = U_arr.shape[1]
        np.random.seed(12345)
        max_error = 0
        for trial in range(5):
            A_t = np.random.randn(n, m)
            B_t = np.random.randn(m, p)
            C_exp = A_t @ B_t
            C_comp = np.zeros((n, p))
            for r in range(rank_used):
                M_u = np.dot(U_arr[:, r], A_t.flatten())
                M_v = np.dot(V_arr[:, r], B_t.flatten())
                M_r = M_u * M_v
                C_comp += M_r * W_arr[:, r].reshape(p, n).T
            max_error = max(max_error, np.max(np.abs(C_comp - C_exp)))
        return max_error

    def build_strassen_laderman_6x6():
        """Build 6x6x6 decomposition using Strassen on 2x2 blocks of 3x3 matrices (rank 161)."""
        block_size = 3
        blocks_per_dim = 2
        rank = 7 * 23  # 161
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)

        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ]

        col_idx = 0
        for a_blocks, b_blocks, c_blocks in strassen_terms:
            for a_coeffs, b_coeffs, c_coeffs in laderman_23_patterns:
                for (bi, bj), val_block in a_blocks.items():
                    for (pi, pj), val_lad in a_coeffs.items():
                        r = bi * block_size + pi
                        c = bj * block_size + pj
                        U[r * m + c, col_idx] += val_block * val_lad

                for (bi, bj), val_block in b_blocks.items():
                    for (pi, pj), val_lad in b_coeffs.items():
                        r = bi * block_size + pi
                        c = bj * block_size + pj
                        V[r * p + c, col_idx] += val_block * val_lad

                for (bi, bj), val_block in c_blocks.items():
                    for (pi, pj), val_lad in c_coeffs.items():
                        i_global = bi * block_size + pi
                        k_global = bj * block_size + pj
                        W[k_global * n + i_global, col_idx] += val_block * val_lad

                col_idx += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_laderman_6x6():
        """Build 6x6x6 decomposition using Laderman's 3x3 algorithm on 2x2 blocks."""
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23  # 8 * 23 = 184
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for a_coeffs, b_coeffs, c_coeffs in laderman_23_patterns:
                        for (pi, pj), val in a_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val

                        for (pi, pj), val in b_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val

                        for (pi, pj), val in c_coeffs.items():
                            i = I * block_size + pi
                            k = K * block_size + pj
                            W[k * n + i, col_idx] += val

                        col_idx += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_strassen_6x6():
        """Build the 6x6x6 decomposition using Strassen on 2x2 blocks (rank 189)."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)

        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val

                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val

                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val

                        col_idx += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Try Strassen-Laderman-based 6x6 (rank 161) first
    best_solution = None
    best_rank = float('inf')

    sl_sol, sl_rank = build_strassen_laderman_6x6()
    U_sl = np.array(sl_sol["U"], dtype=float)
    V_sl = np.array(sl_sol["V"], dtype=float)
    W_sl = np.array(sl_sol["W"], dtype=float)
    err_sl = verify_decomposition(U_sl, V_sl, W_sl)
    if err_sl < 1e-10:
        best_solution = sl_sol
        best_rank = sl_rank

    # Fallback: Laderman-based 6x6 (rank 184)
    if best_solution is None or best_rank > 184:
        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # Fallback: Strassen-based 6x6 (rank 189)
    if best_solution is None or (time.time() < deadline - 10 and best_rank > 189):
        strassen_sol, strassen_rank = build_strassen_6x6()
        U_s = np.array(strassen_sol["U"], dtype=float)
        V_s = np.array(strassen_sol["V"], dtype=float)
        W_s = np.array(strassen_sol["W"], dtype=float)
        err_s = verify_decomposition(U_s, V_s, W_s)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # If both failed (shouldn't happen), use trivial schoolbook
    if best_solution is None:
        rank_trivial = n * m * p  # 216
        U_t = np.zeros((n * m, rank_trivial), dtype=int)
        V_t = np.zeros((m * p, rank_trivial), dtype=int)
        W_t = np.zeros((p * n, rank_trivial), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i * m + j, idx] = 1
                    V_t[j * p + k, idx] = 1
                    W_t[k * n + i, idx] = 1
                    idx += 1
        best_solution = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
        best_rank = rank_trivial

    return best_solution
# EVOLVE-BLOCK-END
