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
    # Correct patterns for C = A*B where A,B,C are 3x3
    # Using the standard Laderman decomposition
    laderman_23_patterns = [
        # M1 = A00 * (B00+B10+B20) -> C00 += M1
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        # M2 = (A10+A20) * B10 -> C10 += M2
        ({(1,0):1,(2,0):1}, {(1,0):1}, {(1,0):1}),
        # M3 = A20 * (B20-B10) -> C20 += M3
        ({(2,0):1}, {(2,0):1,(1,0):-1}, {(2,0):1}),
        # M4 = (A01+A11) * (B01-B00) -> C01 += M4, C10 -= M4
        ({(0,1):1,(1,1):1}, {(0,1):1,(0,0):-1}, {(0,1):1,(1,0):-1}),
        # M5 = A11 * (B11+B21) -> C11 += M5
        ({(1,1):1}, {(1,1):1,(2,1):1}, {(1,1):1}),
        # M6 = (A21+A11) * (B21-B11) -> C21 += M6, C11 -= M6
        ({(2,1):1,(1,1):1}, {(2,1):1,(1,1):-1}, {(2,1):1,(1,1):-1}),
        # M7 = A21 * (B20-B10) -> C20 += M7
        ({(2,1):1}, {(2,0):1,(1,0):-1}, {(2,0):1}),
        # M8 = (A02+A12) * (B02-B00) -> C02 += M8, C10 -= M8
        ({(0,2):1,(1,2):1}, {(0,2):1,(0,0):-1}, {(0,2):1,(1,0):-1}),
        # M9 = A12 * (B12+B22) -> C12 += M9
        ({(1,2):1}, {(1,2):1,(2,2):1}, {(1,2):1}),
        # M10 = (A22+A12) * (B22-B12) -> C22 += M10, C12 -= M10
        ({(2,2):1,(1,2):1}, {(2,2):1,(1,2):-1}, {(2,2):1,(1,2):-1}),
        # M11 = A22 * (B20-B10) -> C20 += M11
        ({(2,2):1}, {(2,0):1,(1,0):-1}, {(2,0):1}),
        # M12 = (A01+A21) * (B01+B11+B21) -> C01 += M12, C21 += M12
        ({(0,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(0,1):1,(2,1):1}),
        # M13 = (A02+A22) * (B02+B12+B22) -> C02 += M13, C22 += M13
        ({(0,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(0,2):1,(2,2):1}),
        # M14 = (A10+A20) * (B11+B21) -> C11 += M14, C21 += M14
        ({(1,0):1,(2,0):1}, {(1,1):1,(2,1):1}, {(1,1):1,(2,1):1}),
        # M15 = (A10+A20) * (B12+B22) -> C12 += M15, C22 += M15
        ({(1,0):1,(2,0):1}, {(1,2):1,(2,2):1}, {(1,2):1,(2,2):1}),
        # M16 = (A00+A11+A22) * (B00+B11+B22) -> C00 += M16, C11 += M16, C22 += M16
        ({(0,0):1,(1,1):1,(2,2):1}, {(0,0):1,(1,1):1,(2,2):1}, {(0,0):1,(1,1):1,(2,2):1}),
        # M17 = (A00+A11) * (B10+B20) -> C01 += M17, C11 += M17
        ({(0,0):1,(1,1):1}, {(1,0):1,(2,0):1}, {(0,1):1,(1,1):1}),
        # M18 = (A00+A22) * (B20+B10) -> C02 += M18, C22 += M18
        ({(0,0):1,(2,2):1}, {(2,0):1,(1,0):1}, {(0,2):1,(2,2):1}),
        # M19 = (A11+A22) * (B00+B10) -> C12 += M19, C22 += M19
        ({(1,1):1,(2,2):1}, {(0,0):1,(1,0):1}, {(1,2):1,(2,2):1}),
        # M20 = (A01+A12) * (B10+B20) -> C01 += M20, C12 += M20
        ({(0,1):1,(1,2):1}, {(1,0):1,(2,0):1}, {(0,1):1,(1,2):1}),
        # M21 = (A02+A21) * (B00+B10) -> C02 += M21, C21 += M21
        ({(0,2):1,(2,1):1}, {(0,0):1,(1,0):1}, {(0,2):1,(2,1):1}),
        # M22 = (A10+A21) * (B01+B11+B21) -> C11 += M22, C21 += M22
        ({(1,0):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1,(2,1):1}),
        # M23 = (A12+A20) * (B02+B12+B22) -> C12 += M23, C22 += M23
        ({(1,2):1,(2,0):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1,(2,2):1}),
    ]

    # Strassen 2x2 patterns
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def verify_3x3_patterns(patterns):
        """Verify that the 3x3 patterns correctly compute C = A*B."""
        np.random.seed(123)
        for trial in range(10):
            A_t = np.random.randn(3, 3)
            B_t = np.random.randn(3, 3)
            C_expected = A_t @ B_t
            C_computed = np.zeros((3, 3))
            for u_c, v_c, w_c in patterns:
                mu = sum(val * A_t[r, c] for (r, c), val in u_c.items())
                mv = sum(val * B_t[r, c] for (r, c), val in v_c.items())
                for (r, c), val in w_c.items():
                    C_computed[r, c] += val * mu * mv
            if np.max(np.abs(C_computed - C_expected)) > 1e-10:
                return False
        return True

    def verify_2x2_patterns(patterns):
        """Verify that the 2x2 patterns correctly compute C = A*B."""
        np.random.seed(456)
        for trial in range(5):
            A_t = np.random.randn(2, 2)
            B_t = np.random.randn(2, 2)
            C_expected = A_t @ B_t
            C_computed = np.zeros((2, 2))
            for u_c, v_c, w_c in patterns:
                mu = sum(val * A_t[r, c] for (r, c), val in u_c.items())
                mv = sum(val * B_t[r, c] for (r, c), val in v_c.items())
                for (r, c), val in w_c.items():
                    C_computed[r, c] += val * mu * mv
            if np.max(np.abs(C_computed - C_expected)) > 1e-10:
                return False
        return True

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        """Verify a decomposition is correct."""
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        for r in range(rank_used):
            M_u = np.dot(U_arr[:, r], A_test.flatten())
            M_v = np.dot(V_arr[:, r], B_test.flatten())
            M_r = M_u * M_v
            C_computed += M_r * W_arr[:, r].reshape(p, n).T
        error = np.max(np.abs(C_computed - C_expected))
        return error

    def build_laderman_6x6():
        """Build 6x6x6 decomposition using Laderman's 3x3 algorithm on 2x2 blocks."""
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23  # 8 * 23 = 184
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

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
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

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

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # Verify base patterns
    laderman_valid = verify_3x3_patterns(laderman_23_patterns)
    strassen_valid = verify_2x2_patterns(strassen_patterns)
    
    # If Laderman fails, try a corrected version
    if not laderman_valid:
        # Alternative Laderman patterns (different indexing)
        laderman_alt_patterns = [
            ({(0,0): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,0): 1}),
            ({(1,0): 1}, {(1,0): 1, (2,0): 1}, {(1,0): 1}),
            ({(2,0): 1}, {(2,0): 1}, {(2,0): 1}),
            ({(1,1): 1, (2,1): 1}, {(0,1): 1, (1,1): 1, (2,1): 1}, {(1,0): 1, (2,0): -1}),
            ({(1,1): 1}, {(2,1): 1}, {(1,0): -1}),
            ({(2,1): 1}, {(2,1): 1}, {(2,0): 1}),
            ({(1,2): 1, (2,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}, {(1,0): -1, (2,0): 1}),
            ({(1,2): 1}, {(2,2): 1}, {(1,0): 1}),
            ({(2,2): 1}, {(2,2): 1}, {(2,0): -1}),
            ({(0,1): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,1): 1}),
            ({(0,2): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,2): 1}),
            ({(1,1): 1}, {(0,1): 1, (1,1): 1, (2,1): 1}, {(1,1): 1}),
            ({(1,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}, {(1,2): 1}),
            ({(2,1): 1}, {(0,1): 1, (1,1): 1, (2,1): 1}, {(2,1): 1}),
            ({(2,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}, {(2,2): 1}),
            ({(0,0): 1}, {(1,1): 1, (2,1): 1}, {(0,1): 1}),
            ({(0,0): 1}, {(1,2): 1, (2,2): 1}, {(0,2): 1}),
            ({(1,1): 1}, {(2,2): 1}, {(1,2): -1}),
            ({(1,2): 1}, {(2,1): 1}, {(1,1): -1}),
            ({(2,1): 1}, {(2,2): 1}, {(2,2): -1}),
            ({(2,2): 1}, {(2,1): 1}, {(2,1): -1}),
            ({(1,1): 1, (2,2): 1}, {(0,1): 1, (1,2): 1}, {(1,1): 1, (2,2): 1}),
            ({(1,2): 1, (2,1): 1}, {(0,2): 1, (1,1): 1}, {(1,2): 1, (2,1): 1}),
        ]
        laderman_valid = verify_3x3_patterns(laderman_alt_patterns)
        if laderman_valid:
            laderman_23_patterns = laderman_alt_patterns

    # Try known constructions and pick the lowest-rank valid one
    best_solution = None
    best_rank = float('inf')

    # 1) Laderman-based 6x6 (rank 184) - best known
    if laderman_valid and time.time() < deadline - 20:
        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 2) Strassen-based 6x6 (rank 189)
    if strassen_valid and (best_solution is None or time.time() < deadline - 20):
        strassen_sol, strassen_rank = build_strassen_6x6()
        U_s = np.array(strassen_sol["U"], dtype=float)
        V_s = np.array(strassen_sol["V"], dtype=float)
        W_s = np.array(strassen_sol["W"], dtype=float)
        err_s = verify_decomposition(U_s, V_s, W_s, A_test, B_test, C_expected)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # 3) Fallback: schoolbook (rank 216)
    if best_solution is None:
        rank_trivial = n * m * p  # 216
        U_t = np.zeros((n * m, rank_trivial))
        V_t = np.zeros((m * p, rank_trivial))
        W_t = np.zeros((p * n, rank_trivial))
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

    # Final verification of the best solution
    if best_solution is not None:
        U_final = np.array(best_solution["U"], dtype=float)
        V_final = np.array(best_solution["V"], dtype=float)
        W_final = np.array(best_solution["W"], dtype=float)
        err_final = verify_decomposition(U_final, V_final, W_final, A_test, B_test, C_expected)
        if err_final > 1e-10:
            # Fallback to schoolbook if something went wrong
            rank_trivial = n * m * p
            U_t = np.zeros((n * m, rank_trivial))
            V_t = np.zeros((m * p, rank_trivial))
            W_t = np.zeros((p * n, rank_trivial))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_t[i * m + j, idx] = 1
                        V_t[j * p + k, idx] = 1
                        W_t[k * n + i, idx] = 1
                        idx += 1
            best_solution = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
    
    return best_solution
# EVOLVE-BLOCK-END
