import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    def patterns_to_matrices(patterns, n, m, p):
        """Convert list of patterns to U, V, W matrices."""
        rank = len(patterns)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for r, (u_c, v_c, w_c) in enumerate(patterns):
            for (i, j), val in u_c.items():
                U[i * m + j, r] = val
            for (j, k), val in v_c.items():
                V[j * p + k, r] = val
            for (k, i), val in w_c.items():
                W[k * n + i, r] = val
        return U, V, W


    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    laderman_23_patterns = [
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1,(2,1):1}),
        ({(1,1):1}, {(2,1):1}, {(1,1):-1}),
        ({(2,1):1}, {(2,1):1}, {(2,1):-1}),
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1,(2,2):1}),
        ({(1,2):1}, {(2,2):1}, {(1,2):-1}),
        ({(2,2):1}, {(2,2):1}, {(2,2):-1}),
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        ({(1,1):1}, {(2,2):1}, {(1,2):1}),
        ({(1,2):1}, {(2,1):1}, {(1,1):1}),
        ({(2,1):1}, {(2,2):1}, {(2,2):1}),
        ({(2,2):1}, {(2,1):1}, {(2,1):1}),
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):-1,(2,2):-1}),
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):-1,(2,1):-1}),
    ]
    
    # Strassen 2x2 patterns (rank 7)
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
        """Verify Laderman patterns against random 3x3 matrix products."""
        np.random.seed(123)
        for trial in range(5):
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
        for A_t, B_t in [
            (np.eye(3), np.eye(3)),
            (np.ones((3,3)), np.ones((3,3))),
            (np.arange(9).reshape(3,3).astype(float), np.eye(3)),
            (np.eye(3), np.arange(9).reshape(3,3).astype(float)),
        ]:
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

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        """Verify that U, V, W decompose the matrix multiplication tensor.
        C[i,k] = sum_j A[i,j] * B[j,k]
        U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        W index: k*n + i
        """
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        A_flat = A_test.flatten()  # index: i*m + j
        B_flat = B_test.flatten()  # index: j*p + k
        for r in range(rank_used):
            M_u = np.dot(U_arr[:, r], A_flat)
            M_v = np.dot(V_arr[:, r], B_flat)
            M_r = M_u * M_v
            C_computed += M_r * W_arr[:, r].reshape(p, n).T
        error = np.max(np.abs(C_computed - C_expected))
        return error

    def build_kronecker_6x6():
        """Build 6x6x6 decomposition using Kronecker product of Strassen (2x2) and Laderman (3x3). Rank = 7*23 = 161."""
        n1, n2 = 2, 3
        m1, m2 = 2, 3
        p1, p2 = 2, 3
        
        U1, V1, W1 = patterns_to_matrices(strassen_patterns, n1, m1, p1)
        U2, V2, W2 = patterns_to_matrices(laderman_23_patterns, n2, m2, p2)
        
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R_total = R1 * R2
        
        U = np.zeros((n * m, R_total))
        V = np.zeros((m * p, R_total))
        W = np.zeros((p * n, R_total))
        
        # Precompute index mappings
        # A: i, j -> i1, i2, j1, j2 -> idx1, idx2
        idx_map_U = np.zeros((n * m, 2), dtype=int)
        for i in range(n):
            for j in range(m):
                i1 = i // n2
                i2 = i % n2
                j1 = j // m2
                j2 = j % m2
                idx1 = i1 * m1 + j1
                idx2 = i2 * m2 + j2
                idx_map_U[i * m + j, 0] = idx1
                idx_map_U[i * m + j, 1] = idx2
        
        # V: j, k -> j1, j2, k1, k2 -> idx1, idx2
        idx_map_V = np.zeros((m * p, 2), dtype=int)
        for j in range(m):
            for k in range(p):
                j1 = j // m2
                j2 = j % m2
                k1 = k // p2
                k2 = k % p2
                idx1 = j1 * p1 + k1
                idx2 = j2 * p2 + k2
                idx_map_V[j * p + k, 0] = idx1
                idx_map_V[j * p + k, 1] = idx2
        
        # W: k, i -> k1, k2, i1, i2 -> idx1, idx2
        idx_map_W = np.zeros((p * n, 2), dtype=int)
        for k in range(p):
            for i in range(n):
                k1 = k // p2
                k2 = k % p2
                i1 = i // n2
                i2 = i % n2
                idx1 = k1 * n1 + i1
                idx2 = k2 * n2 + i2
                idx_map_W[k * n + i, 0] = idx1
                idx_map_W[k * n + i, 1] = idx2
        
        # Construct U
        # U[row, r1*R2 + r2] = U1[idx1, r1] * U2[idx2, r2]
        for r1 in range(R1):
            for r2 in range(R2):
                col = r1 * R2 + r2
                U[:, col] = U1[idx_map_U[:, 0], r1] * U2[idx_map_U[:, 1], r2]
                
        # Construct V
        for r1 in range(R1):
            for r2 in range(R2):
                col = r1 * R2 + r2
                V[:, col] = V1[idx_map_V[:, 0], r1] * V2[idx_map_V[:, 1], r2]
                
        # Construct W
        for r1 in range(R1):
            for r2 in range(R2):
                col = r1 * R2 + r2
                W[:, col] = W1[idx_map_W[:, 0], r1] * W2[idx_map_W[:, 1], r2]
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, R_total


    def build_strassen_6x6():
        """Build 6x6x6 using Strassen on 3x3 blocks. Rank = 27*7 = 189."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
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

    def build_schoolbook_6x6():
        """Schoolbook rank 216."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    test_cases = [(A_test, B_test, C_expected)]
    np.random.seed(7)
    for _ in range(2):
        A_t = np.random.randn(n, m)
        B_t = np.random.randn(m, p)
        test_cases.append((A_t, B_t, A_t @ B_t))

    def verify_solution(sol, test_cases):
        U_arr = np.array(sol["U"], dtype=float)
        V_arr = np.array(sol["V"], dtype=float)
        W_arr = np.array(sol["W"], dtype=float)
        max_err = 0.0
        for A_t, B_t, C_exp in test_cases:
            err = verify_decomposition(U_arr, V_arr, W_arr, A_t, B_t, C_exp)
            max_err = max(max_err, err)
        return max_err

    # Verify base patterns
    laderman_valid = verify_3x3_patterns(laderman_23_patterns)

    # Try known constructions and pick the lowest-rank valid one
    best_solution = None
    best_rank = float('inf')

    # 1) Kronecker product (rank 161)
    if time.time() < deadline - 20:
        kronecker_sol, kronecker_rank = build_kronecker_6x6()
        err_k = verify_solution(kronecker_sol, test_cases)
        if err_k < 1e-10 and kronecker_rank < best_rank:
            best_solution = kronecker_sol
            best_rank = kronecker_rank


    # 2) Laderman-based 6x6 (rank 184) - Fallback
    if (best_solution is None or time.time() < deadline - 20) and laderman_valid:
        laderman_sol, laderman_rank = build_laderman_6x6()
        err_l = verify_solution(laderman_sol, test_cases)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 3) Strassen-based 6x6 (rank 189) - Fallback
    if (best_solution is None or time.time() < deadline - 20):
        strassen_sol, strassen_rank = build_strassen_6x6()
        err_s = verify_solution(strassen_sol, test_cases)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # 4) Fallback: schoolbook (rank 216)
    if best_solution is None:
        schoolbook_sol, schoolbook_rank = build_schoolbook_6x6()
        best_solution = schoolbook_sol
        best_rank = schoolbook_rank

    return best_solution
# EVOLVE-BLOCK-END
