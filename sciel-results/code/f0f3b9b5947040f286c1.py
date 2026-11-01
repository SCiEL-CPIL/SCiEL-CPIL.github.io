"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    Starts with block-Strassen rank-36, then searches for linear dependencies
    among the rank terms to reduce the rank further.
    """
    import time
    import numpy as np
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 4, 5

    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)

    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset

        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]

        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))

    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset

        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]

        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))

    # Build the base rank-36 decomposition
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

    def build_matrices(terms):
        """Build U, V, W matrices from list of term dicts."""
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff

        return U, V, W

    def verify_decomposition(U, V, W, max_checks=1000):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        np.random.seed(12345)
        for _ in range(min(max_checks, 20)):
            A_test = np.random.randint(-10, 11, size=(n, m)).astype(float)
            B_test = np.random.randint(-10, 11, size=(m, p)).astype(float)
            C_expected = A_test @ B_test

            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_col = U_arr[:, r]
                v_col = V_arr[:, r]
                w_col = W_arr[:, r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C_computed[i, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]

            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    def check_linear_dependence(terms, indices_to_check):
        """
        Check if removing the terms at indices_to_check still gives a valid decomposition.
        We do this by checking if the remaining terms can still produce the correct output.
        """
        remaining = [terms[i] for i in range(len(terms)) if i not in set(indices_to_check)]
        U, V, W = build_matrices(remaining)
        return verify_decomposition(U, V, W, max_checks=50)

    # Start with the base rank-36
    current_terms = list(rank_terms)
    current_rank = len(current_terms)

    # Try to remove one term at a time (greedy approach)
    removed_any = True
    while removed_any and time.time() < deadline - 10:
        removed_any = False
        for i in range(len(current_terms)):
            if time.time() > deadline - 10:
                break
            # Try removing term i
            test_terms = [current_terms[j] for j in range(len(current_terms)) if j != i]
            U, V, W = build_matrices(test_terms)
            if verify_decomposition(U, V, W, max_checks=30):
                current_terms = test_terms
                current_rank -= 1
                removed_any = True
                break  # Restart the loop after removing a term

    # If we have time, try removing pairs of terms
    if time.time() < deadline - 20 and current_rank > 30:
        for i in range(len(current_terms)):
            if time.time() > deadline - 20:
                break
            for j in range(i + 1, len(current_terms)):
                if time.time() > deadline - 20:
                    break
                test_terms = [current_terms[k] for k in range(len(current_terms)) if k != i and k != j]
                U, V, W = build_matrices(test_terms)
                if verify_decomposition(U, V, W, max_checks=20):
                    current_terms = test_terms
                    current_rank -= 2
                    break
            if time.time() > deadline - 20:
                break

    # Final verification and return
    U, V, W = build_matrices(current_terms)
    final_rank = len(current_terms)

    # Final verification with more checks
    if not verify_decomposition(U, V, W, max_checks=50):
        # Fallback to schoolbook
        rank_terms_fb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rank_terms_fb.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        U, V, W = build_matrices(rank_terms_fb)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
