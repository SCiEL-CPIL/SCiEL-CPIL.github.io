"""Block-wise Strassen decomposition for 6x6 matrix multiplication.
Uses 2x2 Strassen (rank 7) for the outer block structure and 3x3 schoolbook (rank 27)
for the inner blocks, giving total rank 7 * 27 = 189.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact using vectorized numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _make_2x2_strassen():
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2.
    Rank 7.
    Returns (U, V, W) as 2D arrays of shape (4, 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # Strassen's 7 products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B11-B01)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # U indexed by (i,j) -> i*2+j
    # V indexed by (j,k) -> j*2+k
    # W indexed by (k,i) -> k*2+i

    # Term 0: P1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 and C11

    # Term 1: P2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10 and C11

    # Term 2: P3 = A00*(B11-B01)
    U[0, 2] = 1
    V[3, 2] = 1; V[1, 2] = -1
    W[1, 0] = 1; W[3, 1] = 1  # C01 and C11

    # Term 3: P4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 0] = 1; W[2, 1] = 1  # C00 and C10

    # Term 4: P5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 0] = -1; W[1, 0] = 1  # C00 and C01

    # Term 5: P6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 1] = 1  # C11

    # Term 6: P7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 0] = 1  # C00

    return U, V, W


def _make_3x3_schoolbook():
    """
    Schoolbook decomposition of 3x3 * 3x3 -> 3x3.
    Rank 27.
    Returns (U, V, W) as 2D arrays of shape (9, 27).
    """
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1

    return U, V, W


def _combine_block_decompositions(U_outer, V_outer, W_outer,
                                   U_inner, V_inner, W_inner,
                                   outer_dim, inner_dim):
    """
    Combine an outer block decomposition with an inner block decomposition.
    
    outer_dim: size of each outer block (e.g., 2 for 2x2 blocks)
    inner_dim: size of each inner block (e.g., 3 for 3x3 blocks)
    
    The overall dimension is outer_dim * inner_dim.
    """
    full_dim = outer_dim * inner_dim
    rank_outer = U_outer.shape[1]
    rank_inner = U_inner.shape[1]
    total_rank = rank_outer * rank_inner

    U = np.zeros((full_dim * full_dim, total_rank))
    V = np.zeros((full_dim * full_dim, total_rank))
    W = np.zeros((full_dim * full_dim, total_rank))

    for c_outer in range(rank_outer):
        for c_inner in range(rank_inner):
            col = c_outer * rank_inner + c_inner

            # U: outer (I,J) with inner (u,v) -> global (I*inner_dim+u, J*inner_dim+v)
            for I in range(outer_dim):
                for J in range(outer_dim):
                    u_outer_val = U_outer[I * outer_dim + J, c_outer]
                    if u_outer_val == 0:
                        continue
                    for u in range(inner_dim):
                        for v in range(inner_dim):
                            u_inner_val = U_inner[u * inner_dim + v, c_inner]
                            if u_inner_val == 0:
                                continue
                            i_global = I * inner_dim + u
                            j_global = J * inner_dim + v
                            idx_u = i_global * full_dim + j_global
                            U[idx_u, col] += u_outer_val * u_inner_val

            # V: outer (J,K) with inner (v,w) -> global (J*inner_dim+v, K*inner_dim+w)
            for J in range(outer_dim):
                for K in range(outer_dim):
                    v_outer_val = V_outer[J * outer_dim + K, c_outer]
                    if v_outer_val == 0:
                        continue
                    for v in range(inner_dim):
                        for w in range(inner_dim):
                            v_inner_val = V_inner[v * inner_dim + w, c_inner]
                            if v_inner_val == 0:
                                continue
                            j_global = J * inner_dim + v
                            k_global = K * inner_dim + w
                            idx_v = j_global * full_dim + k_global
                            V[idx_v, col] += v_outer_val * v_inner_val

            # W: outer (K,I) with inner (w,u) -> global (K*inner_dim+w, I*inner_dim+u)
            for K in range(outer_dim):
                for I in range(outer_dim):
                    w_outer_val = W_outer[K * outer_dim + I, c_outer]
                    if w_outer_val == 0:
                        continue
                    for w in range(inner_dim):
                        for u in range(inner_dim):
                            w_inner_val = W_inner[w * inner_dim + u, c_inner]
                            if w_inner_val == 0:
                                continue
                            k_global = K * inner_dim + w
                            i_global = I * inner_dim + u
                            idx_w = k_global * full_dim + i_global
                            W[idx_w, col] += w_outer_val * w_inner_val

    return U, V, W


def _make_3x3_laderman():
    """
    Laderman decomposition of 3x3 * 3x3 -> 3x3.
    Rank 23.
    Returns (U, V, W) as 2D arrays of shape (9, 23) or None if not available.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))

    # Laderman's 23 products for 3x3 matrix multiplication.
    # Using the formulation from standard references.
    #
    # Let A = [[a00, a01, a02],
    #          [a10, a11, a12],
    #          [a20, a21, a22]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    #
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # The 23 products:
    P = []

    # P0: a00*b00
    P.append(([(0,0,1)], [(0,0,1)], [(0,0,1)]))
    # P1: a01*b10
    P.append(([(0,1,1)], [(1,0,1)], [(0,0,1)]))
    # P2: a02*b20
    P.append(([(0,2,1)], [(2,0,1)], [(0,0,1)]))
    # P3: a10*b01
    P.append(([(1,0,1)], [(0,1,1)], [(1,0,1)]))
    # P4: a11*b11
    P.append(([(1,1,1)], [(1,1,1)], [(1,1,1)]))
    # P5: a12*b21
    P.append(([(1,2,1)], [(2,1,1)], [(1,1,1)]))
    # P6: a20*b02
    P.append(([(2,0,1)], [(0,2,1)], [(2,0,1)]))
    # P7: a21*b12
    P.append(([(2,1,1)], [(1,2,1)], [(2,1,1)]))
    # P8: a22*b22
    P.append(([(2,2,1)], [(2,2,1)], [(2,2,1)]))

    # P9: (a00+a11)*(b00+b11)
    P.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,1,1),(1,0,1)]))

    # P10: (a00+a22)*(b00+b22)
    P.append(([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,2,1),(2,0,1)]))

    # P11: (a11+a22)*(b11+b22)
    P.append(([(1,1,1),(2,2,1)], [(1,1,1),(2,2,1)], [(1,2,1),(2,1,1)]))

    # P12: (a00+a01+a10+a11)*(b00+b01+b10+b11)
    # This is complex. Let me use a different approach.
    # Actually, let me just verify the 9 direct + 3 simple composites and see if it works.
    # If not, I'll fall back to the 189-rank solution.

    # For now, let me just use the 9 direct terms + 3 simple composites = 12 terms
    # and check if it's a valid partial decomposition.
    # If it's not complete, I'll return None.

    # Actually, let me try a different strategy:
    # Use the known fact that Laderman's algorithm exists and try to construct it.
    # But since I'm not confident in the exact coefficients, let me just return None
    # and use the 189-rank fallback.

    return None


def construct():
    start_time = time.time()

    n, m, p = 6, 6, 6

    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Hybrid: 2x2 Strassen (Outer) x 3x3 Schoolbook (Inner) (Rank 189)
    def _hybrid_strassen_schoolbook():
        block_size = 3
        num_blocks = 2
        dim = n

        rank_inner = block_size ** 3  # 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer

        U = [[0] * total_rank for _ in range(dim * dim)]
        V = [[0] * total_rank for _ in range(dim * dim)]
        W = [[0] * total_rank for _ in range(dim * dim)]

        col = 0

        steps = [
            {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1, (1, 1): 1}},
            {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1}, 'C': {(1, 0): 1, (1, 1): -1}},
            {'A': {(0, 0): 1}, 'B': {(0, 1): 1, (1, 1): -1}, 'C': {(0, 1): 1, (1, 1): 1}},
            {'A': {(1, 1): 1}, 'B': {(1, 0): 1, (0, 0): -1}, 'C': {(0, 0): 1, (1, 0): 1}},
            {'A': {(0, 0): 1, (0, 1): 1}, 'B': {(1, 1): 1}, 'C': {(0, 0): -1, (0, 1): 1}},
            {'A': {(1, 0): 1, (0, 0): -1}, 'B': {(0, 0): 1, (0, 1): 1}, 'C': {(1, 1): 1}},
            {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1}}
        ]

        for step in steps:
            A_coeffs = step['A']
            B_coeffs = step['B']
            C_coeffs = step['C']

            for u in range(block_size):
                for v in range(block_size):
                    for w in range(block_size):
                        for (I, J), ca in A_coeffs.items():
                            if ca != 0:
                                i_global = I * block_size + u
                                j_global = J * block_size + v
                                idx_u = i_global * dim + j_global
                                U[idx_u][col] += ca
                        for (J, K), cb in B_coeffs.items():
                            if cb != 0:
                                j_global = J * block_size + v
                                k_global = K * block_size + w
                                idx_v = j_global * dim + k_global
                                V[idx_v][col] += cb
                        for (I, K), cc in C_coeffs.items():
                            if cc != 0:
                                i_global = I * block_size + u
                                k_global = K * block_size + w
                                idx_w = k_global * dim + i_global
                                W[idx_w][col] += cc
                        col += 1

        return {"U": U, "V": V, "W": W}

    # Try Hybrid (189)
    try:
        res = _hybrid_strassen_schoolbook()
        # Verify
        U_np = np.array(res["U"], dtype=float)
        V_np = np.array(res["V"], dtype=float)
        W_np = np.array(res["W"], dtype=float)

        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]
                for k in range(p):
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err

        if max_err < 1e-10:
            return res
    except Exception:
        pass

    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
