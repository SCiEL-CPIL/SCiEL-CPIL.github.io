"""Matrix multiplication 4x4x6 via Strassen + rank-reduced 4x4x2 block."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _get_strassen_2x2_factors():
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prods = [
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)]),
        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.max(np.abs(T - 1.0))

def _rank_reduce(U, V, W, n, m, p, rank, tol=1e-9, deadline=None):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    flat_size = n * m * p
    Tmat = np.zeros((flat_size, rank))
    for c in range(rank):
        Tc = np.einsum('ij,jk,ki->ijk',
                       Ua[:, c].reshape(n, m),
                       Va[:, c].reshape(m, p),
                       Wa[:, c].reshape(p, n))
        Tmat[:, c] = Tc.flatten()

    U_reduced, V_reduced, W_reduced = [], [], []
    Q = None
    num_kept = 0

    for c in range(rank):
        if deadline and time.time() > deadline:
            break
        col = Tmat[:, c:c+1]
        col_norm = np.linalg.norm(col)
        if col_norm < tol:
            continue
        if num_kept == 0:
            Q = col / col_norm
            U_reduced.append(Ua[:, c])
            V_reduced.append(Va[:, c])
            W_reduced.append(Wa[:, c])
            num_kept += 1
        else:
            coeff = Q.T @ col
            residual = col - Q @ coeff
            resid_norm = np.linalg.norm(residual)
            if resid_norm <= tol * max(1.0, col_norm):
                continue
            else:
                q_new = residual / resid_norm
                Q = np.hstack([Q, q_new])
                U_reduced.append(Ua[:, c])
                V_reduced.append(Va[:, c])
                W_reduced.append(Wa[:, c])
                num_kept += 1

    if num_kept < rank and num_kept > 0:
        U_new = np.column_stack(U_reduced)
        V_new = np.column_stack(V_reduced)
        W_new = np.column_stack(W_reduced)
        Ur = U_new.reshape(n, m, num_kept)
        Vr = V_new.reshape(m, p, num_kept)
        Wr = W_new.reshape(p, n, num_kept)
        T_check = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        if np.max(np.abs(T_check - 1.0)) < 1e-6:
            return U_new.tolist(), V_new.tolist(), W_new.tolist(), num_kept
    return U, V, W, rank

def _build_444_strassen():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication."""
    U_s, V_s, W_s = _get_strassen_2x2_factors()
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for i in range(4):
        I, i0 = i // 2, i % 2
        for j in range(4):
            J, j0 = j // 2, j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * m + j, col] += val

    for j in range(4):
        J, j0 = j // 2, j % 2
        for k in range(4):
            K, k0 = k // 2, k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * p + k, col] += val

    for k in range(4):
        K, k0 = k // 2, k % 2
        for i in range(4):
            I, i0 = i // 2, i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * n + i, col] += val

    return U, V, W

def _build_442_block():
    """Build rank-28 decomposition for 4x4x2 (cols 4-5) using 4 Strassen blocks."""
    U_s, V_s, W_s = _get_strassen_2x2_factors()
    n, m, p = 4, 4, 2
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    products = [
        (0, 0, 0),  # A_00 * B_top
        (0, 1, 1),  # A_01 * B_bot
        (1, 0, 0),  # A_10 * B_top
        (1, 1, 1),  # A_11 * B_bot
    ]

    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = p_idx * 7
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * m + j, col_offset + s] += val
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = k0  # only 2 cols (0,1)
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * p + k_global, col_offset + s] += val
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * n + i_global, col_offset + s] += val

    return U, V, W

def _combine_444_442(U444, V444, W444, U442, V442, W442):
    """Combine 4x4x4 and 4x4x2 decompositions into a 4x4x6 decomposition.
    
    The 4x4x6 problem: A (4x4) * B (4x6) -> C (4x6).
    Split B into B_left (4x4) and B_right (4x2).
    Split C into C_left (4x4) and C_right (4x2).
    
    C_left = A * B_left  (4x4 * 4x4 -> 4x4)
    C_right = A * B_right (4x4 * 4x2 -> 4x2)
    
    For the 4x4x6 tensor, indices are (i, j, k) where:
    - i: row of A and C (0..3)
    - j: col of A / row of B (0..3)
    - k: col of B / col of C (0..5)
    
    U has shape (n*m, rank) = (16, rank), indexed by [i*m + j]
    V has shape (m*p, rank) = (24, rank), indexed by [j*p + k]
    W has shape (p*n, rank) = (24, rank), indexed by [k*n + i]
    """
    n, m, p_full = 4, 4, 6
    rank444 = U444.shape[1]
    rank442 = U442.shape[1]
    total_rank = rank444 + rank442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p_full, total_rank))
    W = np.zeros((p_full * n, total_rank))

    # Place 4x4x4 part in first rank444 columns
    # U444: (16, 49) -> U[:, 0:49]
    U[:, :rank444] = U444

    # V444: (16, 49) indexed by [j*4 + k] where k is 0..3
    # In 4x4x6, V is indexed by [j*6 + k] where k is 0..5
    # So V444[j*4 + k] goes to V[j*6 + k]
    for j in range(4):
        for k in range(4):
            V[j * p_full + k, :rank444] = V444[j * 4 + k, :]

    # W444: (16, 49) indexed by [k*4 + i] where k is 0..3
    # In 4x4x6, W is indexed by [k*4 + i] where k is 0..5
    for k in range(4):
        for i in range(4):
            W[k * n + i, :rank444] = W444[k * 4 + i, :]

    # Place 4x4x2 part in remaining rank442 columns
    # U442: (16, 28) -> U[:, 49:77]
    U[:, rank444:] = U442

    # V442: (8, 28) indexed by [j*2 + k] where j is 0..3, k is 0..1
    # In 4x4x6, V is indexed by [j*6 + k] where j is 0..3, k is 0..5
    # B_right has columns 4,5 of B, so k in 442 corresponds to k=4,5 in 4x4x6
    for j in range(4):
        for k in range(2):
            V[j * p_full + (4 + k), rank444:] = V442[j * 2 + k, :]

    # W442: (8, 28) indexed by [k*4 + i] where k is 0..1
    # In 4x4x6, W is indexed by [k*4 + i] where k is 0..5
    # C_right has columns 4,5, so k in 442 corresponds to k=4,5 in 4x4x6
    for k in range(2):
        for i in range(4):
            W[(4 + k) * n + i, rank444:] = W442[k * 4 + i, :]

    return U.tolist(), V.tolist(), W.tolist(), total_rank

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0

    # Build 4x4x4 Strassen decomposition (rank 49)
    U444, V444, W444 = _build_444_strassen()

    # Build 4x4x2 decomposition (rank 28)
    U442, V442, W442 = _build_442_block()

    # Combine into 4x4x6
    U, V, W, total_rank = _combine_444_442(U444, V444, W444, U442, V442, W442)

    # Verify
    err = _verify(U, V, W, n, m, p, total_rank)
    if err > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try rank reduction
    if time.time() < deadline - 5.0:
        U_red, V_red, W_red, new_rank = _rank_reduce(U, V, W, n, m, p, total_rank, deadline=deadline - 3.0)
        if new_rank < total_rank:
            return {"U": U_red, "V": V_red, "W": W_red}

    return {"U": U, "V": V, "W": W}
