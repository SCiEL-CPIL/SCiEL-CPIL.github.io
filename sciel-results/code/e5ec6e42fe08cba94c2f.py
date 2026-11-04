import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy einsum for speed with early termination."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    diff = np.abs(reconstructed - 1.0)
    if np.any(diff > tol):
        return False
    return np.allclose(reconstructed, 1.0, atol=tol)

def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 x 4x4 using 2x2 Strassen blocks."""
    block_coeffs_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    block_coeffs_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    block_coeffs_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(1, 0, 1), (0, 0, 1)],
        [(0, 1, -1), (1, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc

                for (br, bc, coeff_u) in block_coeffs_U[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U[i * 4 + j][c] += coeff_u

                for (br, bc, coeff_v) in block_coeffs_V[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V[j * 4 + k][c] += coeff_v

                for (br, bc, coeff_w) in block_coeffs_W[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            k = br * 2 + lr2
                            i = bc * 2 + lc2
                            W[k * 4 + i][c] += coeff_w

    return U, V, W

def _try_embed_5th_column(U, V, W, n, m, p, deadline):
    """
    Try to embed the 5th column into the existing 28 terms.
    We need to find v_c (length 4) and w_c (length 4) for each term c
    such that sum_c U_{ij}^c * v_j^c * w_i^c = 1 for all i,j.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    
    # We want to solve: sum_c U_{ij}^c * v_j^c * w_i^c = 1
    # This is a system of 16 equations (for i,j) with 28*4 + 28*4 = 224 variables.
    # We can use a simple iterative approach or linearization.
    # Let's try fixing w_c = 1 for all c, and solving for v_c.
    # sum_c U_{ij}^c * v_j^c = 1
    # This is a linear system for v_c.
    # Variable vector x of length 28*4 = 112.
    # x[c*4 + j] = v_j^c
    
    A_mat = np.zeros((16, 112))
    b_vec = np.ones(16)
    
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            for c in range(rank):
                v_idx = c * m + j
                A_mat[idx, v_idx] = U_arr[i, j, c]
    
    # Solve A_mat * x = b_vec
    # Least squares solution
    try:
        x, residuals, rank_ls, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        # Check if solution is exact
        if np.allclose(A_mat @ x, b_vec, atol=1e-9):
            # Construct V and W for the 5th column
            # V[j, 4, c] = v_j^c
            # W[4, i, c] = w_i^c = 1
            V_new = [row[:] for row in V]
            W_new = [row[:] for row in W]
            
            for c in range(rank):
                for j in range(m):
                    V_new[j * p + 4][c] = x[c * m + j]
                for i in range(n):
                    W_new[4 * n + i][c] = 1.0
            
            # Verify
            if _verify_decomposition_fast(U, V_new, W_new, n, m, p):
                return U, V_new, W_new, rank
    except Exception:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Build 4x4x4 Strassen
    U, V, W = _build_4x4x4_strassen()
    
    # Try to embed 5th column
    result = _try_embed_5th_column(U, V, W, n, m, p, deadline)
    if result:
        U_final, V_final, W_final, rank_final = result
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: Standard 44 rank
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_boundary = 16
    rank_total = rank_core + rank_boundary

    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]

    for c in range(rank_core):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][c] = U44[i * 4 + j][c]
        for j in range(4):
            for k in range(4):
                V[j * 5 + k][c] = V44[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W44[k * 4 + i][c]

    for i in range(4):
        for j in range(4):
            c = rank_core + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1

    return {"U": U, "V": V, "W": W}
