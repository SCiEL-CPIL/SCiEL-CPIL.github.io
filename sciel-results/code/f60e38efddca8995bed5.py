"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    r = 0
    
    # Strassen coefficients for 2x2x2
    # U rows: 0,1,2,3 (A: [[0,0],[0,1],[1,0],[1,1]] mapped to 0,1,2,3? No, A is 2x2)
    # A indices: i in 0,1; j in 0,1.
    # U row index: i*2 + j.
    # V row index: j*2 + k.
    # W row index: k*2 + i.
    
    strassen_u = {
        0: [(0, 1), (3, 1)],  # A00 + A11
        1: [(2, 1), (3, 1)],  # A10 + A11
        2: [(0, 1)],          # A00
        3: [(3, 1)],          # A11
        4: [(0, 1), (1, 1)],  # A00 + A01
        5: [(2, 1), (0, -1)], # A10 - A00
        6: [(1, 1), (3, -1)], # A01 - A11
    }
    
    strassen_v = {
        0: [(0, 1), (1, 1)],  # B00 + B01
        1: [(1, 1)],          # B01
        2: [(2, 1), (3, -1)], # B10 - B11
        3: [(2, 1)],          # B10
        4: [(0, 1), (2, 1)],  # B00 + B10
        5: [(3, 1), (1, 1)],  # B11 + B01
        6: [(3, 1), (2, -1)], # B11 - B10
    }
    
    strassen_w = {
        0: [(0, 1), (3, 1)],  # C11 + C00
        1: [(1, 1)],          # C01
        2: [(2, 1), (3, -1)], # C10 - C11
        3: [(2, 1)],          # C10
        4: [(0, 1), (2, 1)],  # C00 + C10
        5: [(3, 1), (1, 1)],  # C11 + C01
        6: [(0, -1), (1, 1)], # C01 - C00
    }
    
    # We have 5 blocks of 2x2x2 (columns 0-1, 2-3, ..., 8-9)
    # And 1 block of 2x1x10 (row 2 of A, all columns)
    
    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        
        for s in range(7):
            # Update U
            for (u_row_offset, coeff) in strassen_u[s]:
                # u_row_offset is index in 2x2 A block (0..3)
                # Global U row: i*3 + j.
                # In the 2x2 block, i is 0 or 1, j is 0 or 1.
                # u_row_offset = i*2 + j.
                i = u_row_offset // 2
                j = u_row_offset % 2
                global_u_row = i * 3 + j
                U[global_u_row][r] += coeff
            
            # Update V
            for (v_row_offset, coeff) in strassen_v[s]:
                # v_row_offset is index in 2x2 B block (0..3)
                # Global V row: j*10 + k.
                # In the 2x2 block, j is 0 or 1, k is 0 or 1 (local).
                # Local k 0 -> global col0, local k 1 -> global col1.
                j = v_row_offset // 2
                local_k = v_row_offset % 2
                global_k = col0 + local_k
                global_v_row = j * 10 + global_k
                V[global_v_row][r] += coeff
            
            # Update W
            for (w_row_offset, coeff) in strassen_w[s]:
                # w_row_offset is index in 2x2 C block (0..3)
                # Global W row: k*2 + i.
                # In the 2x2 block, k is 0 or 1 (local), i is 0 or 1.
                local_k = w_row_offset // 2
                i = w_row_offset % 2
                global_k = col0 + local_k
                global_w_row = global_k * 2 + i
                W[global_w_row][r] += coeff
            
            r += 1
    
    # Schoolbook for A[:,2] * B[2,:]
    # A row 2 (global index 2), B col 2 (global index 2)
    # C = A[:,2] * B[2,:]
    # C[i, k] = A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[2 * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin under 120s cap

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        U, V, W, rank = _build_trivial(n, m, p)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
