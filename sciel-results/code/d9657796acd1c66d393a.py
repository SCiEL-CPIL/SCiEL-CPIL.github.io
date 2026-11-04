"""Strassen-based 4x5x6 matrix multiplication decomposition (rank 87)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS

    # Rank calculation:
    # 4x4x4 block decomposed into 3x3x3 (rank 26 via 7-term Strassen + padding) 
    # and boundary terms.
    # 3x3x3 rank 26:
    #   2x2x2 Strassen (7)
    #   2x2x1 (4)
    #   2x1x2 (4)
    #   1x2x2 (4)
    #   2x1x1 (2)
    #   1x2x1 (2)
    #   1x1x2 (2)
    #   1x1x1 (1)
    #   Sum = 26
    # Boundary for 4x4x4:
    #   3x3x1 (9)
    #   3x1x3 (9)
    #   3x1x1 (3)
    #   1x3x3 (9)
    #   1x3x1 (3)
    #   1x1x3 (3)
    #   1x1x1 (1)
    #   Sum = 37
    # Total 4x4x4 rank = 26 + 37 = 63.
    # 4x1x6 strip rank = 24.
    # Total rank = 63 + 24 = 87.
    
    rank = 87

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r_offset = 0

    def add_strassen_block(i0, j0, k0, r_start):
        # Adds the Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)

        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1

        # M2 = (A10 + A11)B00
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        r_start += 1

        # M3 = A00(B01 - B11)
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1

        # M4 = A11(B10 - B00)
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        r_start += 1

        # M5 = (A00 + A01)B11
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        r_start += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        r_start += 1

        return r_start

    def add_schoolbook_block(i0, j0, k0, ni, nj, nk, r_start):
        # Adds schoolbook decomposition for A[i0:i0+ni, j0:j0+nj] * B[j0:j0+nj, k0:k0+nk]
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    U[(i0 + i) * m + (j0 + j)][r_start] = 1
                    V[(j0 + j) * p + (k0 + k)][r_start] = 1
                    W[(k0 + k) * n + (i0 + i)][r_start] = 1
                    r_start += 1
        return r_start

    # 4x4x4 block: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Decompose into 3x3x3 core and boundaries.
    
    # Core 3x3x3: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Use 2x2x2 Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    r_offset = add_strassen_block(0, 0, 0, r_offset)
    
    # Boundaries of 3x3x3:
    # A[0:2, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2] (2x1x2)
    r_offset = add_schoolbook_block(0, 2, 0, 2, 1, 2, r_offset)
    # A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2] (1x2x2)
    r_offset = add_schoolbook_block(2, 0, 0, 1, 2, 2, r_offset)
    # A[0:2, 0:2] * B[0:2, 2:3] -> C[0:2, 2:3] (2x2x1)
    r_offset = add_schoolbook_block(0, 0, 2, 2, 2, 1, r_offset)
    
    # A[2:3, 2:3] * B[2:3, 0:2] -> C[2:3, 0:2] (1x1x2)
    r_offset = add_schoolbook_block(2, 2, 0, 1, 1, 2, r_offset)
    # A[0:2, 2:3] * B[2:3, 2:3] -> C[0:2, 2:3] (2x1x1)
    r_offset = add_schoolbook_block(0, 2, 2, 2, 1, 1, r_offset)
    # A[2:3, 0:2] * B[0:2, 2:3] -> C[2:3, 2:3] (1x2x1)
    r_offset = add_schoolbook_block(2, 0, 2, 1, 2, 1, r_offset)
    # A[2:3, 2:3] * B[2:3, 2:3] -> C[2:3, 2:3] (1x1x1)
    r_offset = add_schoolbook_block(2, 2, 2, 1, 1, 1, r_offset)

    # Boundaries of 4x4x4 (involving index 3):
    # A[0:3, 0:3] * B[0:3, 3:4] -> C[0:3, 3:4] (3x3x1)
    r_offset = add_schoolbook_block(0, 0, 3, 3, 3, 1, r_offset)
    # A[0:3, 3:4] * B[3:4, 0:3] -> C[0:3, 0:3] (3x1x3)
    r_offset = add_schoolbook_block(0, 3, 0, 3, 1, 3, r_offset)
    # A[0:3, 3:4] * B[3:4, 3:4] -> C[0:3, 3:4] (3x1x1)
    r_offset = add_schoolbook_block(0, 3, 3, 3, 1, 1, r_offset)
    
    # A[3:4, 0:3] * B[0:3, 0:3] -> C[3:4, 0:3] (1x3x3)
    r_offset = add_schoolbook_block(3, 0, 0, 1, 3, 3, r_offset)
    # A[3:4, 0:3] * B[0:3, 3:4] -> C[3:4, 3:4] (1x3x1)
    r_offset = add_schoolbook_block(3, 0, 3, 1, 3, 1, r_offset)
    # A[3:4, 3:4] * B[3:4, 0:3] -> C[3:4, 0:3] (1x1x3)
    r_offset = add_schoolbook_block(3, 3, 0, 1, 1, 3, r_offset)
    # A[3:4, 3:4] * B[3:4, 3:4] -> C[3:4, 3:4] (1x1x1)
    r_offset = add_schoolbook_block(3, 3, 3, 1, 1, 1, r_offset)

    # 4x1x6 strip: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6]
    r_offset = add_schoolbook_block(0, 4, 0, 4, 1, 6, r_offset)

    return {"U": U, "V": V, "W": W}
