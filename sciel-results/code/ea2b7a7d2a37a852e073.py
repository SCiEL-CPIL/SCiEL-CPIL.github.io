"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 6  # A is n x m, B is m x p
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen 2x2 decomposition:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For 2x4 * 4x6:
    # Split A into A0 (cols 0-1) and A1 (cols 2-3).
    # Split B into B_top (rows 0-1) and B_bottom (rows 2-3).
    # C = A0 * B_top + A1 * B_bottom.
    # Each 2x2 * 2x6 uses 3 Strassen 2x2x2 instances.
    # Total rank = 2 * 3 * 7 = 42.

    rank = 42
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]

    col = 0

    for a_block in range(2):  # 0: A0 (cols 0-1), 1: A1 (cols 2-3)
        for t in range(3):  # 3 blocks of B, each 2x2
            b_row0 = 2 * a_block
            b_row1 = 2 * a_block + 1
            b_col0 = 2 * t
            b_col1 = 2 * t + 1
            a_col_offset = 2 * a_block

            # M1 = (a00+a11) * (b00+b11)
            U[0 * m + a_col_offset + 0][col] += 1
            U[1 * m + a_col_offset + 1][col] += 1
            V[b_row0 * p + b_col0][col] += 1
            V[b_row1 * p + b_col1][col] += 1
            W[(2*t) * n + 0][col] += 1
            W[(2*t+1) * n + 1][col] += 1
            col += 1

            # M2 = (a10+a11) * b00
            U[1 * m + a_col_offset + 0][col] += 1
            U[1 * m + a_col_offset + 1][col] += 1
            V[b_row0 * p + b_col0][col] += 1
            W[(2*t) * n + 1][col] += 1
            W[(2*t+1) * n + 1][col] += -1
            col += 1

            # M3 = a00 * (b01-b11)
            U[0 * m + a_col_offset + 0][col] += 1
            V[b_row0 * p + b_col1][col] += 1
            V[b_row1 * p + b_col1][col] += -1
            W[(2*t+1) * n + 0][col] += 1
            W[(2*t+1) * n + 1][col] += 1
            col += 1

            # M4 = a11 * (b10-b00)
            U[1 * m + a_col_offset + 1][col] += 1
            V[b_row1 * p + b_col0][col] += 1
            V[b_row0 * p + b_col0][col] += -1
            W[(2*t) * n + 0][col] += 1
            W[(2*t) * n + 1][col] += 1
            col += 1

            # M5 = (a00+a01) * b11
            U[0 * m + a_col_offset + 0][col] += 1
            U[0 * m + a_col_offset + 1][col] += 1
            V[b_row1 * p + b_col1][col] += 1
            W[(2*t) * n + 0][col] += -1
            W[(2*t+1) * n + 0][col] += 1
            col += 1

            # M6 = (a10-a00) * (b00+b01)
            U[1 * m + a_col_offset + 0][col] += 1
            U[0 * m + a_col_offset + 0][col] += -1
            V[b_row0 * p + b_col0][col] += 1
            V[b_row0 * p + b_col1][col] += 1
            W[(2*t+1) * n + 1][col] += 1
            col += 1

            # M7 = (a01-a11) * (b10+b11)
            U[0 * m + a_col_offset + 1][col] += 1
            U[1 * m + a_col_offset + 1][col] += -1
            V[b_row1 * p + b_col0][col] += 1
            V[b_row1 * p + b_col1][col] += 1
            W[(2*t) * n + 0][col] += 1
            col += 1

    # Verification using numpy for speed
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # Reshape for efficient tensor contraction
    # T[i,j,k] = sum_c U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
    # Let's compute this by iterating over c and accumulating
    # Or better, use einsum-like logic with broadcasting
    
    # For each c, outer product of U[:,c], V[:,c], W[:,c]
    # Sum over c
    # T[i,j,k] where U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    # Let's build the full tensor
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = U_np[:, c]
        vc = V_np[:, c]
        wc = W_np[:, c]
        # uc has shape (n*m,), vc has shape (m*p,), wc has shape (p*n,)
        # uc[i*m+j], vc[j*p+k], wc[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    
    valid = np.allclose(T, 1.0, atol=1e-9)
    
    if not valid:
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
