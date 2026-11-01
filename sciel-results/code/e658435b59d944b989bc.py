import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 matrix multiplication
    # We have 4 blocks of 2x2 multiplication.
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Strassen formulas:
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y12 - Y22)
    # M4 = X22(Y21 - Y11)
    # M5 = (X11 + X12)Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6

    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, base):
        # a_r0, a_c0: top-left indices of A block (size 2x2)
        # b_r0, b_c0: top-left indices of B block (size 2x2)
        # c_r0, c_c0: top-left indices of C block (size 2x2)
        # base: starting column index in U, V, W
        
        # Indices for A:
        # X11: a_r0, a_c0
        # X12: a_r0, a_c0+1
        # X21: a_r0+1, a_c0
        # X22: a_r0+1, a_c0+1
        
        # Indices for B:
        # Y11: b_r0, b_c0
        # Y12: b_r0, b_c0+1
        # Y21: b_r0+1, b_c0
        # Y22: b_r0+1, b_c0+1
        
        # Indices for C:
        # Z11: c_r0, c_c0
        # Z12: c_r0, c_c0+1
        # Z21: c_r0+1, c_c0
        # Z22: c_r0+1, c_c0+1
        
        # W index convention: W[k*n + i, r] where C[i, k] is the element.
        # Wait, task says: "output factor follows the official trace convention W[k*n+i,r]"
        # This usually means W is a matrix of size (p*n, rank).
        # Row index of W: k*n + i. This corresponds to C[i, k].
        # So for C[i, k], the row in W is k*n + i.
        
        # M1
        col = base
        U[a_r0 * m + a_c0][col] += 1
        U[(a_r0+1) * m + (a_c0+1)][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        W[c_c0 * n + c_r0][col] += 1      # Z11
        W[(c_c0+1) * n + (c_r0+1)][col] += 1 # Z22

        # M2
        col = base + 1
        U[(a_r0+1) * m + a_c0][col] += 1
        U[(a_r0+1) * m + (a_c0+1)][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        W[c_c0 * n + (c_r0+1)][col] += 1      # Z21
        W[(c_c0+1) * n + (c_r0+1)][col] -= 1  # Z22

        # M3
        col = base + 2
        U[a_r0 * m + a_c0][col] += 1
        V[b_r0 * p + (b_c0+1)][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] -= 1
        W[(c_c0+1) * n + c_r0][col] += 1      # Z12
        W[(c_c0+1) * n + (c_r0+1)][col] += 1  # Z22

        # M4
        col = base + 3
        U[(a_r0+1) * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + b_c0][col] += 1
        V[b_r0 * p + b_c0][col] -= 1
        W[c_c0 * n + c_r0][col] += 1      # Z11
        W[c_c0 * n + (c_r0+1)][col] += 1  # Z21

        # M5
        col = base + 4
        U[a_r0 * m + a_c0][col] += 1
        U[a_r0 * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        W[c_c0 * n + c_r0][col] -= 1      # Z11
        W[(c_c0+1) * n + c_r0][col] += 1  # Z12

        # M6
        col = base + 5
        U[(a_r0+1) * m + a_c0][col] += 1
        U[a_r0 * m + a_c0][col] -= 1
        V[b_r0 * p + b_c0][col] += 1
        V[b_r0 * p + (b_c0+1)][col] += 1
        W[(c_c0+1) * n + (c_r0+1)][col] += 1 # Z22

        # M7
        col = base + 6
        U[a_r0 * m + (a_c0+1)][col] += 1
        U[(a_r0+1) * m + (a_c0+1)][col] -= 1
        V[(b_r0+1) * p + b_c0][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        W[c_c0 * n + c_r0][col] += 1      # Z11

    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0, 7)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2, 14)
    
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2, 2, 0, 2, 21)

    return {"U": U, "V": V, "W": W}
