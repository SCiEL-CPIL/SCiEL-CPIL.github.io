import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # X indices in A: rows i0:i0+2, cols j0:j0+2
    # Y indices in B: rows j0:j0+2, cols k0:k0+2
    # Z indices in C: rows i0:i0+2, cols k0:k0+2
    def add_strassen_222(i0, j0, k0, col_offset):
        nonlocal col
        # M1 = (X11+X22)(Y11+Y22)
        # U: X11, X22. V: Y11, Y22. W: Z11, Z22
        # Z11 += M1, Z22 += M1
        U[i0*m + j0][col_offset] = 1; V[j0*p + k0][col_offset] = 1; W[k0*n + i0][col_offset] = 1
        U[(i0+1)*m + (j0+1)][col_offset] = 1; V[(j0+1)*p + (k0+1)][col_offset] = 1; W[(k0+1)*n + (i0+1)][col_offset] = 1
        col_offset += 1
        
        # M2 = (X21+X22)Y11
        # U: X21, X22. V: Y11. W: Z21
        U[(i0+1)*m + j0][col_offset] = 1; V[j0*p + k0][col_offset] = 1; W[k0*n + (i0+1)][col_offset] = 1
        U[(i0+1)*m + (j0+1)][col_offset] = 1; V[j0*p + k0][col_offset] = 1; W[k0*n + (i0+1)][col_offset] = 1
        col_offset += 1
        
        # M3 = X11(Y12-Y22)
        # U: X11. V: Y12, -Y22. W: Z12
        U[i0*m + j0][col_offset] = 1; V[j0*p + (k0+1)][col_offset] = 1; W[(k0+1)*n + i0][col_offset] = 1
        U[i0*m + j0][col_offset] = 1; V[(j0+1)*p + (k0+1)][col_offset] = -1; W[(k0+1)*n + i0][col_offset] = 1
        col_offset += 1
        
        # M4 = X22(Y21-Y11)
        # U: X22. V: Y21, -Y11. W: Z11, Z21
        U[(i0+1)*m + (j0+1)][col_offset] = 1; V[(j0+1)*p + k0][col_offset] = 1; W[k0*n + i0][col_offset] = 1
        U[(i0+1)*m + (j0+1)][col_offset] = 1; V[j0*p + k0][col_offset] = -1; W[k0*n + (i0+1)][col_offset] = 1
        col_offset += 1
        
        # M5 = (X11+X12)Y22
        # U: X11, X12. V: Y22. W: Z11, Z12
        U[i0*m + j0][col_offset] = 1; V[(j0+1)*p + (k0+1)][col_offset] = 1; W[k0*n + i0][col_offset] = -1
        U[i0*m + (j0+1)][col_offset] = 1; V[(j0+1)*p + (k0+1)][col_offset] = 1; W[(k0+1)*n + i0][col_offset] = 1
        col_offset += 1
        
        # M6 = (X21-X11)(Y11+Y12)
        # U: X21, -X11. V: Y11, Y12. W: Z22
        U[(i0+1)*m + j0][col_offset] = 1; V[j0*p + k0][col_offset] = 1; W[(k0+1)*n + (i0+1)][col_offset] = 1
        U[i0*m + j0][col_offset] = -1; V[j0*p + k0][col_offset] = 1; W[(k0+1)*n + (i0+1)][col_offset] = 1
        U[(i0+1)*m + j0][col_offset] = 1; V[j0*p + (k0+1)][col_offset] = 1; W[(k0+1)*n + (i0+1)][col_offset] = 1
        U[i0*m + j0][col_offset] = -1; V[j0*p + (k0+1)][col_offset] = 1; W[(k0+1)*n + (i0+1)][col_offset] = 1
        col_offset += 1
        
        # M7 = (X12-X22)(Y21+Y22)
        # U: X12, -X22. V: Y21, Y22. W: Z11, Z12
        U[i0*m + (j0+1)][col_offset] = 1; V[(j0+1)*p + k0][col_offset] = 1; W[k0*n + i0][col_offset] = 1
        U[(i0+1)*m + (j0+1)][col_offset] = -1; V[(j0+1)*p + k0][col_offset] = 1; W[k0*n + i0][col_offset] = 1
        U[i0*m + (j0+1)][col_offset] = 1; V[(j0+1)*p + (k0+1)][col_offset] = 1; W[(k0+1)*n + i0][col_offset] = 1
        U[(i0+1)*m + (j0+1)][col_offset] = -1; V[(j0+1)*p + (k0+1)][col_offset] = 1; W[(k0+1)*n + i0][col_offset] = 1
        col_offset += 1
        
        return col_offset

    # Helper to add a standard 2x2x21 block
    # X indices in A: rows i0:i0+2, cols j0:j0+2
    # Y indices in B: rows j0:j0+2, cols k0:k0+1
    # Z indices in C: rows i0:i0+2, cols k0:k0+1
    def add_standard_221(i0, j0, k0, col_offset):
        nonlocal col
        # Z11 = X11 Y1 + X12 Y2
        # Z21 = X21 Y1 + X22 Y2
        U[i0*m + j0][col_offset] = 1; V[j0*p + k0][col_offset] = 1; W[k0*n + i0][col_offset] = 1
        col_offset += 1
        U[i0*m + (j0+1)][col_offset] = 1; V[(j0+1)*p + k0][col_offset] = 1; W[k0*n + i0][col_offset] = 1
        col_offset += 1
        U[(i0+1)*m + j0][col_offset] = 1; V[j0*p + k0][col_offset] = 1; W[k0*n + (i0+1)][col_offset] = 1
        col_offset += 1
        U[(i0+1)*m + (j0+1)][col_offset] = 1; V[(j0+1)*p + k0][col_offset] = 1; W[k0*n + (i0+1)][col_offset] = 1
        col_offset += 1
        return col_offset

    # Part 1: C_left = A * B1 (2x2 result)
    # A11 * B11 (i=0, j=0, k=0)
    col = add_strassen_222(0, 0, 0, col)
    # A12 * B21 (i=0, j=2, k=0)
    col = add_strassen_222(0, 2, 0, col)
    
    # Part 2: C_right = A * B2 (2x3 result)
    # Split B2 into B2a (4x2) and B2b (4x1)
    # A * B2a (2x2 result)
    # A11 * B1a (i=0, j=0, k=2)
    col = add_strassen_222(0, 0, 2, col)
    # A12 * B2a (i=0, j=2, k=2)
    col = add_strassen_222(0, 2, 2, col)
    
    # A * B2b (2x1 result)
    # A11 * B1b (i=0, j=0, k=4)
    col = add_standard_221(0, 0, 4, col)
    # A12 * B2b (i=0, j=2, k=4)
    col = add_standard_221(0, 2, 4, col)
    
    return {"U": U, "V": V, "W": W}
