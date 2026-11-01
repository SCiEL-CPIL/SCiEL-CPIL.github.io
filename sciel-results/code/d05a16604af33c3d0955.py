import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition with rank 30.
    # U: 9 x R, V: 21 x R, W: 21 x R
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    R = 30
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term for a 2x2x2 block
    # A block is at (i0, j0), B block is at (j0, k0)
    # The output C block is at (i0, k0)
    # i0, j0, k0 are offsets.
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen factors for 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 + B11)
        # M4 = A11B11
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Term 1
        # U: A00, A11
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        # V: B00, B11
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C00, C11 (with signs)
        # C00 += M1, C11 += M1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # Term 2
        # U: A10, A11
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        # V: B00
        V[j0*p + k0][col] = 1
        # W: C10 += M2, C11 -= M2
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = -1
        col += 1
        
        # Term 3
        # U: A00
        U[i0*m + j0][col] = 1
        # V: B10, B11
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C01 += M3, C10 += M3
        W[(k0+1)*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # Term 4
        # U: A11
        U[(i0+1)*m + (j0+1)][col] = 1
        # V: B11
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C00 += M4, C10 += M4
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # Term 5
        # U: A00, A01
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        # V: B11
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C00 -= M5, C01 += M5
        W[k0*n + i0][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # Term 6
        # U: A10, -A00
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        # V: B00, B10
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        # W: C11 += M6
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # Term 7
        # U: A01, -A11
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        # V: B10, B11
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C00 += M7
        W[k0*n + i0][col] = 1
        col += 1

    # Apply Strassen to the three 2x2x2 blocks of A11 * B1
    # A11 is at i=0,1; j=0,1
    # B11 is at j=0,1; k=0,1
    add_strassen(0, 0, 0)
    # B12 is at j=0,1; k=2,3
    add_strassen(0, 0, 2)
    # B13 is at j=0,1; k=4,5
    add_strassen(0, 0, 4)
    
    # Now handle A11 * B14 (2x2x1) at k=6
    # Standard rank 4
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    # Term 1: A00 B00
    U[0*m + 0][col] = 1
    V[0*p + 6][col] = 1
    W[6*n + 0][col] = 1
    col += 1
    # Term 2: A01 B10
    U[0*m + 1][col] = 1
    V[1*p + 6][col] = 1
    W[6*n + 0][col] = 1
    col += 1
    # Term 3: A10 B00
    U[1*m + 0][col] = 1
    V[0*p + 6][col] = 1
    W[6*n + 1][col] = 1
    col += 1
    # Term 4: A11 B10
    U[1*m + 1][col] = 1
    V[1*p + 6][col] = 1
    W[6*n + 1][col] = 1
    col += 1

    # Now handle A12 * B2 (2x1x7)
    # A12 is at i=0,1; j=2
    # B2 is at j=2; k=0..6
    # C0k = A02 B2k
    # C1k = A12 B2k
    # Term 5: A02 * B2
    U[0*m + 2][col] = 1
    for k in range(7):
        V[2*p + k][col] = 1
        W[k*n + 0][col] = 1
    col += 1
    # Term 6: A12 * B2
    U[1*m + 2][col] = 1
    for k in range(7):
        V[2*p + k][col] = 1
        W[k*n + 1][col] = 1
    col += 1

    # Now handle A21 * B1 (1x2x7)
    # A21 is at i=2; j=0,1
    # B1 is at j=0,1; k=0..6
    # C2k = A20 B0k + A21 B1k
    # Term 7: A20 * B0
    U[2*m + 0][col] = 1
    for k in range(7):
        V[0*p + k][col] = 1
        W[k*n + 2][col] = 1
    col += 1
    # Term 8: A21 * B1
    U[2*m + 1][col] = 1
    for k in range(7):
        V[1*p + k][col] = 1
        W[k*n + 2][col] = 1
    col += 1

    # Now handle A22 * B2 (1x1x7)
    # A22 is at i=2; j=2
    # B2 is at j=2; k=0..6
    # C2k = A22 B2k
    # Term 9: A22 * B2
    U[2*m + 2][col] = 1
    for k in range(7):
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
    col += 1

    # Total columns used: 21 + 4 + 2 + 2 + 1 = 30.
    # R was 30.
    
    return {"U": U, "V": V, "W": W}
