"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Decompose 3x5x5 into 2x5x5 (rows 0,1) + 1x5x5 (row 2)
    # For 2x5x5: partition j,k into {0,1},{2,3},{4}
    # Apply Strassen to 2x2x2 blocks: (j0,k0) in {(0,0),(0,2),(2,0),(2,2)}
    # Each Strassen block: rank 7. 4 blocks: 28
    # Remaining terms for 2x5x5: j=4 (5 terms) + k=4 with j in {0,1,2,3} (4 terms) = 9
    # Subtotal 2x5x5: 37
    # For 1x5x5: schoolbook = 25
    # Total rank: 62
    
    total_rank = 62
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # (15, 62)
    V = [[0.0] * total_rank for _ in range(m * p)]  # (25, 62)
    W = [[0.0] * total_rank for _ in range(p * n)]  # (15, 62)
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2 at offsets (i0, j0, k0)
    # i0 is the starting row index (0 for our 2x5x5 block)
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    def add_strassen_block(i0, j0, k0, start_col):
        # M1 = (A00+A11)(B00+B11)
        c = start_col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = start_col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = start_col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = start_col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = start_col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = start_col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = start_col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        
        return start_col + 7
    
    # Add Strassen blocks for 2x5x5 (i0=0, covering rows 0 and 1)
    strassen_blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
    ]
    
    for (i0, j0, k0) in strassen_blocks:
        col = add_strassen_block(i0, j0, k0, col)
    
    # Add schoolbook terms for remaining parts of 2x5x5
    # Terms NOT covered by Strassen blocks (i in {0,1})
    for i in range(2):
        for j in range(m):
            for k in range(p):
                covered = False
                for (i0, j0, k0) in strassen_blocks:
                    if i in (i0, i0+1) and j in (j0, j0+1) and k in (k0, k0+1):
                        covered = True
                        break
                if not covered:
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
    
    # Add schoolbook terms for 1x5x5 (i=2)
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 2][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
