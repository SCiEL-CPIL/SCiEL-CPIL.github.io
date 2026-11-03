"""Matrix multiplication 2x4x5 rank-minimal decomposition using Strassen blocks."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * rank for _ in range(p * n)]   # 10 x 36
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """Add Strassen 2x2x2: A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        # p1 = (A00+A11)(B00+B01)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        
        # p2 = (A10+A11)B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        
        # p3 = A00(B01-B11)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # p4 = A11(B10-B00)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # p5 = (A00+A01)B11
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += -1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # p6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # p7 = (A01-A11)(B10+B11)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
    
    def add_schoolbook_221(i0, j0, k0):
        """Add schoolbook 2x2x1: A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
        nonlocal col
        for ii in range(2):
            for jj in range(2):
                U[(i0+ii)*m + (j0+jj)][col] += 1
                V[(j0+jj)*p + k0][col] += 1
                W[k0*n + (i0+ii)][col] += 1
                col += 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:5] (2x2x5)
    add_strassen_222(0, 0, 0)   # B[0:2, 0:2]
    add_strassen_222(0, 0, 2)   # B[0:2, 2:4]
    add_schoolbook_221(0, 0, 4) # B[0:2, 4:5]
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:5] (2x2x5)
    add_strassen_222(0, 2, 0)   # B[2:4, 0:2]
    add_strassen_222(0, 2, 2)   # B[2:4, 2:4]
    add_schoolbook_221(0, 2, 4) # B[2:4, 4:5]
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
