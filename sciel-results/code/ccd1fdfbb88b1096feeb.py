import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 construction
    rank = 42
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # A is 2x2, B is 2x2
    # A rows correspond to global i in {i0, i0+1}, cols j in {j0, j0+1}
    # B rows correspond to global j in {j0, j0+1}, cols k in {k0, k0+1}
    # C cols/rows correspond to global k in {k0, k0+1}, i in {i0, i0+1}
    # Strassen coefficients
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b12-b22)
    # P4 = a22(b21-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    
    def add_strassen(i0, j0, k0, base_col):
        # Map Strassen indices to U, V, W
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # P1
        # a11 (i0, j0), a22 (i0+1, j0+1)
        # b11 (j0, k0), b22 (j0+1, k0+1)
        # Terms:
        # U[i0, j0] += 1, V[j0, k0] += 1, W[k0, i0] += 1
        # U[i0, j0] += 1, V[j0+1, k0+1] += 1, W[k0+1, i0] += 1
        # U[i0+1, j0+1] += 1, V[j0, k0] += 1, W[k0, i0+1] += 1
        # U[i0+1, j0+1] += 1, V[j0+1, k0+1] += 1, W[k0+1, i0+1] += 1
        U[i0*m + j0][base_col] += 1
        V[j0*p + k0][base_col] += 1
        W[k0*n + i0][base_col] += 1
        
        U[i0*m + j0][base_col] += 1
        V[(j0+1)*p + (k0+1)][base_col] += 1
        W[(k0+1)*n + i0][base_col] += 1
        
        U[(i0+1)*m + (j0+1)][base_col] += 1
        V[j0*p + k0][base_col] += 1
        W[k0*n + (i0+1)][base_col] += 1
        
        U[(i0+1)*m + (j0+1)][base_col] += 1
        V[(j0+1)*p + (k0+1)][base_col] += 1
        W[(k0+1)*n + (i0+1)][base_col] += 1
        
        # P2
        # (a21+a22)b11
        # a21 (i0+1, j0), a22 (i0+1, j0+1)
        # b11 (j0, k0)
        U[(i0+1)*m + j0][base_col+1] += 1
        V[j0*p + k0][base_col+1] += 1
        W[k0*n + (i0+1)][base_col+1] += 1
        
        U[(i0+1)*m + (j0+1)][base_col+1] += 1
        V[j0*p + k0][base_col+1] += 1
        W[k0*n + (i0+1)][base_col+1] += 1
        
        # P3
        # a11(b12-b22)
        # a11 (i0, j0)
        # b12 (j0, k0+1), b22 (j0+1, k0+1)
        U[i0*m + j0][base_col+2] += 1
        V[j0*p + (k0+1)][base_col+2] += 1
        W[(k0+1)*n + i0][base_col+2] += 1
        
        U[i0*m + j0][base_col+2] -= 1
        V[(j0+1)*p + (k0+1)][base_col+2] += 1
        W[(k0+1)*n + i0][base_col+2] -= 1
        
        # P4
        # a22(b21-b11)
        # a22 (i0+1, j0+1)
        # b21 (j0+1, k0), b11 (j0, k0)
        U[(i0+1)*m + (j0+1)][base_col+3] += 1
        V[(j0+1)*p + k0][base_col+3] += 1
        W[k0*n + (i0+1)][base_col+3] += 1
        
        U[(i0+1)*m + (j0+1)][base_col+3] -= 1
        V[j0*p + k0][base_col+3] += 1
        W[k0*n + (i0+1)][base_col+3] -= 1
        
        # P5
        # (a11+a12)b22
        # a11 (i0, j0), a12 (i0, j0+1)
        # b22 (j0+1, k0+1)
        U[i0*m + j0][base_col+4] += 1
        V[(j0+1)*p + (k0+1)][base_col+4] += 1
        W[(k0+1)*n + i0][base_col+4] += 1
        
        U[i0*m + (j0+1)][base_col+4] += 1
        V[(j0+1)*p + (k0+1)][base_col+4] += 1
        W[(k0+1)*n + i0][base_col+4] += 1
        
        # P6
        # (a21-a11)(b11+b12)
        # a21 (i0+1, j0), a11 (i0, j0)
        # b11 (j0, k0), b12 (j0, k0+1)
        U[(i0+1)*m + j0][base_col+5] += 1
        V[j0*p + k0][base_col+5] += 1
        W[k0*n + (i0+1)][base_col+5] += 1
        
        U[(i0+1)*m + j0][base_col+5] += 1
        V[j0*p + (k0+1)][base_col+5] += 1
        W[(k0+1)*n + (i0+1)][base_col+5] += 1
        
        U[i0*m + j0][base_col+5] -= 1
        V[j0*p + k0][base_col+5] += 1
        W[k0*n + i0][base_col+5] -= 1
        
        U[i0*m + j0][base_col+5] -= 1
        V[j0*p + (k0+1)][base_col+5] += 1
        W[(k0+1)*n + i0][base_col+5] -= 1
        
        # P7
        # (a12-a22)(b21+b22)
        # a12 (i0, j0+1), a22 (i0+1, j0+1)
        # b21 (j0+1, k0), b22 (j0+1, k0+1)
        U[i0*m + (j0+1)][base_col+6] += 1
        V[(j0+1)*p + k0][base_col+6] += 1
        W[k0*n + i0][base_col+6] += 1
        
        U[i0*m + (j0+1)][base_col+6] += 1
        V[(j0+1)*p + (k0+1)][base_col+6] += 1
        W[(k0+1)*n + i0][base_col+6] += 1
        
        U[(i0+1)*m + (j0+1)][base_col+6] -= 1
        V[(j0+1)*p + k0][base_col+6] += 1
        W[k0*n + (i0+1)][base_col+6] -= 1
        
        U[(i0+1)*m + (j0+1)][base_col+6] -= 1
        V[(j0+1)*p + (k0+1)][base_col+6] += 1
        W[(k0+1)*n + (i0+1)][base_col+6] -= 1

    # We have 6 Strassen blocks
    # Block 1: A_L (i0=0, j0=0) with B_T1 (j0=0, k0=0)
    add_strassen(0, 0, 0, 0)
    # Block 2: A_L (i0=0, j0=0) with B_T2 (j0=0, k0=2)
    add_strassen(0, 0, 2, 7)
    # Block 3: A_L (i0=0, j0=0) with B_T3 (j0=0, k0=4)
    add_strassen(0, 0, 4, 14)
    # Block 4: A_R (i0=0, j0=2) with B_B1 (j0=2, k0=0)
    add_strassen(0, 2, 0, 21)
    # Block 5: A_R (i0=0, j0=2) with B_B2 (j0=2, k0=2)
    add_strassen(0, 2, 2, 28)
    # Block 6: A_R (i0=0, j0=2) with B_B3 (j0=2, k0=4)
    add_strassen(0, 2, 4, 35)

    return {"U": U, "V": V, "W": W}
