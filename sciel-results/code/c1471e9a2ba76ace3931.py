import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a standard 1x1x1 term
    # U[i*m+j], V[j*p+k], W[k*n+i]
    def add_standard(i, j, k, val=1):
        nonlocal col
        U[i * m + j][col] += val
        V[j * p + k][col] += val
        W[k * n + i][col] += val
        col += 1

    # Helper to add a Strassen 2x2x2 term
    # Indices are relative to the 2x2 block
    # i0, j0, k0 are the top-left indices in the global matrix
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[(i0 + 0) * m + (j0 + 0)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 0) * p + (k0 + 0)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 0) * n + (i0 + 0)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(i0 + 1) * m + (j0 + 0)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 0) * p + (k0 + 0)][col] = 1
        W[(k0 + 0) * n + (i0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = -1
        col += 1
        
        # M3 = A11(B12-B22)
        U[(i0 + 0) * m + (j0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = -1
        W[(k0 + 1) * n + (i0 + 0)][col] = 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 0)][col] = -1
        W[(k0 + 0) * n + (i0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 0)][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[(i0 + 0) * m + (j0 + 0)][col] = 1
        U[(i0 + 0) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 0)][col] = -1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(i0 + 1) * m + (j0 + 0)][col] = 1
        U[(i0 + 0) * m + (j0 + 0)][col] = -1
        V[(j0 + 0) * p + (k0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[(i0 + 0) * m + (j0 + 1)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = -1
        V[(j0 + 1) * p + (k0 + 0)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 0) * n + (i0 + 0)][col] = 1
        col += 1

    # Decomposition of M(3,4,5)
    # Split m=4 into m1=2, m2=2
    # For each m_part in [0, 1]:
    #   Split n=3 into n1=2, n2=1
    #   Split p=5 into p1=2, p2=2, p3=1
    
    for m_part in range(2):
        j_start = m_part * 2
        
        # n1=2, p1=2 -> M(2,2,2) Strassen
        add_strassen(0, j_start, 0)
        
        # n1=2, p2=2 -> M(2,2,2) Strassen
        add_strassen(0, j_start, 2)
        
        # n1=2, p3=1 -> M(2,2,1) Standard
        # A is 2x2, B is 2x1
        # i in 0,1; j in j_start, j_start+1; k in 4
        for i in range(2):
            for j in range(2):
                add_standard(i, j_start + j, 4)
                
        # n2=1, p1=2 -> M(1,2,2) Standard
        # A is 1x2, B is 2x2
        # i in 2; j in j_start, j_start+1; k in 0,1
        for j in range(2):
            for k in range(2):
                add_standard(2, j_start + j, k)
                
        # n2=1, p2=2 -> M(1,2,2) Standard
        # A is 1x2, B is 2x2
        # i in 2; j in j_start, j_start+1; k in 2,3
        for j in range(2):
            for k in range(2):
                add_standard(2, j_start + j, 2 + k)
                
        # n2=1, p3=1 -> M(1,2,1) Standard
        # A is 1x2, B is 2x1
        # i in 2; j in j_start, j_start+1; k in 4
        for j in range(2):
            add_standard(2, j_start + j, 4)

    return {"U": U, "V": V, "W": W}
