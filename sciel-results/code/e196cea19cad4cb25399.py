import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # We will use 7 Strassen blocks (2x2 each) for the first 14 columns
    # and 1 Schoolbook block (2x1) for the last column.
    
    # Strassen rank 7 for 2x2
    # Schoolbook rank 4 for 2x1
    
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)] # 4 x rank
    V = [[0] * rank for _ in range(m * p)] # 30 x rank
    W = [[0] * rank for _ in range(p * n)] # 30 x rank
    
    col = 0
    # 7 Strassen blocks
    for block in range(7):
        # Columns 2*block and 2*block+1
        c0 = 2 * block
        c1 = 2 * block + 1
        
        # Strassen terms
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(f+h)
        # M3 = a(e-f)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = c(f-g)
        # M7 = (a-c)(e+g)
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 relative to block
        # W indices: 00, 01, 10, 11 -> 0, 1, 2, 3 relative to block (but W is k*n+i? No, W is p*n. Index is k*n+i. k is col index in B, i is row index in A/C)
        # Wait, W index is k*n+i. k is 0..14. i is 0..1.
        # So for block b, k is 2b, 2b+1.
        
        # Term 1: (a+d)(e+h)
        U[0][col] = 1; U[3][col] = 1
        V[c0*2+0][col] = 1; V[c1*2+0][col] = 1 # e is B[0, c0], h is B[1, c1]? No.
        # B is 2x15. B[row, col].
        # e = B[0, c0], f = B[0, c1], g = B[1, c0], h = B[1, c1].
        # V index is j*p+k? No, V is m*p. Index is j*p+k?
        # The interface says V is (m*p, rank).
        # Usually V index corresponds to B indices.
        # B is m x p. So V index is j*p + k? Or j + k*m?
        # The seed code uses V[j*p+k]. So row j, col k.
        # So e (B[0, c0]) is V[0*p + c0].
        # h (B[1, c1]) is V[1*p + c1].
        
        V[0*p + c0][col] = 1
        V[1*p + c1][col] = 1
        # W index is k*n+i.
        # Output C is n x p.
        # C[0, c0] += M1, C[0, c1] += M1?
        # C = AB.
        # C[0, c0] = a*e + b*g.
        # C[0, c1] = a*f + b*h.
        # C[1, c0] = c*e + d*g.
        # C[1, c1] = c*f + d*h.
        
        # Strassen:
        # C[0, c0] = M1 + M4 - M5 - M7
        # C[0, c1] = M3 + M5
        # C[1, c0] = M2 + M7
        # C[1, c1] = M1 - M2 + M3 + M4
        
        # W indices:
        # C[0, c0] -> k=c0, i=0 -> c0*n + 0
        # C[0, c1] -> k=c1, i=0 -> c1*n + 0
        # C[1, c0] -> k=c0, i=1 -> c0*n + 1
        # C[1, c1] -> k=c1, i=1 -> c1*n + 1
        
        W[c0*n + 0][col] = 1
        W[c1*n + 0][col] = 1
        W[c0*n + 1][col] = 1
        W[c1*n + 1][col] = 1
        col += 1
        
        # Term 2: (c+d)(f+h)
        U[2][col] = 1; U[3][col] = 1
        V[0*p + c1][col] = 1 # f
        V[1*p + c1][col] = 1 # h
        W[c0*n + 1][col] = 1
        W[c1*n + 1][col] = -1
        col += 1
        
        # Term 3: a(e-f)
        U[0][col] = 1
        V[0*p + c0][col] = 1 # e
        V[0*p + c1][col] = -1 # f
        W[c0*n + 0][col] = 1
        W[c1*n + 0][col] = 1
        col += 1
        
        # Term 4: d(g-e)
        U[3][col] = 1
        V[1*p + c0][col] = 1 # g
        V[0*p + c0][col] = -1 # e
        W[c0*n + 0][col] = -1
        W[c1*n + 0][col] = 1
        W[c0*n + 1][col] = 1
        W[c1*n + 1][col] = 1
        col += 1
        
        # Term 5: (a+b)h
        U[0][col] = 1; U[1][col] = 1
        V[1*p + c1][col] = 1 # h
        W[c0*n + 0][col] = -1
        W[c1*n + 0][col] = 1
        col += 1
        
        # Term 6: c(f-g)
        U[2][col] = 1
        V[0*p + c1][col] = 1 # f
        V[1*p + c0][col] = -1 # g
        W[c1*n + 1][col] = 1
        col += 1
        
        # Term 7: (a-c)(e+g)
        U[0][col] = 1; U[2][col] = -1
        V[0*p + c0][col] = 1 # e
        V[1*p + c0][col] = 1 # g
        W[c0*n + 0][col] = -1
        W[c1*n + 0][col] = 1
        W[c0*n + 1][col] = 1
        W[c1*n + 1][col] = 1
        col += 1

    # 1 Schoolbook block for last column c=14
    c = 14
    # au
    U[0][col] = 1
    V[0*p + c][col] = 1
    W[c*n + 0][col] = 1
    col += 1
    # bv
    U[1][col] = 1
    V[1*p + c][col] = 1
    W[c*n + 0][col] = 1
    col += 1
    # cu
    U[2][col] = 1
    V[0*p + c][col] = 1
    W[c*n + 1][col] = 1
    col += 1
    # dv
    U[3][col] = 1
    V[1*p + c][col] = 1
    W[c*n + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
