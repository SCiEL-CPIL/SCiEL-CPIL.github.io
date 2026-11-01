import numpy as np

def construct():
    n, m, p = 2, 2, 11
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks for k=0..9
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1: (A00+A11)(B0k0+B1k1)
        # U: A00, A11
        # V: B0k0, B1k1
        # W: C0k0, C0k1, C1k0, C1k1
        U[0][col] = 1 # A00
        U[3][col] = 1 # A11
        V[0*p + k0][col] = 1 # B0k0
        V[1*p + k1][col] = 1 # B1k1
        W[k0*n + 0][col] = 1 # C0k0
        W[k1*n + 0][col] = -1 # C0k1? No, M1 is in C0k0 and C1k1.
        # C0k0 += M1
        # C1k1 += M1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M2: (A10+A11)B0k0
        U[2][col] = 1 # A10
        U[3][col] = 1 # A11
        V[0*p + k0][col] = 1 # B0k0
        W[k0*n + 1][col] = 1 # C1k0 += M2
        col += 1
        
        # M3: A00(B0k1-B1k1)
        U[0][col] = 1 # A00
        V[0*p + k1][col] = 1 # B0k1
        V[1*p + k1][col] = -1 # B1k1
        W[k1*n + 0][col] = 1 # C0k1 += M3
        W[k1*n + 1][col] = 1 # C1k1 += M3
        col += 1
        
        # M4: A11(B1k0-B0k0)
        U[3][col] = 1 # A11
        V[1*p + k0][col] = 1 # B1k0
        V[0*p + k0][col] = -1 # B0k0
        W[k0*n + 0][col] = 1 # C0k0 += M4
        W[k0*n + 1][col] = 1 # C1k0 += M4
        col += 1
        
        # M5: (A00+A01)B1k1
        U[0][col] = 1 # A00
        U[1][col] = 1 # A01
        V[1*p + k1][col] = 1 # B1k1
        W[k0*n + 0][col] = -1 # C0k0 -= M5
        W[k1*n + 0][col] = 1 # C0k1 += M5
        col += 1
        
        # M6: (A10-A00)(B0k0+B0k1)
        U[2][col] = 1 # A10
        U[0][col] = -1 # A00
        V[0*p + k0][col] = 1 # B0k0
        V[0*p + k1][col] = 1 # B0k1
        W[k1*n + 1][col] = 1 # C1k1 += M6
        col += 1
        
        # M7: (A01-A11)(B1k0+B1k1)
        U[1][col] = 1 # A01
        U[3][col] = -1 # A11
        V[1*p + k0][col] = 1 # B1k0
        V[1*p + k1][col] = 1 # B1k1
        W[k0*n + 0][col] = 1 # C0k0 += M7
        col += 1

    # Schoolbook for k=10
    k = 10
    # M1: A00 B0k
    U[0][col] = 1
    V[0*p + k][col] = 1
    W[k*n + 0][col] = 1
    col += 1
    
    # M2: A01 B1k
    U[1][col] = 1
    V[1*p + k][col] = 1
    W[k*n + 0][col] = 1
    col += 1
    
    # M3: A10 B0k
    U[2][col] = 1
    V[0*p + k][col] = 1
    W[k*n + 1][col] = 1
    col += 1
    
    # M4: A11 B1k
    U[3][col] = 1
    V[1*p + k][col] = 1
    W[k*n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
