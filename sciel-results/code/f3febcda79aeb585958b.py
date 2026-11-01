import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # 4 blocks of size 2 in p dimension
    # Each block uses rank 11 (7 Strassen + 4 Schoolbook)
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Schoolbook for 2x1x2
    # S1 = a0 b0
    # S2 = a0 b1
    # S3 = a1 b0
    # S4 = a1 b1
    # C00 += S1, C01 += S2, C10 += S3, C11 += S4
    
    for block in range(4):
        # B columns are 2*block, 2*block+1
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen part (uses A cols 0,1 and B rows 0,1)
        # A indices: i*m + j. m=3.
        # A00: 0*3+0=0, A01: 0*3+1=1, A10: 1*3+0=3, A11: 1*3+1=4
        # B indices: j*p + k. p=8.
        # B00: 0*8+k0, B01: 0*8+k1, B10: 1*8+k0, B11: 1*8+k1
        # C indices: k*n + i. n=2.
        # C00: k0*2+0, C01: k1*2+0, C10: k0*2+1, C11: k1*2+1
        
        # M1
        U[0][col] += 1
        U[4][col] += 1
        V[0*8+k0][col] += 1
        V[0*8+k1][col] += 1
        W[k0*2+0][col] += 1
        W[k1*2+0][col] += -1
        W[k0*2+1][col] += 0
        W[k1*2+1][col] += 1
        col += 1
        
        # M2
        U[3][col] += 1
        U[4][col] += 1
        V[0*8+k0][col] += 1
        W[k0*2+0][col] += 0
        W[k1*2+0][col] += 0
        W[k0*2+1][col] += 1
        W[k1*2+1][col] += 0
        col += 1
        
        # M3
        U[0][col] += 1
        V[0*8+k1][col] += 1
        V[1*8+k1][col] += -1
        W[k0*2+0][col] += 0
        W[k1*2+0][col] += 1
        W[k0*2+1][col] += 0
        W[k1*2+1][col] += 0
        col += 1
        
        # M4
        U[4][col] += 1
        V[1*8+k0][col] += 1
        V[0*8+k0][col] += -1
        W[k0*2+0][col] += 1
        W[k1*2+0][col] += 0
        W[k0*2+1][col] += 1
        W[k1*2+1][col] += 0
        col += 1
        
        # M5
        U[0][col] += 1
        U[1][col] += 1
        V[1*8+k1][col] += 1
        W[k0*2+0][col] += -1
        W[k1*2+0][col] += 1
        W[k0*2+1][col] += 0
        W[k1*2+1][col] += 0
        col += 1
        
        # M6
        U[3][col] += 1
        U[0][col] += -1
        V[0*8+k0][col] += 1
        V[0*8+k1][col] += 1
        W[k0*2+0][col] += 0
        W[k1*2+0][col] += 0
        W[k0*2+1][col] += 0
        W[k1*2+1][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1
        U[4][col] += -1
        V[1*8+k0][col] += 1
        V[1*8+k1][col] += 1
        W[k0*2+0][col] += 1
        W[k1*2+0][col] += 0
        W[k0*2+1][col] += 0
        W[k1*2+1][col] += 0
        col += 1
        
        # Schoolbook part (uses A col 2 and B row 2)
        # A20: 0*3+2=2, A21: 1*3+2=5
        # B20: 2*8+k0, B21: 2*8+k1
        
        # S1: a20 b20 -> C10 (global i=1, k=k0)
        U[2][col] += 1
        V[2*8+k0][col] += 1
        W[k0*2+1][col] += 1
        col += 1
        
        # S2: a20 b21 -> C11 (global i=1, k=k1)
        U[2][col] += 1
        V[2*8+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        
        # S3: a21 b20 -> C10 (global i=1, k=k0) -- Wait, C10 is i=1, k=k0?
        # C is 2x8. C[i, k].
        # C10 in local 2x2 is i=1, j=0. Global i=1, k=k0.
        # C11 in local 2x2 is i=1, j=1. Global i=1, k=k1.
        # C00 in local 2x2 is i=0, j=0. Global i=0, k=k0.
        # C01 in local 2x2 is i=0, j=1. Global i=0, k=k1.
        
        # My Strassen mapping:
        # C00 (local) -> W[k0*2+0] (global i=0, k=k0)
        # C01 (local) -> W[k1*2+0] (global i=0, k=k1)
        # C10 (local) -> W[k0*2+1] (global i=1, k=k0)
        # C11 (local) -> W[k1*2+1] (global i=1, k=k1)
        
        # S1: a20 b20 -> C10 (local) -> W[k0*2+1]
        # S2: a20 b21 -> C11 (local) -> W[k1*2+1]
        # S3: a21 b20 -> C10 (local) -> W[k0*2+1]
        # S4: a21 b21 -> C11 (local) -> W[k1*2+1]
        
        # Wait, I already assigned S1 and S2 above.
        # S1: U[2] (a20), V[2*8+k0] (b20), W[k0*2+1] (C10 local)
        # S2: U[2] (a20), V[2*8+k1] (b21), W[k1*2+1] (C11 local)
        
        # S3: a21 b20 -> C10 (local)
        U[5][col] += 1
        V[2*8+k0][col] += 1
        W[k0*2+1][col] += 1
        col += 1
        
        # S4: a21 b21 -> C11 (local)
        U[5][col] += 1
        V[2*8+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
