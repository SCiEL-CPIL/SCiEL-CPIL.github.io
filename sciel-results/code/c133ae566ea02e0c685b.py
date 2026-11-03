import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2x8 using 4 Strassen blocks (2x2x2)
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0..7}
    # Output C indices: i in {0,1}, k in {0..7}
    
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # A elements:
        # A00 = U[0*3+0], A01 = U[0*3+1]
        # A10 = U[1*3+0], A11 = U[1*3+1]
        
        # B elements for this block:
        # B0k0 = V[0*8+k0], B0k1 = V[0*8+k1]
        # B1k0 = V[1*8+k0], B1k1 = V[1*8+k1]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0*3 + 0][col] += 1
        U[1*3 + 1][col] += 1
        V[0*8 + k0][col] += 1
        V[1*8 + k1][col] += 1
        # C0k0 += M1, C1k1 += M1
        W[k0*2 + 0][col] += 1
        W[k1*2 + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1*3 + 0][col] += 1
        U[1*3 + 1][col] += 1
        V[0*8 + k0][col] += 1
        # C1k0 += M2, C1k1 -= M2
        W[k0*2 + 1][col] += 1
        W[k1*2 + 1][col] -= 1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0*3 + 0][col] += 1
        V[0*8 + k1][col] += 1
        V[1*8 + k1][col] -= 1
        # C0k1 += M3, C1k1 += M3
        W[k1*2 + 0][col] += 1
        W[k1*2 + 1][col] += 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1*3 + 1][col] += 1
        V[1*8 + k0][col] += 1
        V[0*8 + k0][col] -= 1
        # C0k0 += M4, C1k0 += M4
        W[k0*2 + 0][col] += 1
        W[k0*2 + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0*3 + 0][col] += 1
        U[0*3 + 1][col] += 1
        V[1*8 + k1][col] += 1
        # C0k0 -= M5, C0k1 += M5
        W[k0*2 + 0][col] -= 1
        W[k1*2 + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1*3 + 0][col] += 1
        U[0*3 + 0][col] -= 1
        V[0*8 + k0][col] += 1
        V[0*8 + k1][col] += 1
        # C1k1 += M6
        W[k1*2 + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0*3 + 1][col] += 1
        U[1*3 + 1][col] -= 1
        V[1*8 + k0][col] += 1
        V[1*8 + k1][col] += 1
        # C0k0 += M7
        W[k0*2 + 0][col] += 1
        col += 1
        
    # Part 2: 2x1x8 Schoolbook
    # A indices: i in {0,1}, j=2
    # B indices: j=2, k in {0..7}
    
    for k in range(8):
        # M_k0 = A02 * B2k
        U[0*3 + 2][col] += 1
        V[2*8 + k][col] += 1
        W[k*2 + 0][col] += 1
        col += 1
        
        # M_k1 = A12 * B2k
        U[1*3 + 2][col] += 1
        V[2*8 + k][col] += 1
        W[k*2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
