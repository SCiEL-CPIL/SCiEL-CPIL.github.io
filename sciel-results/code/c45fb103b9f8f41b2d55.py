import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44: 4 blocks * 7 (Strassen) + 2 * 8 (Outer Product)
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks for A[0:2, 0:2] * B[0:2, 0:8]
    # B is split into 4 blocks of 2x2
    for c in range(4):
        K_base = c * 2
        
        # Define local indices for A (0,1) and B rows (0,1)
        # U index: i * 3 + j
        # V index: j * 8 + (K_base + k)
        # W index: (K_base + k) * 2 + i
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        # U: A00(0,0), A11(1,1)
        # V: B00(0,K_base), B11(1,K_base+1)
        # W: C00(K_base,0), C11(K_base+1,1)
        U[0*3 + 0][col] = 1
        U[1*3 + 1][col] = 1
        V[0*8 + K_base][col] = 1
        V[1*8 + (K_base+1)][col] = 1
        W[K_base*2 + 0][col] = 1   # C00 coeff +1
        W[(K_base+1)*2 + 1][col] = 1 # C11 coeff +1
        col += 1
        
        # Term 2: M2 = (A10 + A11)B00
        # U: A10(1,0), A11(1,1)
        # V: B00(0,K_base)
        # W: C10(K_base,1)
        U[1*3 + 0][col] = 1
        U[1*3 + 1][col] = 1
        V[0*8 + K_base][col] = 1
        W[K_base*2 + 1][col] = 1   # C10 coeff +1
        col += 1
        
        # Term 3: M3 = A00(B01 - B11)
        # U: A00(0,0)
        # V: B01(0,K_base+1), B11(1,K_base+1)
        # W: C01(K_base+1,0)
        U[0*3 + 0][col] = 1
        V[0*8 + (K_base+1)][col] = 1
        V[1*8 + (K_base+1)][col] = -1
        W[(K_base+1)*2 + 0][col] = 1 # C01 coeff +1
        col += 1
        
        # Term 4: M4 = A11(B10 - B00)
        # U: A11(1,1)
        # V: B10(1,K_base), B00(0,K_base)
        # W: C10(K_base,1)
        U[1*3 + 1][col] = 1
        V[1*8 + K_base][col] = 1
        V[0*8 + K_base][col] = -1
        W[K_base*2 + 1][col] = 1   # C10 coeff +1
        col += 1
        
        # Term 5: M5 = (A00 + A01)B11
        # U: A00(0,0), A01(0,1)
        # V: B11(1,K_base+1)
        # W: C01(K_base+1,0)
        U[0*3 + 0][col] = 1
        U[0*3 + 1][col] = 1
        V[1*8 + (K_base+1)][col] = 1
        W[(K_base+1)*2 + 0][col] = -1 # C01 coeff -1
        col += 1
        
        # Term 6: M6 = (A10 - A00)(B00 + B01)
        # U: A10(1,0), A00(0,0)
        # V: B00(0,K_base), B01(0,K_base+1)
        # W: C11(K_base+1,1)
        U[1*3 + 0][col] = 1
        U[0*3 + 0][col] = -1
        V[0*8 + K_base][col] = 1
        V[0*8 + (K_base+1)][col] = 1
        W[(K_base+1)*2 + 1][col] = 1 # C11 coeff +1
        col += 1
        
        # Term 7: M7 = (A01 - A11)(B10 + B11)
        # U: A01(0,1), A11(1,1)
        # V: B10(1,K_base), B11(1,K_base+1)
        # W: C11(K_base+1,1)
        U[0*3 + 1][col] = 1
        U[1*3 + 1][col] = -1
        V[1*8 + K_base][col] = 1
        V[1*8 + (K_base+1)][col] = 1
        W[(K_base+1)*2 + 1][col] = 1 # C11 coeff +1
        col += 1

    # Outer product part: A[:, 2] * B[2, :]
    # A is 2x3, col 2. B is 3x8, row 2.
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(8):
            U[i*3 + 2][col] = 1
            V[2*8 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
