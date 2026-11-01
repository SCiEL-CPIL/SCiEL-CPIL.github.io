import numpy as np

def construct():
    n, m, p = 2, 2, 5
    rank = 18
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices for A (2x2)
    # a = i*m + j
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # Indices for B (2x5)
    # b = j*p + k
    # j=0: 0, 1, 2, 3, 4
    # j=1: 5, 6, 7, 8, 9
    
    # Indices for C (2x5)
    # c = k*n + i
    # k=0: 0, 1
    # k=1: 2, 3
    # k=2: 4, 5
    # k=3: 6, 7
    # k=4: 8, 9
    
    def set_strassen(offset, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1
        r = offset
        U[0, r] = 1; U[3, r] = 1
        V[b00, r] = 1; V[b11, r] = 1
        W[c00, r] = 1; W[c01, r] = -1; W[c11, r] = 1
        
        # M2
        r = offset + 1
        U[2, r] = 1; U[3, r] = 1
        V[b00, r] = 1
        W[c10, r] = -1; W[c11, r] = -1
        
        # M3
        r = offset + 2
        U[0, r] = 1
        V[b01, r] = 1; V[b11, r] = -1
        W[c01, r] = 1; W[c11, r] = 1
        
        # M4
        r = offset + 3
        U[3, r] = 1
        V[b10, r] = 1; V[b00, r] = -1
        W[c00, r] = 1; W[c10, r] = 1; W[c11, r] = -1
        
        # M5
        r = offset + 4
        U[0, r] = 1; U[1, r] = 1
        V[b11, r] = 1
        W[c00, r] = -1; W[c01, r] = 1
        
        # M6
        r = offset + 5
        U[2, r] = 1; U[0, r] = -1
        V[b00, r] = 1; V[b01, r] = 1
        W[c01, r] = 1; W[c10, r] = 1
        
        # M7
        r = offset + 6
        U[1, r] = 1; U[3, r] = -1
        V[b10, r] = 1; V[b11, r] = 1
        W[c00, r] = 1; W[c10, r] = 1

    # Block 1: cols 0, 1
    # b00=0, b01=1, b10=5, b11=6
    # c00=0, c01=1, c10=2, c11=3
    set_strassen(0, 0, 1, 5, 6, 0, 1, 2, 3)
    
    # Block 2: cols 2, 3
    # b00=2, b01=3, b10=7, b11=8
    # c00=4, c01=5, c10=6, c11=7
    set_strassen(7, 2, 3, 7, 8, 4, 5, 6, 7)
    
    # Block 3: col 4 (Schoolbook)
    # b00=4, b10=9
    # c00=8, c10=9
    
    # P14: A00 * B[0,4]
    r = 14
    U[0, r] = 1
    V[4, r] = 1
    W[8, r] = 1
    
    # P15: A01 * B[1,4]
    r = 15
    U[1, r] = 1
    V[9, r] = 1
    W[8, r] = 1
    
    # P16: A10 * B[0,4]
    r = 16
    U[2, r] = 1
    V[4, r] = 1
    W[9, r] = 1
    
    # P17: A11 * B[1,4]
    r = 17
    U[3, r] = 1
    V[9, r] = 1
    W[9, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
