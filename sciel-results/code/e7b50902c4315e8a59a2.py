def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2 matrix multiplication (Strassen's algorithm).
    
    U shape: (4, 7) -> corresponds to A indices (i*2+j)
    V shape: (4, 7) -> corresponds to B indices (j*2+k)
    W shape: (4, 7) -> corresponds to C indices (k*2+i)
    
    The decomposition satisfies:
    T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    where T is the matmul tensor.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen's 7 products
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices for A (U):
    # A00: 0
    # A01: 1
    # A10: 2
    # A11: 3
    
    # Indices for B (V):
    # B00: 0
    # B01: 1
    # B10: 2
    # B11: 3
    
    # Indices for C (W):
    # C00 (i=0,k=0): 0
    # C10 (i=1,k=0): 1  (k*2+i = 0*2+1)
    # C01 (i=0,k=1): 2  (k*2+i = 1*2+0)
    # C11 (i=1,k=1): 3  (k*2+i = 1*2+1)
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
