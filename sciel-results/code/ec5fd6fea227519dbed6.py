import numpy as np

def solve_2_2_2():
    # Strassen
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A00: U[0], A11: U[3]
    # B00: V[0], B11: V[3]
    # C00: W[0], C11: W[3]
    # C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # A10: U[2], A11: U[3]
    # B00: V[0]
    # C10: W[2], C11: W[3]
    # C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    # A00: U[0]
    # B10: V[2], B11: V[3]
    # C00: W[0], C01: W[1]
    # C00 -= M3, C01 += M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = -1; W[1, 2] = 1
    
    # M4 = A11(B01-B00)
    # A11: U[3]
    # B01: V[1], B00: V[0]
    # C00: W[0], C10: W[2]
    # C00 += M4, C10 += M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    # A00: U[0], A01: U[1]
    # B11: V[3]
    # C00: W[0], C01: W[1]
    # C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A10: U[2], A00: U[0]
    # B00: V[0], B01: V[1]
    # C11: W[3]
    # C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01: U[1], A11: U[3]
    # B10: V[2], B11: V[3]
    # C00: W[0]
    # C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def solve_2_2_1():
    # Schoolbook
    # U: 4x4, V: 2x4, W: 2x4
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C0 = A00 B0 + A10 B1
    # C1 = A01 B0 + A11 B1
    # W index: k*n+i. n=2, k=0. W[0]=C0, W[1]=C1.
    
    # Term A00 B0 -> C0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term A10 B1 -> C0
    U[2, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term A01 B0 -> C1
    U[1, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term A11 B1 -> C1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def solve_2_1_2():
    # Rank 1
    # U: 2x1, V: 2x1, W: 4x1
    U = np.ones((2, 1))
    V = np.ones((2, 1))
    W = np.ones((4, 1))
    return U, V, W

def solve_2_1_1():
    # Rank 1
    # U: 2x1, V: 1x1, W: 2x1
    U = np.ones((2, 1))
    V = np.ones((1, 1))
    W = np.ones((2, 1))
    return U, V, W

def solve_2_1_3():
    # Rank 1
    # U: 2x1, V: 3x1, W: 6x1
    U = np.ones((2, 1))
    V = np.ones((3, 1))
    W = np.ones((6, 1))
    return U, V, W

def solve_2_2_3():
    # Split p=3 into 1, 2
    # Block 1: 2x2x1
    # Block 2: 2x2x2
    U1, V1, W1 = solve_2_2_1()
    U2, V2, W2 = solve_2_2_2()
    
    # Embed
    # U: 4x?
    # V: 2x? (p=1 vs p=2)
    # W: 2x? (p=1 vs p=2)
    
    # U is same shape for both (4xR).
    U = np.hstack([U1, U2])
    
    # V: 
    # Block 1 (p=1): V1 is 2x4. Indices j*1+k. j in 0,1. k=0.
    # Global V for 2x2x3: j*3+k. j in 0,1. k in 0,1,2.
    # Block 1 uses k=0.
    V_global_1 = np.zeros((6, 4))
    for j in range(2):
        for k in range(1):
            V_global_1[j*3+k, :] = V1[j*1+k, :]
            
    # Block 2 (p=2): V2 is 4x7. Indices j*2+k. j in 0,1. k in 0,1.
    # Global V for 2x2x3: j*3+k.
    # Block 2 uses k=1,2. So global k = k_local + 1.
    V_global_2 = np.zeros((6, 7))
    for j in range(2):
        for k in range(2):
            V_global_2[j*3 + (k+1), :] = V2[j*2+k, :]
            
    V = np.hstack([V_global_1, V_global_2])
    
    # W:
    # Block 1 (p=1): W1 is 2x4. Indices k*2+i. k=0.
    # Global W for 2x2x3: k*2+i. k in 0,1,2.
    # Block 1 uses k=0.
    W_global_1 = np.zeros((6, 4))
    for k in range(1):
        for i in range(2):
            W_global_1[k*2+i, :] = W1[k*2+i, :]
            
    # Block 2 (p=2): W2 is 4x7. Indices k*2+i. k in 0,1.
    # Global W for 2x2x3: k*2+i.
    # Block 2 uses k=1,2. So global k = k_local + 1.
    W_global_2 = np.zeros((6, 7))
    for k in range(2):
        for i in range(2):
            W_global_2[(k+1)*2+i, :] = W2[k*2+i, :]
            
    W = np.hstack([W_global_1, W_global_2])
    
    return U, V, W

def solve_2_3_3():
    # Split m=3 into 1, 2
    # Block 1: 2x1x3
    # Block 2: 2x2x3
    U1, V1, W1 = solve_2_1_3()
    U2, V2, W2 = solve_2_2_3()
    
    # U:
    # Block 1 (m=1): U1 is 2x1. Indices i*1+j. i in 0,1. j=0.
    # Global U for 2x3x3: i*3+j. i in 0,1. j in 0,1,2.
    # Block 1 uses j=0.
    U_global_1 = np.zeros((6, 1))
    for i in range(2):
        for j in range(1):
            U_global_1[i*3+j, :] = U1[i*1+j, :]
            
    # Block 2 (m=2): U2 is 4x11. Indices i*2+j. i in 0,1. j in 0,1.
    # Global U for 2x3x3: i*3+j.
    # Block 2 uses j=1,2. So global j = j_local + 1.
    U_global_2 = np.zeros((6, 11))
    for i in range(2):
        for j in range(2):
            U_global_2[i*3 + (j+1), :] = U2[i*2+j, :]
            
    U = np.hstack([U_global_1, U_global_2])
    
    # V:
    # Block 1 (m=1): V1 is 3x1. Indices j*3+k. j=0. k in 0,1,2.
    # Global V for 2x3x3: j*3+k.
    # Block 1 uses j=0.
    V_global_1 = np.zeros((9, 1))
    for j in range(1):
        for k in range(3):
            V_global_1[j*3+k, :] = V1[j*3+k, :]
            
    # Block 2 (m=2): V2 is 6x11. Indices j*3+k. j in 0,1. k in 0,1,2.
    # Global V for 2x3x3: j*3+k.
    # Block 2 uses j=1,2. So global j = j_local + 1.
    V_global_2 = np.zeros((9, 11))
    for j in range(2):
        for k in range(3):
            V_global_2[(j+1)*3+k, :] = V2[j*3+k, :]
            
    V = np.hstack([V_global_1, V_global_2])
    
    # W:
    # W is same shape for both (6xR).
    W = np.hstack([W1, W2])
    
    return U, V, W

def solve_2_3_2():
    # Split m=3 into 1, 2
    # Block 1: 2x1x2
    # Block 2: 2x2x2
    U1, V1, W1 = solve_2_1_2()
    U2, V2, W2 = solve_2_2_2()
    
    # U:
    # Block 1 (m=1): U1 is 2x1.
    # Global U for 2x3x2: i*3+j.
    # Block 1 uses j=0.
    U_global_1 = np.zeros((6, 1))
    for i in range(2):
        U_global_1[i*3, :] = U1[i, :]
        
    # Block 2 (m=2): U2 is 4x7.
    # Global U for 2x3x2: i*3+j.
    # Block 2 uses j=1,2.
    U_global_2 = np.zeros((6, 7))
    for i in range(2):
        for j in range(2):
            U_global_2[i*3 + (j+1), :] = U2[i*2+j, :]
            
    U = np.hstack([U_global_1, U_global_2])
    
    # V:
    # Block 1 (m=1): V1 is 2x1. Indices j*2+k. j=0. k in 0,1.
    # Global V for 2x3x2: j*2+k.
    # Block 1 uses j=0.
    V_global_1 = np.zeros((6, 1))
    for j in range(1):
        for k in range(2):
            V_global_1[j*2+k, :] = V1[j*2+k, :]
            
    # Block 2 (m=2): V2 is 4x7.
    # Global V for 2x3x2: j*2+k.
    # Block 2 uses j=1,2.
    V_global_2 = np.zeros((6, 7))
    for j in range(2):
        for k in range(2):
            V_global_2[(j+1)*2+k, :] = V2[j*2+k, :]
            
    V = np.hstack([V_global_1, V_global_2])
    
    # W:
    # W is same shape for both (4xR).
    W = np.hstack([W1, W2])
    
    return U, V, W

def construct():
    n, m, p = 2, 5, 5
    
    # Split m=5 into 3, 2
    # Split p=5 into 3, 2
    
    # Block 1,1: 2x3x3
    U11, V11, W11 = solve_2_3_3()
    # Block 1,2: 2x3x2
    U12, V12, W12 = solve_2_3_2()
    # Block 2,1: 2x2x3
    U21, V21, W21 = solve_2_2_3()
    # Block 2,2: 2x2x2
    U22, V22, W22 = solve_2_2_2()
    
    # Embed into global 2x5x5
    # U: 10xR
    # V: 25xR
    # W: 10xR
    
    U = np.zeros((10, 0))
    V = np.zeros((25, 0))
    W = np.zeros((10, 0))
    
    # Helper to embed
    def embed_block(U_b, V_b, W_b, i_off, j_off, k_off, n_b, m_b, p_b, n_g, m_g, p_g):
        U_g = np.zeros((n_g * m_g, U_b.shape[1]))
        V_g = np.zeros((m_g * p_g, V_b.shape[1]))
        W_g = np.zeros((p_g * n_g, W_b.shape[1]))
        
        for i in range(n_b):
            for j in range(m_b):
                U_g[(i + i_off) * m_g + (j + j_off), :] = U_b[i * m_b + j, :]
                
        for j in range(m_b):
            for k in range(p_b):
                V_g[(j + j_off) * p_g + (k + k_off), :] = V_b[j * p_b + k, :]
                
        for k in range(p_b):
            for i in range(n_b):
                W_g[(k + k_off) * n_g + (i + i_off), :] = W_b[k * n_b + i, :]
                
        return U_g, V_g, W_g
        
    U11_g, V11_g, W11_g = embed_block(U11, V11, W11, 0, 0, 0, 2, 3, 3, 2, 5, 5)
    U12_g, V12_g, W12_g = embed_block(U12, V12, W12, 0, 0, 3, 2, 3, 2, 2, 5, 5)
    U21_g, V21_g, W21_g = embed_block(U21, V21, W21, 0, 3, 0, 2, 2, 3, 2, 5, 5)
    U22_g, V22_g, W22_g = embed_block(U22, V22, W22, 0, 3, 3, 2, 2, 2, 2, 5, 5)
    
    U = np.hstack([U11_g, U12_g, U21_g, U22_g])
    V = np.hstack([V11_g, V12_g, V21_g, V22_g])
    W = np.hstack([W11_g, W12_g, W21_g, W22_g])
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
