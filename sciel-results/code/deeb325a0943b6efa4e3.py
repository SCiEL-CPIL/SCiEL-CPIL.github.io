import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # A is n x m (2x3)
    # B is m x p (3x10)
    # C is n x p (2x10)
    
    # Rank 55 decomposition
    # 1. A_L (2x2) * B_L (2x10) -> 5 blocks of 2x2x2 using Strassen (rank 7 each) -> 35
    # 2. A_R (2x1) * B_R (1x10) -> Outer product (rank 20) -> 20
    # Total rank 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen for 2x2x2
    # A indices: 00, 01, 10, 11 -> flat: 0, 1, 3, 4 (since m=3, A[1,0] is 1*3+0=3, A[1,1] is 1*3+1=4)
    # B indices for block i: 
    # B is 3x10.
    # Block i uses rows 0,1 and cols 2i, 2i+1.
    # B[0, 2i] -> flat 0*10 + 2i = 2i
    # B[0, 2i+1] -> flat 0*10 + 2i+1 = 2i+1
    # B[1, 2i] -> flat 1*10 + 2i = 10+2i
    # B[1, 2i+1] -> flat 1*10 + 2i+1 = 10+2i+1
    
    # C indices for block i:
    # C is 2x10.
    # C[0, 2i] -> flat 0*10 + 2i = 2i
    # C[0, 2i+1] -> flat 0*10 + 2i+1 = 2i+1
    # C[1, 2i] -> flat 1*10 + 2i = 10+2i
    # C[1, 2i+1] -> flat 1*10 + 2i+1 = 10+2i+1
    
    for i in range(5):
        k0 = 2 * i
        k1 = 2 * i + 1
        
        # B flat indices
        b_00 = k0
        b_01 = k1
        b_10 = 10 + k0
        b_11 = 10 + k1
        
        # C flat indices
        c_00 = k0
        c_01 = k1
        c_10 = 10 + k0
        c_11 = 10 + k1
        
        # A flat indices (constant for all blocks)
        a_00 = 0
        a_01 = 1
        a_10 = 3
        a_11 = 4
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00 (+1), C11 (+1)
        U[a_00][col] = 1; U[a_11][col] = 1
        V[b_00][col] = 1; V[b_11][col] = 1
        W[c_00][col] = 1; W[c_11][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        # U: A10, A11
        # V: B00
        # W: C10 (+1), C11 (-1)
        U[a_10][col] = 1; U[a_11][col] = 1
        V[b_00][col] = 1
        W[c_10][col] = 1; W[c_11][col] = -1
        col += 1
        
        # M3 = A00(B01 - B11)
        # U: A00
        # V: B01, B11
        # W: C01 (+1), C11 (+1)
        U[a_00][col] = 1
        V[b_01][col] = 1; V[b_11][col] = -1
        W[c_01][col] = 1; W[c_11][col] = 1
        col += 1
        
        # M4 = A11(B10 - B00)
        # U: A11
        # V: B10, B00
        # W: C00 (+1), C10 (+1)
        U[a_11][col] = 1
        V[b_10][col] = 1; V[b_00][col] = -1
        W[c_00][col] = 1; W[c_10][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        # U: A00, A01
        # V: B11
        # W: C00 (-1), C01 (+1)
        U[a_00][col] = 1; U[a_01][col] = 1
        V[b_11][col] = 1
        W[c_00][col] = -1; W[c_01][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # U: A10, A00
        # V: B00, B01
        # W: C11 (+1)
        U[a_10][col] = 1; U[a_00][col] = -1
        V[b_00][col] = 1; V[b_01][col] = 1
        W[c_11][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # U: A01, A11
        # V: B10, B11
        # W: C00 (+1)
        U[a_01][col] = 1; U[a_11][col] = -1
        V[b_10][col] = 1; V[b_11][col] = 1
        W[c_00][col] = 1
        col += 1

    # Outer product part: A_R (2x1) * B_R (1x10)
    # A_R indices: A[0,2]=2, A[1,2]=5
    # B_R indices: B[2,k] = 20+k for k in 0..9
    # C indices: C[0,k]=k, C[1,k]=10+k
    
    a_02 = 2
    a_12 = 5
    
    for k in range(10):
        b_2k = 20 + k
        c_0k = k
        c_1k = 10 + k
        
        # Term 1: A02 * B2k -> C0k
        U[a_02][col] = 1
        V[b_2k][col] = 1
        W[c_0k][col] = 1
        col += 1
        
        # Term 2: A12 * B2k -> C1k
        U[a_12][col] = 1
        V[b_2k][col] = 1
        W[c_1k][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
