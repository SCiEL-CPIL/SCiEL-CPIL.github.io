import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(i_start, j_start, k_start, n_b, m_b, p_b, local_U, local_V, local_W):
        # local_U: (n_b * m_b, r_b)
        # local_V: (m_b * p_b, r_b)
        # local_W: (p_b * n_b, r_b)
        r_b = local_U.shape[1]
        for r in range(r_b):
            # Add to U
            for i in range(n_b):
                for j in range(m_b):
                    U[(i_start + i) * m + (j_start + j)][r] += local_U[i * m_b + j][r]
            # Add to V
            for j in range(m_b):
                for k in range(p_b):
                    V[(j_start + j) * p + (k_start + k)][r] += local_V[j * p_b + k][r]
            # Add to W
            # Global W index: k * n + i
            # Local W index: k_loc * n_b + i_loc
            for k in range(p_b):
                for i in range(n_b):
                    W[(k_start + k) * n + (i_start + i)][r] += local_W[k * n_b + i][r]
    
    # Define Strassen 2x2x2
    # M1..M7
    # U (2x2), V (2x2), W (2x2)
    # W index k*n+i. n=2.
    # W[0*2+0] = W00, W[0*2+1] = W10, W[1*2+0] = W01, W[1*2+1] = W11
    # C00 = M1 + M4 - M5 + M7 -> W00
    # C01 = M3 + M5 -> W10 (k=0, i=1) -> W[0*2+1]
    # C10 = M2 + M4 -> W01 (k=1, i=0) -> W[1*2+0]
    # C11 = M1 - M2 + M3 + M6 -> W11 (k=1, i=1) -> W[1*2+1]
    
    U_222 = [[0.0]*7 for _ in range(4)]
    V_222 = [[0.0]*7 for _ in range(4)]
    W_222 = [[0.0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U_222[0][0] = 1; U_222[3][0] = 1
    V_222[0][0] = 1; V_222[3][0] = 1
    W_222[0][0] = 1; W_222[3][0] = 1 # W00, W11
    
    # M2: (A10+A11)B00
    U_222[2][1] = 1; U_222[3][1] = 1
    V_222[0][1] = 1
    W_222[2][1] = 1; W_222[3][1] = -1 # W10, W11
    
    # M3: A00(B01-B11)
    U_222[0][2] = 1
    V_222[1][2] = 1; V_222[3][2] = -1
    W_222[1][2] = 1; W_222[3][2] = 1 # W01, W11
    
    # M4: A11(B10-B00)
    U_222[3][3] = 1
    V_222[2][3] = 1; V_222[0][3] = -1
    W_222[0][3] = 1; W_222[2][3] = 1 # W00, W10
    
    # M5: (A00+A01)B11
    U_222[0][4] = 1; U_222[1][4] = 1
    V_222[3][4] = 1
    W_222[0][4] = -1; W_222[1][4] = 1 # W00, W01
    
    # M6: (A10-A00)(B00+B01)
    U_222[2][5] = 1; U_222[0][5] = -1
    V_222[0][5] = 1; V_222[1][5] = 1
    W_222[3][5] = 1 # W11
    
    # M7: (A01-A11)(B10+B11)
    U_222[1][6] = 1; U_222[3][6] = -1
    V_222[2][6] = 1; V_222[3][6] = 1
    W_222[0][6] = 1 # W00
    
    U_222 = np.array(U_222)
    V_222 = np.array(V_222)
    W_222 = np.array(W_222)
    
    # Define 2x2x1
    # M1: A00 B0
    # M2: A01 B1
    # M3: A10 B0
    # M4: A11 B1
    # C0 = M1 + M2
    # C1 = M3 + M4
    # W index k*n+i. n=2. p=1.
    # W[0*2+0] = W00 (C0)
    # W[0*2+1] = W10 (C1)
    
    U_221 = [[0.0]*4 for _ in range(4)]
    V_221 = [[0.0]*4 for _ in range(2)] # m=2, p=1 -> 2 rows
    W_221 = [[0.0]*4 for _ in range(2)] # p=1, n=2 -> 2 rows
    
    # M1
    U_221[0][0] = 1
    V_221[0][0] = 1
    W_221[0][0] = 1
    
    # M2
    U_221[1][1] = 1
    V_221[1][1] = 1
    W_221[0][1] = 1
    
    # M3
    U_221[2][2] = 1
    V_221[0][2] = 1
    W_221[1][2] = 1
    
    # M4
    U_221[3][3] = 1
    V_221[1][3] = 1
    W_221[1][3] = 1
    
    U_221 = np.array(U_221)
    V_221 = np.array(V_221)
    W_221 = np.array(W_221)
    
    # Apply blocks
    # Block 1: A(0,0..1, 0..1), B(0..1, 0..1), C(0..1, 0..1)
    add_block(0, 0, 0, 2, 2, 2, U_222, V_222, W_222)
    # Block 2: A(0,0..1, 2..3), B(2..3, 0..1), C(0..1, 0..1)
    add_block(0, 2, 0, 2, 2, 2, U_222, V_222, W_222)
    # Block 3: A(0,0..1, 0..1), B(0..1, 2..3), C(0..1, 2..3)
    add_block(0, 0, 2, 2, 2, 2, U_222, V_222, W_222)
    # Block 4: A(0,0..1, 2..3), B(2..3, 2..3), C(0..1, 2..3)
    add_block(0, 2, 2, 2, 2, 2, U_222, V_222, W_222)
    # Block 5: A(0,0..1, 0..1), B(0..1, 4), C(0..1, 4)
    add_block(0, 0, 4, 2, 2, 1, U_221, V_221, W_221)
    # Block 6: A(0,0..1, 2..3), B(2..3, 4), C(0..1, 4)
    add_block(0, 2, 4, 2, 2, 1, U_221, V_221, W_221)
    
    return {"U": U, "V": V, "W": W}
