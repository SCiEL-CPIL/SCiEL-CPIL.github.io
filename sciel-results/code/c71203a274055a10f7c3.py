# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 8
    
    # Get the 49-rank decomposition for 4x4x4
    U_444, V_444, W_444 = _strassen_4x4x4()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Block 1: k in 0..3
    for t in range(49):
        for i in range(n):
            for j in range(m):
                U[i * m + j][t] = U_444[i * m + j][t]
        
        for j in range(m):
            for k in range(4):
                V[j * p + k][t] = V_444[j * 4 + k][t]
            for k in range(4, 8):
                V[j * p + k][t] = 0
        
        for k in range(4):
            for i in range(n):
                W[k * n + i][t] = W_444[k * n + i][t]
        for k in range(4, 8):
            for i in range(n):
                W[k * n + i][t] = 0
                
    # Block 2: k in 4..7
    for t in range(49, 98):
        t_444 = t - 49
        for i in range(n):
            for j in range(m):
                U[i * m + j][t] = U_444[i * m + j][t_444]
        
        for j in range(m):
            for k in range(4):
                V[j * p + k][t] = 0
            for k in range(4, 8):
                V[j * p + k][t] = V_444[j * 4 + (k - 4)][t_444]
        
        for k in range(4):
            for i in range(n):
                W[k * n + i][t] = 0
        for k in range(4, 8):
            for i in range(n):
                W[k * n + i][t] = W_444[(k - 4) * n + i][t_444]

    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # t=0: M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # W[k*2+i]: C00 (0,0)->0, C11 (1,1)->3
    
    # t=1: M2 = (A10+A11)B00 -> C10+=M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # C10 (1,0) -> k=0, i=1 -> idx 1
    
    # t=2: M3 = A00(B01-B11) -> C01+=M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1 # C01 (0,1) -> k=1, i=0 -> idx 2
    
    # t=3: M4 = A11(B10-B00) -> C00+=M4, C10+=M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1 # C00 (0,0)->0, C10 (1,0)->1
    
    # t=4: M5 = (A00+A01)B11 -> C01+=M5, C00-=M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1 # C01 (0,1)->2, C00 (0,0)->0
    
    # t=5: M6 = (A10-A00)(B00+B01) -> C11+=M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11 (1,1) -> k=1, i=1 -> idx 3
    
    # t=6: M7 = (A01-A11)(B10+B11) -> C00+=M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C00 (0,0) -> k=0, i=0 -> idx 0

    return U, V, W

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 decomposition using two levels of Strassen.
    U: 16x49, V: 16x49, W: 16x49.
    """
    U_out, V_out, W_out = _strassen_2x2x2()
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            idx = s * 7 + t
            # Outer coefficients (block level)
            # U_out is 2x2 blocks? No, U_out is 4x7 for 2x2 matrices.
            # We need to map block indices (I,J) to local indices.
            # U_out[I*2+J] is the coefficient for block A[I,J].
            
            for I in range(2):
                for J in range(2):
                    coeff_u = U_out[I*2+J][s]
                    if coeff_u == 0: continue
                    
                    for i_prime in range(2):
                        for j_prime in range(2):
                            i = 2*I + i_prime
                            j = 2*J + j_prime
                            u_idx = i*4 + j
                            # Inner coefficient
                            inner_u = U_out[i_prime*2+j_prime][t]
                            U[u_idx][idx] = coeff_u * inner_u

            for J in range(2):
                for K in range(2):
                    coeff_v = V_out[J*2+K][s]
                    if coeff_v == 0: continue
                    
                    for j_prime in range(2):
                        for k_prime in range(2):
                            j = 2*J + j_prime
                            k = 2*K + k_prime
                            v_idx = j*4 + k
                            inner_v = V_out[j_prime*2+k_prime][t]
                            V[v_idx][idx] = coeff_v * inner_v

            for K in range(2):
                for I in range(2):
                    coeff_w = W_out[K*2+I][s]
                    if coeff_w == 0: continue
                    
                    for k_prime in range(2):
                        for i_prime in range(2):
                            k = 2*K + k_prime
                            i = 2*I + i_prime
                            w_idx = k*4 + i
                            inner_w = W_out[k_prime*2+i_prime][t]
                            W[w_idx][idx] = coeff_w * inner_w

    return U, V, W
# EVOLVE-BLOCK-END
