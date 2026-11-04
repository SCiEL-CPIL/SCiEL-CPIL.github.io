import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # p = 5
    # Split B into B1 (4x4) and B2 (4x1)
    
    # Part 1: Strassen for 4x4x4
    # Rank 49
    # We need to generate U1, V1, W1 for 4x4x4
    # U1: 16x49, V1: 16x49, W1: 16x49
    
    # Part 2: Schoolbook for 4x4x1
    # Rank 16
    # U2: 16x16, V2: 4x16 (local), W2: 4x16 (local)
    # Map V2 to global V (indices j*5+4)
    # Map W2 to global W (indices 4*4+i)
    
    # Total rank R = 49 + 16 = 65
    R = 65
    U = [[0.0] * R for _ in range(n * m)] # 16 x 65
    V = [[0.0] * R for _ in range(m * p)] # 20 x 65
    W = [[0.0] * R for _ in range(p * n)] # 20 x 65
    
    # --- Strassen 4x4x4 ---
    # Recursive Strassen
    # We can hardcode the 49 entries or generate them.
    # Generating is safer.
    
    def strassen_2x2():
        # Returns U, V, W for 2x2x2
        # n=2, m=2, p=2
        # Rank 7
        U_s = [[0.0]*7 for _ in range(4)]
        V_s = [[0.0]*7 for _ in range(4)]
        W_s = [[0.0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11: 0, A22: 3
        # B11: 0, B22: 3
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        # C11, C22
        W_s[0][0] = 1; W_s[3][0] = 1 # W index k*n+i. C11: k=0,i=0 -> 0. C22: k=1,i=1 -> 1*2+1=3.
        
        # M2 = (A21+A22)B11
        # A21: 2, A22: 3
        # B11: 0
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        # C21: k=1, i=0 -> 1*2+0=2
        W_s[2][1] = 1
        
        # M3 = A11(B12-B22)
        # A11: 0
        # B12: 1, B22: 3
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        # C12: k=0, i=1 -> 0*2+1=1
        W_s[1][2] = 1
        
        # M4 = A22(B21-B11)
        # A22: 3
        # B21: 2, B11: 0
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        # C11, C21
        W_s[0][3] = 1; W_s[2][3] = 1
        
        # M5 = (A11+A12)B22
        # A11: 0, A12: 1
        # B22: 3
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        # C11, C12
        W_s[0][4] = -1; W_s[1][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21: 2, A11: 0
        # B11: 0, B12: 1
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        # C22
        W_s[3][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12: 1, A22: 3
        # B21: 2, B22: 3
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        # C11, C22
        W_s[0][6] = 1; W_s[3][6] = 1
        
        return U_s, V_s, W_s

    def strassen_4x4():
        # n=4, m=4, p=4
        # Rank 49
        U_4 = [[0.0]*49 for _ in range(16)]
        V_4 = [[0.0]*49 for _ in range(16)]
        W_4 = [[0.0]*49 for _ in range(16)]
        
        U_s, V_s, W_s = strassen_2x2()
        
        # 7 blocks
        # Each block is 2x2x2
        # Block b (0..6) uses U_s, V_s, W_s
        # A blocks: A11, A12, A21, A22
        # B blocks: B11, B12, B21, B22
        
        # M1 = (A11+A22)(B11+B22)
        # A11: 0, A22: 9
        # B11: 0, B22: 9
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # A11
                    U_4[i*4+j][c] += U_s[i*2+j][c]
                    # A22
                    U_4[(i+2)*4+(j+2)][c] += U_s[i*2+j][c]
                    
                    # B11
                    V_4[i*4+j][c] += V_s[i*2+j][c]
                    # B22
                    V_4[(i+2)*4+(j+2)][c] += V_s[i*2+j][c]
                    
                    # C11, C22
                    # W index k*4+i
                    # C11: k=0..1, i=0..1 -> k*4+i
                    # C22: k=2..3, i=2..3 -> k*4+i
                    for k in range(2):
                        W_4[k*4+i][c] += W_s[k*2+i][c]
                        W_4[(k+2)*4+(i+2)][c] += W_s[k*2+i][c]

        # M2 = (A21+A22)B11
        # A21: 4, A22: 9
        # B11: 0
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # A21
                    U_4[(i+2)*4+j][c+7] += U_s[i*2+j][c]
                    # A22
                    U_4[(i+2)*4+(j+2)][c+7] += U_s[i*2+j][c]
                    
                    # B11
                    V_4[i*4+j][c+7] += V_s[i*2+j][c]
                    
                    # C21
                    # k=2..3, i=0..1 -> k*4+i
                    for k in range(2):
                        W_4[(k+2)*4+i][c+7] += W_s[k*2+i][c]

        # M3 = A11(B12-B22)
        # A11: 0
        # B12: 4, B22: 9
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # A11
                    U_4[i*4+j][c+14] += U_s[i*2+j][c]
                    
                    # B12
                    V_4[i*4+(j+2)][c+14] += V_s[i*2+j][c]
                    # B22
                    V_4[(i+2)*4+(j+2)][c+14] -= V_s[i*2+j][c]
                    
                    # C12
                    # k=0..1, i=2..3 -> k*4+i
                    for k in range(2):
                        W_4[k*4+(i+2)][c+14] += W_s[k*2+i][c]

        # M4 = A22(B21-B11)
        # A22: 9
        # B21: 8, B11: 0
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # A22
                    U_4[(i+2)*4+(j+2)][c+21] += U_s[i*2+j][c]
                    
                    # B21
                    V_4[(i+2)*4+j][c+21] += V_s[i*2+j][c]
                    # B11
                    V_4[i*4+j][c+21] -= V_s[i*2+j][c]
                    
                    # C11, C21
                    for k in range(2):
                        W_4[k*4+i][c+21] += W_s[k*2+i][c]
                        W_4[(k+2)*4+i][c+21] += W_s[k*2+i][c]

        # M5 = (A11+A12)B22
        # A11: 0, A12: 4
        # B22: 9
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # A11
                    U_4[i*4+j][c+28] += U_s[i*2+j][c]
                    # A12
                    U_4[i*4+(j+2)][c+28] += U_s[i*2+j][c]
                    
                    # B22
                    V_4[(i+2)*4+(j+2)][c+28] += V_s[i*2+j][c]
                    
                    # C11, C12
                    for k in range(2):
                        W_4[k*4+i][c+28] -= W_s[k*2+i][c]
                        W_4[k*4+(i+2)][c+28] += W_s[k*2+i][c]

        # M6 = (A21-A11)(B11+B12)
        # A21: 4, A11: 0
        # B11: 0, B12: 4
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # A21
                    U_4[(i+2)*4+j][c+35] += U_s[i*2+j][c]
                    # A11
                    U_4[i*4+j][c+35] -= U_s[i*2+j][c]
                    
                    # B11
                    V_4[i*4+j][c+35] += V_s[i*2+j][c]
                    # B12
                    V_4[i*4+(j+2)][c+35] += V_s[i*2+j][c]
                    
                    # C22
                    for k in range(2):
                        W_4[(k+2)*4+(i+2)][c+35] += W_s[k*2+i][c]

        # M7 = (A12-A22)(B21+B22)
        # A12: 4, A22: 9
        # B21: 8, B22: 9
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # A12
                    U_4[i*4+(j+2)][c+42] += U_s[i*2+j][c]
                    # A22
                    U_4[(i+2)*4+(j+2)][c+42] -= U_s[i*2+j][c]
                    
                    # B21
                    V_4[(i+2)*4+j][c+42] += V_s[i*2+j][c]
                    # B22
                    V_4[(i+2)*4+(j+2)][c+42] += V_s[i*2+j][c]
                    
                    # C11, C22
                    for k in range(2):
                        W_4[k*4+i][c+42] += W_s[k*2+i][c]
                        W_4[(k+2)*4+(i+2)][c+42] += W_s[k*2+i][c]

        return U_4, V_4, W_4

    U1, V1, W1 = strassen_4x4()
    
    # Map U1, V1, W1 to U, V, W
    # U1: 16x49. U: 16x65.
    for i in range(16):
        for c in range(49):
            U[i][c] = U1[i][c]
            
    # V1: 16x49. V: 20x65.
    # V1 indices j*4+k. V indices j*5+k.
    for j in range(4):
        for k in range(4):
            v_idx = j*4+k
            global_v_idx = j*5+k
            for c in range(49):
                V[global_v_idx][c] = V1[v_idx][c]
                
    # W1: 16x49. W: 20x65.
    # W1 indices k*4+i. W indices k*4+i.
    for k in range(4):
        for i in range(4):
            w_idx = k*4+i
            global_w_idx = k*4+i
            for c in range(49):
                W[global_w_idx][c] = W1[w_idx][c]

    # --- Schoolbook 4x4x1 ---
    # Rank 16
    # U2: 16x16, V2: 4x16, W2: 4x16
    # U2[i*4+j][c] = 1 if c == i*4+j
    # V2[j][c] = 1 if c == i*4+j -> j is fixed?
    # Schoolbook: c = i*4+j? No.
    # A is 4x4. B is 4x1.
    # C_i0 = sum_j A_ij B_j0.
    # Terms: (i,j).
    # U index i*4+j.
    # V index j*1+0 = j.
    # W index 0*4+i = i.
    
    for i in range(4):
        for j in range(4):
            c = i*4+j
            U[i*4+j][49+c] = 1.0
            V[j*5+4][49+c] = 1.0
            W[4*4+i][49+c] = 1.0

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    print(construct())
