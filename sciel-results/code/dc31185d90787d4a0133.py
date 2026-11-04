import time
import sys

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use the decomposition T_6 = T_3 (x) T_2
    # Rank = R(3,3,3) * R(2,2,2)
    # R(3,3,3) = 27 (schoolbook)
    # R(2,2,2) = 7 (Strassen)
    # Total Rank = 189
    
    r3 = 27
    r2 = 7
    rank = r3 * r2
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen factors for 2x2
    # Indices: i,j,k in {0,1}
    # U2 row: 2*i + j
    # V2 row: 2*j + k
    # W2 row: 2*k + i
    # Col: 0..6
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2: (a10+a11)b00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # M3: a00(b10-b11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # M4: a11(b01-b00)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5: (a00+a01)b11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6: (a10-a00)(b00+b10)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1
    
    # M7: (a01-a11)(b11+b01)
    U2[1][6] = 1; U2[3][6] = -1
    V2[3][6] = 1; V2[1][6] = 1
    W2[0][6] = 1; W2[3][6] = -1
    
    # Schoolbook 3x3 factors
    # U3 row: 3*i1 + j1, col: 9*i1 + 3*j1 + k1
    # V3 row: 3*j1 + k1, col: 9*i1 + 3*j1 + k1
    # W3 row: 3*k1 + i1, col: 9*i1 + 3*j1 + k1
    
    # We can iterate over the non-zero entries of U3, V3, W3
    # U3[i1,j1,c3] = 1 if c3 corresponds to (i1,j1,k1)
    # Specifically, for a given c3, i1 = c3//9, j1 = (c3%9)//3, k1 = c3%3
    
    for c3 in range(r3):
        i1 = c3 // 9
        j1 = (c3 % 9) // 3
        k1 = c3 % 3
        
        # Non-zero entries in U3 for this c3:
        # Row 3*i1 + j1 has value 1
        
        # Non-zero entries in V3 for this c3:
        # Row 3*j1 + k1 has value 1
        
        # Non-zero entries in W3 for this c3:
        # Row 3*k1 + i1 has value 1
        
        for c2 in range(r2):
            c = c3 * r2 + c2
            
            # U2 non-zeros for c2
            # We need to add U2[i2,j2,c2] to U6[I,J,c]
            # I = 2*i1 + i2, J = 2*j1 + j2
            
            # Instead of iterating all i2,j2, we can just iterate the 4 rows of U2
            for i2 in range(2):
                for j2 in range(2):
                    val_u2 = U2[i2*2 + j2][c2]
                    if val_u2 != 0:
                        I = 2 * i1 + i2
                        J = 2 * j1 + j2
                        U[I*6 + J][c] = val_u2 # U3 entry is 1
                        
            # V2 non-zeros for c2
            # V6[J,K,c] += V3[j1,k1,c3] * V2[j2,k2,c2]
            # V3 entry is 1 at row 3*j1+k1
            for j2 in range(2):
                for k2 in range(2):
                    val_v2 = V2[j2*2 + k2][c2]
                    if val_v2 != 0:
                        J = 2 * j1 + j2
                        K = 2 * k1 + k2
                        V[J*6 + K][c] = val_v2
                        
            # W2 non-zeros for c2
            # W6[K,I,c] += W3[k1,i1,c3] * W2[k2,i2,c2]
            # W3 entry is 1 at row 3*k1+i1
            for k2 in range(2):
                for i2 in range(2):
                    val_w2 = W2[k2*2 + i2][c2]
                    if val_w2 != 0:
                        K = 2 * k1 + k2
                        I = 2 * i1 + i2
                        W[K*6 + I][c] = val_w2

    return {"U": U, "V": V, "W": W}
