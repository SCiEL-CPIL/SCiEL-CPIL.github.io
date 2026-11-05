import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 48
    rank = 48
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (10, 48)
    # V: (m*p, rank) -> (30, 48)
    # W: (p*n, rank) -> (12, 48)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 Decomposition
    # U_s: (4, 7), V_s: (4, 7), W_s: (4, 7)
    # Indices: i,j in 0..1. k in 0..1. c in 0..6.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U[i,j,c]
    # c=0: A00+A11 -> U[0,0,0]=1, U[1,1,0]=1
    # c=1: A10+A11 -> U[1,0,1]=1, U[1,1,1]=1
    # c=2: A00      -> U[0,0,2]=1
    # c=3: A11      -> U[1,1,3]=1
    # c=4: A00+A01 -> U[0,0,4]=1, U[0,1,4]=1
    # c=5: A10-A00 -> U[1,0,5]=1, U[0,0,5]=-1
    # c=6: A01-A11 -> U[0,1,6]=1, U[1,1,6]=-1
    
    U_s = [[0]*7 for _ in range(4)]
    U_s[0][0] = 1; U_s[3][0] = 1
    U_s[2][1] = 1; U_s[3][1] = 1
    U_s[0][2] = 1
    U_s[3][3] = 1
    U_s[0][4] = 1; U_s[1][4] = 1
    U_s[2][5] = 1; U_s[0][5] = -1
    U_s[1][6] = 1; U_s[3][6] = -1
    
    # V[j,k,c]
    # c=0: B00+B11 -> V[0,0,0]=1, V[1,1,0]=1
    # c=1: B00      -> V[0,0,1]=1
    # c=2: B10-B11 -> V[1,0,2]=1, V[1,1,2]=-1
    # c=3: B01-B00 -> V[0,1,3]=1, V[0,0,3]=-1
    # c=4: B11      -> V[1,1,4]=1
    # c=5: B00+B01 -> V[0,0,5]=1, V[0,1,5]=1
    # c=6: B10+B11 -> V[1,0,6]=1, V[1,1,6]=1
    
    V_s = [[0]*7 for _ in range(4)]
    V_s[0][0] = 1; V_s[3][0] = 1
    V_s[0][1] = 1
    V_s[2][2] = 1; V_s[3][2] = -1
    V_s[1][3] = 1; V_s[0][3] = -1
    V_s[3][4] = 1
    V_s[0][5] = 1; V_s[1][5] = 1
    V_s[2][6] = 1; V_s[3][6] = 1
    
    # W[k,i,c]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W[0,0,0]=1, W[0,0,3]=1, W[0,0,4]=-1, W[0,0,6]=1
    # W[0,1,2]=1, W[0,1,4]=1
    # W[1,0,1]=1, W[1,0,3]=1
    # W[1,1,0]=1, W[1,1,1]=-1, W[1,1,2]=1, W[1,1,5]=1
    
    W_s = [[0]*7 for _ in range(4)]
    W_s[0][0] = 1; W_s[3][0] = 1; W_s[4][0] = -1; W_s[6][0] = 1
    W_s[2][1] = 1; W_s[4][1] = 1
    W_s[1][2] = 1; W_s[3][2] = 1
    W_s[0][3] = 1; W_s[1][3] = -1; W_s[2][3] = 1; W_s[5][3] = 1
    
    def add_strassen_block(base_c, j_offset, k_offset):
        # j_offset: shift for j index in U and V
        # k_offset: shift for k index in V and W
        # Note: i is not shifted (always 0,1)
        for r in range(3): # 3 blocks of 2x2 in the 6 width
            for c_prime in range(7):
                c = base_c + 7*r + c_prime
                # U
                for i in range(2):
                    for j_prime in range(2):
                        val = U_s[i*2+j_prime][c_prime]
                        if val != 0:
                            j_global = j_offset + j_prime
                            U[i*m + j_global][c] += val
                            
                # V
                for j_prime in range(2):
                    for k_prime in range(2):
                        val = V_s[j_prime*2+k_prime][c_prime]
                        if val != 0:
                            j_global = j_offset + j_prime
                            k_global = k_offset + 2*r + k_prime
                            V[j_global*p + k_global][c] += val
                            
                # W
                for k_prime in range(2):
                    for i in range(2):
                        val = W_s[k_prime*2+i][c_prime]
                        if val != 0:
                            k_global = k_offset + 2*r + k_prime
                            W[k_global*n + i][c] += val

    # Block 1: 2x2x6 (j=0,1; k=0..5)
    add_strassen_block(0, 0, 0)
    
    # Block 2a: 2x2x6 part of 2x3x6 (j=2,3; k=0..5)
    add_strassen_block(21, 2, 0)
    
    # Block 2b: 2x1x6 part of 2x3x6 (j=4; k=0..5)
    # Rank 6
    for k in range(6):
        c = 42 + k
        # U[i, 4, c] = 1
        for i in range(2):
            U[i*m + 4][c] = 1
        # V[4, k, c] = 1
        V[4*p + k][c] = 1
        # W[k, i, c] = 1
        for i in range(2):
            W[k*n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
