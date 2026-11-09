import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 28
    V = [[0] * rank for _ in range(m * p)] # 15 x 28
    W = [[0] * rank for _ in range(p * n)] # 10 x 28
    
    # Indices:
    # U: A[i, j] -> 3*i + j
    # V: B[j, k] -> 5*j + k
    # W: C[i, k] -> 2*k + i  (Task says W[k*n+i, r], so index is k*n+i. n=2. So 2*k+i)
    
    # --- Block 1: C[:, 0:2] ---
    # 1. Strassen A_left * B_1_top
    # A_left: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B_1_top: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    
    # M1: (A00+A11)(B00+B11)
    col = 0
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[6][col] += 1
    # C00 = W[0], C11 = W[3]
    W[0][col] += 1; W[3][col] += 1

    # M2: (A10+A11)B00
    col = 1
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    # C10 = W[1], C11 = W[3]
    W[1][col] += 1; W[3][col] -= 1

    # M3: A00(B01-B11)
    col = 2
    U[0][col] += 1
    V[1][col] += 1; V[6][col] -= 1
    # C01 = W[2], C11 = W[3]
    W[2][col] += 1; W[3][col] += 1

    # M4: A11(B10-B00)
    col = 3
    U[4][col] += 1
    V[5][col] += 1; V[0][col] -= 1
    # C10 = W[1], C00 = W[0]
    W[1][col] += 1; W[0][col] += 1

    # M5: (A00+A01)B11
    col = 4
    U[0][col] += 1; U[1][col] += 1
    V[6][col] += 1
    # C01 = W[2], C00 = W[0]
    W[2][col] += 1; W[0][col] -= 1

    # M6: (A10-A00)(B00+B01)
    col = 5
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    # C11 = W[3]
    W[3][col] += 1

    # M7: (A01-A11)(B10+B11)
    col = 6
    U[1][col] += 1; U[4][col] -= 1
    V[5][col] += 1; V[6][col] += 1
    # C00 = W[0]
    W[0][col] += 1

    # 2. Schoolbook A_right * B_1_bottom
    # A_right: A[0,2]=2, A[1,2]=5
    # B_1_bottom: B[2,0]=10, B[2,1]=11
    
    col = 7
    U[2][col] += 1
    V[10][col] += 1
    W[0][col] += 1 # C00

    col = 8
    U[2][col] += 1
    V[11][col] += 1
    W[2][col] += 1 # C01

    col = 9
    U[5][col] += 1
    V[10][col] += 1
    W[1][col] += 1 # C10

    col = 10
    U[5][col] += 1
    V[11][col] += 1
    W[3][col] += 1 # C11

    # --- Block 2: C[:, 2:4] ---
    # 1. Strassen A_left * B_2_top
    # B_2_top: B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
    
    col = 11
    # M1
    U[0][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[8][col] += 1
    W[4][col] += 1; W[7][col] += 1 # C02=W[4], C13=W[7]

    col = 12
    # M2
    U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1
    W[5][col] += 1; W[7][col] -= 1 # C12=W[5], C13=W[7]

    col = 13
    # M3
    U[0][col] += 1
    V[3][col] += 1; V[8][col] -= 1
    W[6][col] += 1; W[7][col] += 1 # C03=W[6], C13=W[7]

    col = 14
    # M4
    U[4][col] += 1
    V[7][col] += 1; V[2][col] -= 1
    W[5][col] += 1; W[4][col] += 1 # C12=W[5], C02=W[4]

    col = 15
    # M5
    U[0][col] += 1; U[1][col] += 1
    V[8][col] += 1
    W[6][col] += 1; W[4][col] -= 1 # C03=W[6], C02=W[4]

    col = 16
    # M6
    U[3][col] += 1; U[0][col] -= 1
    V[2][col] += 1; V[3][col] += 1
    W[7][col] += 1 # C13=W[7]

    col = 17
    # M7
    U[1][col] += 1; U[4][col] -= 1
    V[7][col] += 1; V[8][col] += 1
    W[4][col] += 1 # C02=W[4]

    # 2. Schoolbook A_right * B_2_bottom
    # B_2_bottom: B[2,2]=12, B[2,3]=13
    
    col = 18
    U[2][col] += 1
    V[12][col] += 1
    W[4][col] += 1 # C02

    col = 19
    U[2][col] += 1
    V[13][col] += 1
    W[6][col] += 1 # C03

    col = 20
    U[5][col] += 1
    V[12][col] += 1
    W[5][col] += 1 # C12

    col = 21
    U[5][col] += 1
    V[13][col] += 1
    W[7][col] += 1 # C13

    # --- Block 3: C[:, 4] ---
    # 1. Schoolbook A_left * B_3_top
    # B_3_top: B[0,4]=4, B[1,4]=9
    
    col = 22
    U[0][col] += 1
    V[4][col] += 1
    W[8][col] += 1 # C04

    col = 23
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1 # C04

    col = 24
    U[3][col] += 1
    V[4][col] += 1
    W[9][col] += 1 # C14

    col = 25
    U[4][col] += 1
    V[9][col] += 1
    W[9][col] += 1 # C14

    # 2. Schoolbook A_right * B_3_bottom
    # B_3_bottom: B[2,4]=14
    
    col = 26
    U[2][col] += 1
    V[14][col] += 1
    W[8][col] += 1 # C04

    col = 27
    U[5][col] += 1
    V[14][col] += 1
    W[9][col] += 1 # C14

    return {"U": U, "V": V, "W": W}
