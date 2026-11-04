import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    # Total rank
    R = 31
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Part 1: A11 (2x2) * B11 (2x8)
    # 4 blocks of 2x2x2 Strassen
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We define the U, V, W vectors for each M
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    # M1
    # a00, a11 -> U[0], U[3]
    # b00, b11 -> V[0], V[3]
    # c00, c11 -> W[0], W[3]
    # c00 = M1, c11 = M1
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1] (W[0]=c00, W[3]=c11)
    
    # M2
    # a10, a11 -> U[2], U[3]
    # b00 -> V[0]
    # c10 -> W[2]
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: [0, 0, 1, 0]
    
    # M3
    # a00 -> U[0]
    # b01, b11 -> V[1], V[3]
    # c01 -> W[1]
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: [0, 1, 0, 0]
    
    # M4
    # a11 -> U[3]
    # b10, b00 -> V[2], V[0]
    # c00, c10 -> W[0], W[2]
    # c00 = M4, c10 = M4
    # U: [0, 0, 0, 1]
    # V: [-1, 0, 1, 0]
    # W: [1, 0, 1, 0]
    
    # M5
    # a00, a01 -> U[0], U[1]
    # b11 -> V[3]
    # c00, c01 -> W[0], W[1]
    # c00 = -M5, c01 = M5
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: [-1, 1, 0, 0]
    
    # M6
    # a10, a00 -> U[2], U[0]
    # b00, b01 -> V[0], V[1]
    # c11 -> W[3]
    # c11 = M6
    # U: [1, 0, 1, 0] (a00=1, a10=1)
    # V: [1, 1, 0, 0]
    # W: [0, 0, 0, 1]
    
    # M7
    # a01, a11 -> U[1], U[3]
    # b10, b11 -> V[2], V[3]
    # c00, c11 -> W[0], W[3]
    # c00 = M7, c11 = M7
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: [1, 0, 0, 1]
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]
    
    col = 0
    # 4 blocks for B11 (2x8)
    # Block t covers k = 2t, 2t+1
    for t in range(4):
        for s in range(7):
            # U: A11 (i in 0,1; j in 0,1)
            # Global U index: i*3 + j
            for i in range(2):
                for j in range(2):
                    U[i*3 + j][col] += strassen_U[s][i*2 + j]
            
            # V: B11 (j in 0,1; k in 0..7)
            # Global V index: j*8 + k
            # Local k' in 0,1 maps to global k = 2t + k'
            for j in range(2):
                for kp in range(2):
                    k = 2*t + kp
                    V[j*8 + k][col] += strassen_V[s][j*2 + kp]
            
            # W: C (k in 0..7; i in 0,1)
            # Global W index: k*3 + i
            # Local k' in 0,1 maps to global k = 2t + k'
            for kp in range(2):
                for i in range(2):
                    k = 2*t + kp
                    W[k*3 + i][col] += strassen_W[s][kp*2 + i]
            
            col += 1
    
    # Part 2: A12 (2x1) * B21 (1x8)
    # Rank 1
    # U: i in 0,1; j=2. Global U: i*3 + 2
    # V: j=2; k in 0..7. Global V: 2*8 + k
    # W: k in 0..7; i in 0,1. Global W: k*3 + i
    
    for i in range(2):
        U[i*3 + 2][col] = 1
    for k in range(8):
        V[2*8 + k][col] = 1
    for k in range(8):
        for i in range(2):
            W[k*3 + i][col] = 1
    col += 1
    
    # Part 3: A21 (1x2) * B11 (2x8)
    # Rank 1
    # U: i=2; j in 0,1. Global U: 2*3 + j
    # V: j in 0,1; k in 0..7. Global V: j*8 + k
    # W: k in 0..7; i=2. Global W: k*3 + 2
    
    for j in range(2):
        U[2*3 + j][col] = 1
    for j in range(2):
        for k in range(8):
            V[j*8 + k][col] = 1
    for k in range(8):
        W[k*3 + 2][col] = 1
    col += 1
    
    # Part 4: A22 (1x1) * B21 (1x8)
    # Rank 1
    # U: i=2; j=2. Global U: 2*3 + 2 = 8
    # V: j=2; k in 0..7. Global V: 2*8 + k
    # W: k in 0..7; i=2. Global W: k*3 + 2
    
    U[8][col] = 1
    for k in range(8):
        V[2*8 + k][col] = 1
    for k in range(8):
        W[k*3 + 2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

