import numpy as np
import time

DIMENSIONS = [2, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 140  # Leave 10s margin before 150s kill

def construct():
    global START_TIME
    n, m, p = DIMENSIONS
    
    # Return the known rank-44 solution
    return _rank_44_solution()

def _rank_44_solution():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] -> 2x8 (schoolbook, 16 multiplications)
    for i in range(2):
        for k in range(8):
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A[:,1:3] * B[1:3,:] -> 2x8 using Strassen for 2x2 blocks
    # A2 local indices: a00=1, a01=2, a10=4, a11=5
    # B2 rows: row0 at indices 8-15, row1 at indices 16-23
    # 4 blocks of 2 columns each
    
    for block in range(4):
        b00_idx = 1*8 + block*2
        b01_idx = 1*8 + block*2 + 1
        b10_idx = 2*8 + block*2
        b11_idx = 2*8 + block*2 + 1
        
        c00_idx = (block*2)*2 + 0
        c01_idx = (block*2+1)*2 + 0
        c10_idx = (block*2)*2 + 1
        c11_idx = (block*2+1)*2 + 1
        
        # m1 = (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[b00_idx][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[b00_idx][col] = 1
        W[c10_idx][col] = 1
        W[c11_idx][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[1][col] = 1
        V[b01_idx][col] = 1
        V[b11_idx][col] = -1
        W[c01_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[5][col] = 1
        V[b10_idx][col] = 1
        V[b00_idx][col] = -1
        W[c00_idx][col] = 1
        W[c10_idx][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = -1
        W[c01_idx][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[b00_idx][col] = 1
        V[b01_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[b10_idx][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
