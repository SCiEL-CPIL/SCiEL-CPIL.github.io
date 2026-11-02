import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank 55 construction:
    # 5 blocks of 2x2x2 Strassen (rank 7 each) = 35
    # 1 block of 2x1x10 Schoolbook (rank 20) = 20
    # Total = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2 x 2x10 multiplication
    # A indices: i in {0,1}, j in {0,1} -> idx = i*3 + j
    # B indices: j in {0,1}, k in {0..9} -> idx = j*10 + k
    # C indices: i in {0,1}, k in {0..9} -> idx = k*2 + i
    
    # We split the 2x10 B matrix into 5 blocks of 2x2.
    # Block g covers k in {2g, 2g+1}.
    
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # B indices for this block
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1
        
        # A indices (shared across blocks)
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        
        # C indices for this block
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1
        
        # Strassen terms for 2x2 multiplication
        # m1 = (a00 + a11) * (b00 + b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # m2 = (a10 + a11) * b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        
        # m3 = a00 * (b01 - b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        # m4 = a11 * (b10 - b00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        # m5 = (a00 + a01) * b11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        
        # m6 = (a10 - a00) * (b00 + b01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # m7 = (a01 - a11) * (b10 + b11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1
        
    # Part 2: 2x1 x 1x10 multiplication
    # A indices: i in {0,1}, j=2 -> idx = i*3 + 2
    # B indices: j=2, k in {0..9} -> idx = 2*10 + k
    # C indices: i in {0,1}, k in {0..9} -> idx = k*2 + i
    
    for i in range(n):
        for k in range(p):
            a_idx = i * m + 2
            b_idx = 2 * p + k
            c_idx = k * n + i
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
            
    # col should be 55
    if col != 55:
        # Fallback if logic error
        return {"U": [[0]*60 for _ in range(6)], "V": [[0]*60 for _ in range(30)], "W": [[0]*60 for _ in range(20)]}

    return {"U": U, "V": V, "W": W}
