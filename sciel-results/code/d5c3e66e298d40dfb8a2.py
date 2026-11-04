import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 44 decomposition using Strassen for 2x2 blocks
    # Split A into A_left (2x2) and A_right (2x1)
    # C = A_left @ B_left + A_right @ B_right
    # B_left is 2x8 (first 2 rows of B), B_right is 1x8 (last row of B)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # (16, 44)
    
    # Part 1: A_left (2x2) @ B_left (2x8)
    # B_left is split into 4 blocks of 2x2. Each uses Strassen (rank 7).
    # Total: 4 * 7 = 28 terms (indices 0-27)
    
    # Strassen 2x2 @ 2x2 decomposition:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b11)
    # M4 = a11 * (b01 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b10)
    # M7 = (a01 - a11) * (b01 + b11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A_left indices: i in {0,1}, j in {0,1} -> U[i*3+j]
    # B_left indices: j in {0,1}, k in {0..7} -> V[j*8+k]
    # C indices: k in {0..7}, i in {0,1} -> W[k*2+i]
    
    strassen_offsets = [
        # For each of the 4 blocks (k_block = 0,1,2,3), k_offset = k_block * 2
        # M1
        (0, [(0,1), (3,1)], [(0,1), (1,1)], [(0,1), (2,1)]),  # a00,a11 | b00,b11 | C00,C01? No.
    ]
    
    # Let's do it explicitly for each block.
    # Block b corresponds to B columns [2b, 2b+1]
    # A is always the same 2x2 block (cols 0,1 of A)
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        base = b * 7
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00 -> U[0], A11 -> U[3]
        # B00 -> V[0*8+k0], B11 -> V[1*8+k1]
        # C00 -> W[k0*2+0], C11 -> W[k1*2+1]
        U[0][base] = 1
        U[3][base] = 1
        V[0 * 8 + k0][base] = 1
        V[1 * 8 + k1][base] = 1
        W[k0 * 2 + 0][base] = 1
        W[k1 * 2 + 1][base] = 1
        
        # M2 = (A10 + A11) * B00
        # A10 -> U[1], A11 -> U[3]
        # B00 -> V[0*8+k0]
        # C10 -> W[k0*2+1]
        U[1][base+1] = 1
        U[3][base+1] = 1
        V[0 * 8 + k0][base+1] = 1
        W[k0 * 2 + 1][base+1] = 1
        
        # M3 = A00 * (B10 - B11)
        # A00 -> U[0]
        # B10 -> V[1*8+k0], B11 -> V[1*8+k1]
        # C01 -> W[k1*2+0], C10 -> W[k0*2+1]
        U[0][base+2] = 1
        V[1 * 8 + k0][base+2] = 1
        V[1 * 8 + k1][base+2] = -1
        W[k1 * 2 + 0][base+2] = 1
        W[k0 * 2 + 1][base+2] = 1
        
        # M4 = A11 * (B01 - B00)
        # A11 -> U[3]
        # B01 -> V[0*8+k1], B00 -> V[0*8+k0]
        # C00 -> W[k0*2+0], C10 -> W[k0*2+1]
        U[3][base+3] = 1
        V[0 * 8 + k1][base+3] = 1
        V[0 * 8 + k0][base+3] = -1
        W[k0 * 2 + 0][base+3] = -1
        W[k0 * 2 + 1][base+3] = 1
        
        # M5 = (A00 + A01) * B11
        # A00 -> U[0], A01 -> U[1]
        # B11 -> V[1*8+k1]
        # C00 -> W[k0*2+0], C01 -> W[k1*2+0]
        U[0][base+4] = 1
        U[1][base+4] = 1
        V[1 * 8 + k1][base+4] = 1
        W[k0 * 2 + 0][base+4] = 1
        W[k1 * 2 + 0][base+4] = 1
        
        # M6 = (A10 - A00) * (B00 + B10)
        # A10 -> U[1], A00 -> U[0]
        # B00 -> V[0*8+k0], B10 -> V[1*8+k0]
        # C11 -> W[k1*2+1]
        U[1][base+5] = 1
        U[0][base+5] = -1
        V[0 * 8 + k0][base+5] = 1
        V[1 * 8 + k0][base+5] = 1
        W[k1 * 2 + 1][base+5] = 1
        
        # M7 = (A01 - A11) * (B01 + B11)
        # A01 -> U[1], A11 -> U[3]
        # B01 -> V[0*8+k1], B11 -> V[1*8+k1]
        # C00 -> W[k0*2+0], C11 -> W[k1*2+1]
        U[1][base+6] = 1
        U[3][base+6] = -1
        V[0 * 8 + k1][base+6] = 1
        V[1 * 8 + k1][base+6] = 1
        W[k0 * 2 + 0][base+6] = 1
        W[k1 * 2 + 1][base+6] = 1
        
    # Part 2: A_right (2x1) @ B_right (1x8)
    # A_right is col 2 of A: A[0,2] -> U[2], A[1,2] -> U[5]
    # B_right is row 2 of B: B[2,k] -> V[2*8+k]
    # C[i,k] += A[i,2] * B[2,k]
    # This requires 2*8 = 16 terms (indices 28-43)
    
    for i in range(2):
        for k in range(8):
            idx = 28 + i * 8 + k
            U[i * 3 + 2][idx] = 1
            V[2 * 8 + k][idx] = 1
            W[k * 2 + i][idx] = 1
            
    # Verification (optional, for debugging)
    # U_arr = np.array(U)
    # V_arr = np.array(V)
    # W_arr = np.array(W)
    # for i in range(n):
    #     for a in range(m):
    #         for k in range(p):
    #             val = sum(U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r] for r in range(rank))
    #             if val != 1:
    #                 print(f"Error at {i},{a},{k}: {val}")
    #                 return {"U": U, "V": "ERROR", "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
