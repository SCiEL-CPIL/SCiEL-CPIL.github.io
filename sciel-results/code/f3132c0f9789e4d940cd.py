import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition factors
    # U: 4x7, V: 4x7, W: 4x7
    # Indices 0..3 for 2x2 matrices (row-major: 00, 01, 10, 11)
    
    U_strassen = np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [0, 1, 0, -1]
    ], dtype=float).T  # Shape (4, 7) -> rows are local indices, cols are ranks? 
    # Wait, standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Let's define vectors for A (X) and B (Y) for each of the 7 multiplications.
    # X is 2x2, Y is 2x2.
    # X = [[x0, x1], [x2, x3]]
    # Y = [[y0, y1], [y2, y3]]
    
    # M1: (x0+x3)(y0+y3) -> X: [1,0,0,1], Y: [1,0,0,1]
    # M2: (x2+x3)y0     -> X: [0,0,1,1], Y: [1,0,0,0]
    # M3: x0(y1-y3)     -> X: [1,0,0,0], Y: [0,1,0,-1]
    # M4: x3(y2-y0)     -> X: [0,0,0,1], Y: [0,-1,1,0]
    # M5: (x0+x1)y3     -> X: [1,1,0,0], Y: [0,0,0,1]
    # M6: (x2-x0)(y0+y1)-> X: [-1,0,1,0], Y: [1,1,0,0]
    # M7: (x1-x3)(y2+y3)-> X: [0,1,0,-1], Y: [0,0,1,1]
    
    U_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float).T # Shape (4, 7)
    
    V_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float).T # Shape (4, 7)
    
    # Output Z = X*Y
    # Z00 = M1 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    W_s = np.array([
        [1, -1, 0, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, -1, 1, 1, 1]
    ], dtype=float).T # Shape (4, 7)
    
    # Dimensions for 2x4x4
    # A: 2x4, B: 4x4, C: 2x4
    # Rank 28
    rank = 28
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Mapping indices
    # A is 2x4. Split into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    # A_L indices in global U: 
    # 0,0 -> 0
    # 0,1 -> 1
    # 1,0 -> 4
    # 1,1 -> 5
    # Map local [0,1,2,3] (00,01,10,11) to global [0,1,4,5]
    map_A_L = np.array([0, 1, 4, 5])
    map_A_R = np.array([2, 3, 6, 7])
    
    # B is 4x4. Split into B_TL, B_TR, B_BL, B_BR (2x2 blocks)
    # B_TL (rows 0-1, cols 0-1):
    # 0,0 -> 0
    # 0,1 -> 1
    # 1,0 -> 4
    # 1,1 -> 5
    map_B_TL = np.array([0, 1, 4, 5])
    
    # B_TR (rows 0-1, cols 2-3):
    # 0,2 -> 2
    # 0,3 -> 3
    # 1,2 -> 6
    # 1,3 -> 7
    map_B_TR = np.array([2, 3, 6, 7])
    
    # B_BL (rows 2-3, cols 0-1):
    # 2,0 -> 8
    # 2,1 -> 9
    # 3,0 -> 12
    # 3,1 -> 13
    map_B_BL = np.array([8, 9, 12, 13])
    
    # B_BR (rows 2-3, cols 2-3):
    # 2,2 -> 10
    # 2,3 -> 11
    # 3,2 -> 14
    # 3,3 -> 15
    map_B_BR = np.array([10, 11, 14, 15])
    
    # C is 2x4. Split into C_L (2x2, cols 0-1) and C_R (2x2, cols 2-3)
    # W index is k*n + i (col-major for C)
    # C_L (cols 0-1):
    # Z00 (C[0,0]) -> k=0, i=0 -> 0*2+0 = 0
    # Z01 (C[0,1]) -> k=1, i=0 -> 1*2+0 = 2
    # Z10 (C[1,0]) -> k=0, i=1 -> 0*2+1 = 1
    # Z11 (C[1,1]) -> k=1, i=1 -> 1*2+1 = 3
    map_C_L = np.array([0, 2, 1, 3])
    
    # C_R (cols 2-3):
    # Z00 (C[0,2]) -> k=2, i=0 -> 2*2+0 = 4
    # Z01 (C[0,3]) -> k=3, i=0 -> 3*2+0 = 6
    # Z10 (C[1,2]) -> k=2, i=1 -> 2*2+1 = 5
    # Z11 (C[1,3]) -> k=3, i=1 -> 3*2+1 = 7
    map_C_R = np.array([4, 6, 5, 7])
    
    blocks = [
        (map_A_L, map_B_TL, map_C_L),
        (map_A_R, map_B_BL, map_C_L),
        (map_A_L, map_B_TR, map_C_R),
        (map_A_R, map_B_BR, map_C_R)
    ]
    
    col_offset = 0
    for map_A, map_B, map_C in blocks:
        # Add Strassen factors for this block
        # U[global_A_indices, col_offset:col_offset+7] += U_s[global_A_indices] ...
        # Easier to loop or use advanced indexing
        
        # U_s is (4, 7). map_A is (4,).
        # We want to add U_s to U[map_A, col_offset:col_offset+7]
        # U[map_A, col_offset:col_offset+7] += U_s
        
        U[map_A, col_offset:col_offset+7] += U_s
        V[map_B, col_offset:col_offset+7] += V_s
        W[map_C, col_offset:col_offset+7] += W_s
        
        col_offset += 7
        
    # Convert to lists for output if required, though numpy arrays might be accepted?
    # The prompt says "integer/half-integer entries". Numpy floats are fine if they are exact integers/half-integers.
    # The return interface expects {'U', 'V', 'W'}.
    
    return {"U": U, "V": V, "W": W}
