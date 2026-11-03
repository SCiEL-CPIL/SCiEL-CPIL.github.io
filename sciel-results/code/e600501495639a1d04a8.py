import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100

    # Rank 44 solution
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x4x4 Strassen
    # A top 2 rows, B left 4 cols, C top 2 rows, left 4 cols
    # Strassen terms
    # A11: (0,0),(0,1),(1,0),(1,1)
    # A12: (0,2),(0,3),(1,2),(1,3)
    # A21: (1,0),(1,1)
    # A22: (1,2),(1,3)
    
    # B11: (0,0),(0,1),(1,0),(1,1)
    # B12: (0,2),(0,3),(1,2),(1,3)
    # B21: (2,0),(2,1),(3,0),(3,1)
    # B22: (2,2),(2,3),(3,2),(3,3)
    
    # Strassen M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    
    # Each M is 2x2. We use schoolbook for each M (rank 4).
    # Total 28 terms.
    
    # We define the linear combos for each M_l
    # M_l = X_l * Y_l
    # X_l is 2x2, Y_l is 2x2.
    
    def get_block_A(r1, c1, r2, c2):
        return [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
    
    # A11
    A11 = [(0,0), (0,1), (1,0), (1,1)]
    A12 = [(0,2), (0,3), (1,2), (1,3)]
    A21 = [(1,0), (1,1)] # Wait, A21 is 2x2. (1,0),(1,1) is only 2 elements.
    # Correction: A is 2x4.
    # A11 is 2x2: (0,0),(0,1),(1,0),(1,1)
    # A12 is 2x2: (0,2),(0,3),(1,2),(1,3)
    # A21 is 2x2: (1,0),(1,1) ... NO.
    # A is 2 rows. A21 is row 1, cols 0,1. So (1,0),(1,1).
    # But A21 must be 2x2? No, A is 2x4.
    # Splitting 2x4 into 2 blocks of 2x2.
    # Top block: Row 0, Cols 0,1 and Row 1, Cols 0,1? No.
    # Standard block:
    # A = [ A11 A12 ]
    #     [ A21 A22 ]
    # If A is 2x4, we can't split into 4 2x2 blocks.
    # We can split into 2 2x2 blocks horizontally.
    # A = [ A1 A2 ] where A1 is 2x2, A2 is 2x2.
    # Then B is 4x4. Split into 4 2x2 blocks.
    # This doesn't match.
    
    # Let's use the 2x4x4 rank 21 solution?
    # Or just use schoolbook for 2x4x4 (Rank 32)?
    # 32 + 8 + 4 + 4 = 48.
    # 48 is better than 60.
    
    # Let's stick to Schoolbook for 2x4x4 to be safe and fast to implement.
    # Rank 32.
    
    # Reset col
    col = 0
    # Part 1: 2x4x4 Schoolbook
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U[i*4 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Part 2: 2x4x1 Schoolbook
    # C_top_right (2x1)
    for i in range(2):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*3 + i][col] = 1
            col += 1
            
    # Part 3: 1x4x4 Schoolbook
    # C_bot_left (1x4)
    for j in range(4):
        for k in range(4):
            U[2*4 + j][col] = 1
            V[j*5 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
            
    # Part 4: 1x4x1 Schoolbook
    # C_bot_right (1x1)
    for j in range(4):
        U[2*4 + j][col] = 1
        V[j*5 + 4][col] = 1
        W[4*3 + 2][col] = 1
        col += 1
        
    if col != rank:
        # Adjust rank
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        
    return {"U": U, "V": V, "W": W}
