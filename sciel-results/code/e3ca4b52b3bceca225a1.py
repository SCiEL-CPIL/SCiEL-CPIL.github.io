import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2+3, p=5 into 2+3
    # A is 3x5, split into A0 (3x2) and A1 (3x3)
    # B is 5x5, split into B00 (2x2), B01 (2x3), B10 (3x2), B11 (3x3)
    # C is 3x5, split into C0 (3x2) and C1 (3x3)
    
    # C0 = A0 @ B00 + A1 @ B10
    # C1 = A0 @ B01 + A1 @ B11
    
    # Ranks:
    # A0 @ B00: 3x2 by 2x2. Rank 12.
    # A1 @ B10: 3x3 by 3x2. Rank 18.
    # A0 @ B01: 3x2 by 2x3. Rank 18.
    # A1 @ B11: 3x3 by 3x3. Rank 23 (Laderman).
    
    # Total Rank: 12 + 18 + 18 + 23 = 71.
    
    rank = 71
    U = [[0] * rank for _ in range(n * m)] # 15 x 71
    V = [[0] * rank for _ in range(m * p)] # 25 x 71
    W = [[0] * rank for _ in range(p * n)] # 15 x 71
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset):
        nonlocal col
        r = len(V_block[0])
        for i in range(len(U_block)):
            for k in range(r):
                if U_block[i][k] != 0:
                    U[u_offset + i][col + k] += U_block[i][k]
        for i in range(len(V_block)):
            for k in range(r):
                if V_block[i][k] != 0:
                    V[v_offset + i][col + k] += V_block[i][k]
        for i in range(len(W_block)):
            for k in range(r):
                if W_block[i][k] != 0:
                    W[w_offset + i][col + k] += W_block[i][k]
        col += r

    # 1. A0 @ B00 (3x2, 2x2)
    # Schoolbook rank 12
    # U indices: 0..5 (A0: i=0..2, j=0..1 -> i*2+j)
    # V indices: 0..3 (B00: j=0..1, k=0..1 -> j*2+k)
    # W indices: 0..5 (C0: i=0..2, k=0..1 -> k*3+i ? No, W[k*n+i])
    # W offset for C0 is 0.
    
    U0 = [[0]*12 for _ in range(6)]
    V0 = [[0]*12 for _ in range(4)]
    W0 = [[0]*12 for _ in range(6)]
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U0[i*2+j][c] = 1
                V0[j*2+k][c] = 1
                W0[k*3+i][c] = 1 # W index: k*n + i. n=3.
                c += 1
    add_block(U0, V0, W0, 0, 0, 0) # A0 is rows 0-5 of U. B00 is rows 0-3 of V. C0 is rows 0-5 of W.
    
    # 2. A1 @ B10 (3x3, 3x2)
    # A1 is rows 6-14 of U (i=0..2, j=2..4 -> i*3+j? No, A is 3x5. A1 is cols 2-4.
    # U index for A1: i*5 + (2+j).
    # B10 is rows 10-17 of V (j=2..4, k=0..1 -> (2+j)*5 + k? No, B is 5x5. B10 is rows 2-4, cols 0-1.
    # V index: (2+j)*5 + k.
    # C0 is rows 0-5 of W.
    
    U1 = [[0]*18 for _ in range(9)]
    V1 = [[0]*18 for _ in range(6)]
    W1 = [[0]*18 for _ in range(6)]
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U1[i*3+j][c] = 1
                V1[j*2+k][c] = 1
                W1[k*3+i][c] = 1
                c += 1
    # Map to global indices
    # U1 local i*3+j -> Global i*5 + 2 + j
    # V1 local j*2+k -> Global (2+j)*5 + k
    # W1 local k*3+i -> Global k*3 + i (C0 part)
    
    U1_g = [[0]*18 for _ in range(9)]
    V1_g = [[0]*18 for _ in range(6)]
    W1_g = [[0]*18 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U1_g[i*3+j][k*9 + j*3 + i] = 1 # This mapping is wrong.
                
    # Let's just build the full U, V, W directly in the loop for simplicity and correctness.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 1: A0 (3x2) @ B00 (2x2) -> C0 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i*5 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                
    # Block 2: A1 (3x3) @ B10 (3x2) -> C0 (3x2)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*5 + (2+j)][col] = 1
                V[(2+j)*5 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                
    # Block 3: A0 (3x2) @ B01 (2x3) -> C1 (3x3)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[i*5 + j][col] = 1
                V[j*5 + (2+k)][col] = 1
                W[(2+k)*3 + i][col] = 1
                col += 1
                
    # Block 4: A1 (3x3) @ B11 (3x3) -> C1 (3x3)
    # Use Laderman's algorithm here (Rank 23)
    # Laderman's 23 terms
    # A is 3x3, B is 3x3.
    # U indices: i*3+j
    # V indices: j*3+k
    # W indices: k*3+i
    
    # I need the Laderman coefficients.
    # Since I don't have them memorized perfectly, I will use Schoolbook (27) for this block.
    # This gives Rank 75.
    # To beat 75, I MUST use Laderman.
    # I will implement Laderman's algorithm coefficients.
    
    # Laderman's Algorithm
    # M1 = A00 * (B00 + B11 - B22)
    # M2 = (A00 + A11) * (B11 + B22 - B00)
    # M3 = (A00 + A22) * (B22 + B00 - B11)
    # M4 = A11 * (B11 + B22 - B00)
    # M5 = A22 * (B22 + B00 - B11)
    # M6 = (A11 + A22) * (B22 + B00 - B11) -- Wait, this is not symmetric.
    
    # Correct Laderman:
    # 1. A00 * (B00 + B11 - B22)
    # 2. (A00 + A11) * (B11 + B22 - B00)
    # 3. (A00 + A22) * (B22 + B00 - B11)
    # 4. A11 * (B11 + B22 - B00)
    # 5. A22 * (B22 + B00 - B11)
    # 6. (A11 + A22) * (B22 + B00 - B11)
    # 7. (A00 + A11 + A22) * (B00 + B11 + B22)
    # ... This is getting risky.
    
    # I will use Schoolbook for Block 4 to ensure correctness, resulting in Rank 75.
    # This does not improve the score.
    
    # Is there a simpler improvement?
    # No.
    
    # I will return the Rank 75 solution.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*5 + (2+j)][col] = 1
                V[(2+j)*5 + (2+k)][col] = 1
                W[(2+k)*3 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

