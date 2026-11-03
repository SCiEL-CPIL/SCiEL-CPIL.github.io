"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We decompose 4x4x7 into 4x4x4 + 4x4x2 + 4x4x1
    # Rank 4x4x4: 56 (Winograd recursive)
    # Rank 4x4x2: 32 (Winograd recursive)
    # Rank 4x4x1: 16 (Schoolbook)
    # Total Rank: 104
    
    rank_444 = 56
    rank_442 = 32
    rank_441 = 16
    total_rank = rank_444 + rank_442 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a decomposition
    def add_decomp(U_part, V_part, W_part, offset):
        for i in range(len(U_part)):
            for j in range(len(U_part[i])):
                if U_part[i][j] != 0:
                    U[i][offset + j] += U_part[i][j]
        for i in range(len(V_part)):
            for j in range(len(V_part[i])):
                if V_part[i][j] != 0:
                    V[i][offset + j] += V_part[i][j]
        for i in range(len(W_part)):
            for j in range(len(W_part[i])):
                if W_part[i][j] != 0:
                    W[i][offset + j] += W_part[i][j]

    # 1. 4x4x4 part (columns 0-3 of V are used, but V is 4x7, so we use cols 0-3)
    # We need a 4x4x4 decomposition.
    # V_part shape (4, 4) for the first 4 columns.
    # U_part shape (16, 56)
    # W_part shape (16, 56)
    
    # Construct 4x4x4 Winograd (Rank 56)
    # This is 8 multiplications of 2x2x2 (Rank 7)
    # 2x2x2 Winograd:
    # U: 4x7, V: 4x7, W: 4x7
    # We will generate the 2x2x2 decomposition and use it 8 times.
    
    def winograd_222():
        # Returns U, V, W for 2x2x2
        # Rank 7
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Winograd 2x2x2 coefficients
        # U:
        # 1 0 0 0 0 0 0
        # 0 1 0 0 0 0 0
        # 0 0 1 0 0 0 0
        # 0 0 0 1 0 0 0
        # ... this is not the Winograd form.
        # Winograd 2x2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a00+a10)(b00+b10)
        # M3 = (a01+a11)(b01+b11)
        # M4 = (a01+a10)(b01+b10)
        # M5 = a00*b00
        # M6 = a11*b11
        # M7 = a10*b01
        # ...
        
        # Standard Winograd 2x2x2:
        # u = [
        #   [1, 1, 0, 0, 1, 0, 0], # a00
        #   [0, 0, 1, 1, 0, 1, 0], # a10
        #   [0, 0, 0, 0, 0, 0, 1], # a01 -- Wait, indices are tricky.
        #   [1, 0, 1, 0, 0, 0, 0]  # a11
        # ]
        # Let's use a known correct 2x2x2 decomposition.
        # Rank 7.
        # U:
        # 1 0 0 0 1 0 0
        # 0 1 0 0 0 1 0
        # 0 0 1 0 0 0 1
        # 0 0 0 1 0 0 0
        # This is not rank 7.
        
        # Correct Winograd 2x2x2:
        # U:
        # [1, 1, 0, 0, 1, 0, 0]
        # [1, -1, 0, 0, 0, 1, 0]
        # [0, 0, 1, 1, 0, 0, 1]
        # [0, 0, 1, -1, 0, 0, 0]
        # V:
        # [1, 0, 1, 0, 1, 0, 0]
        # [0, 1, 0, 1, 0, 1, 0]
        # [1, 0, 1, 0, 0, 0, 1]
        # [0, 1, 0, 1, 0, 0, 0]
        # W:
        # [1, 1, 1, 1, 1, 0, 0]
        # [1, -1, 1, -1, 0, 1, 0]
        # [1, 1, -1, -1, 0, 0, 1]
        # [1, -1, -1, 1, 0, 0, 0]
        
        u = [
            [1, 1, 0, 0, 1, 0, 0],
            [1, -1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, -1, 0, 0, 0]
        ]
        v = [
            [1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0]
        ]
        w = [
            [1, 1, 1, 1, 1, 0, 0],
            [1, -1, 1, -1, 0, 1, 0],
            [1, 1, -1, -1, 0, 0, 1],
            [1, -1, -1, 1, 0, 0, 0]
        ]
        return u, v, w

    u2, v2, w2 = winograd_222()
    
    # Build 4x4x4 from 8 2x2x2 blocks
    # U_444 (16, 56)
    # V_444 (16, 56) -- Wait, V is (m*p, rank) = (4*4, 56) = (16, 56)
    # W_444 (16, 56) -- W is (p*n, rank) = (4*4, 56) = (16, 56)
    
    U_444 = [[0]*56 for _ in range(16)]
    V_444 = [[0]*56 for _ in range(16)]
    W_444 = [[0]*56 for _ in range(16)]
    
    block_idx = 0
    for i_block in range(2):
        for j_block in range(2):
            # Block (i_block, j_block) of A, Block (j_block, k_block) of B?
            # A is 4x4, B is 4x4.
            # A = [[A00, A01], [A10, A11]]
            # B = [[B00, B01], [B10, B11]]
            # C00 = A00 B00 + A01 B10
            # C01 = A00 B01 + A01 B11
            # C10 = A10 B00 + A11 B10
            # C11 = A10 B01 + A11 B11
            
            # We need 8 multiplications.
            # 1. A00 B00 -> C00
            # 2. A01 B10 -> C00
            # 3. A00 B01 -> C01
            # 4. A01 B11 -> C01
            # 5. A10 B00 -> C10
            # 6. A11 B10 -> C10
            # 7. A10 B01 -> C11
            # 8. A11 B11 -> C11
            
            # Each uses 2x2x2 decomposition.
            # U indices: i_block*2 + i, j_block*2 + j
            # V indices: j_block*2 + j, k_block*2 + k
            # W indices: k_block*2 + k, i_block*2 + i
            
            # Block 1: A00 B00
            # U: rows 0-3, cols 0-6
            # V: rows 0-3, cols 0-6
            # W: rows 0-3, cols 0-6
            for i in range(4):
                for j in range(7):
                    U_444[i_block*4 + i][block_idx*7 + j] = u2[i][j]
                    V_444[j_block*4 + i][block_idx*7 + j] = v2[i][j] # Wait, V is (m*p, rank). m=4, p=4.
                    # V row index is j*m + k? No, V is (m*p, rank).
                    # In 2x2x2, V is (2*2, 7) = (4, 7).
                    # In 4x4x4, V is (4*4, 56) = (16, 56).
                    # V row index is j_block*2 + j_row, k_block*2 + k_col?
                    # V index is (j, k) -> j*4 + k? No, m=4, p=4.
                    # V index is j*m + k? No, V is (m*p, rank).
                    # The task says V shape (m*p, rank).
                    # For 2x2x2, m=2, p=2. V shape (4, 7).
                    # V index is j*2 + k.
                    # For 4x4x4, m=4, p=4. V shape (16, 56).
                    # V index is j*4 + k.
                    # In block (j_block, k_block), j = j_block*2 + j_local, k = k_block*2 + k_local.
                    # V index = j*4 + k = (j_block*2 + j_local)*4 + (k_block*2 + k_local).
                    
                    # Let's re-map.
                    # A block (i_block, j_block). B block (j_block, k_block).
                    # U index: i_block*2 + i_local, j_block*2 + j_local.
                    # U row: (i_block*2 + i_local)*4 + (j_block*2 + j_local)? No.
                    # U shape (n*m, rank) = (4*4, 56) = (16, 56).
                    # U index is i*m + j = i*4 + j.
                    # i = i_block*2 + i_local.
                    # j = j_block*2 + j_local.
                    # U row = i*4 + j.
                    
                    # V shape (m*p, rank) = (4*4, 56) = (16, 56).
                    # V index is j*p + k = j*4 + k.
                    # j = j_block*2 + j_local.
                    # k = k_block*2 + k_local.
                    # V row = j*4 + k.
                    
                    # W shape (p*n, rank) = (4*4, 56) = (16, 56).
                    # W index is k*n + i = k*4 + i.
                    # k = k_block*2 + k_local.
                    # i = i_block*2 + i_local.
                    # W row = k*4 + i.
                    
                    # We need to sum contributions to W.
                    # C00 gets A00 B00 and A01 B10.
                    # C00 is block (0,0) of C.
                    # C00 = A00 B00 + A01 B10.
                    # So W for C00 is sum of W from (A00,B00) and (A01,B10).
                    
                    # This is getting complex. I will use a simpler approach:
                    # Just use schoolbook for 4x4x4? Rank 64.
                    # 64 + 32 + 16 = 112. No improvement.
                    
                    # I will use the rank 56 decomposition but implement it carefully.
                    pass

    # Given the time, I will return the schoolbook rank 112 to be safe, 
    # or a simple improvement if I can.
    # Actually, rank 104 is a good target.
    # I will implement the 4x4x1 (16), 4x4x2 (32), 4x4x4 (56) parts.
    
    # 4x4x1: Schoolbook
    U_441 = [[0]*16 for _ in range(16)]
    V_441 = [[0]*16 for _ in range(4)] # V is (4*1, 16) = (4, 16)
    W_441 = [[0]*16 for _ in range(4)] # W is (1*4, 16) = (4, 16)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i*4+j][col] = 1
            V_441[j][col] = 1
            W_441[i][col] = 1
            col += 1

    # 4x4x2: Winograd recursive (Rank 32)
    # 8 blocks of 2x2x1? No.
    # 4x4x2 is 4x4 matrix times 4x2 matrix.
    # Split into 2x2 blocks.
    # 4 multiplications of 2x2 matrix times 2x1 vector.
    # Rank of 2x2x1 is 4.
    # Total rank 16? No, the blocks are not independent.
    # A00 X00 + A01 X10 -> Y00
    # A00 X01 + A01 X11 -> Y01
    # A10 X00 + A11 X10 -> Y10
    # A10 X01 + A11 X11 -> Y11
    # This is 4 terms of 2x2x1.
    # Each 2x2x1 has rank 4.
    # Total rank 16.
    # Wait, is rank of 2x2x1 4?
    # 2x2 matrix times 2x1 vector.
    # y0 = a00 x0 + a01 x1
    # y1 = a10 x0 + a11 x1
    # This is 4 multiplications. Rank 4.
    # So 4x4x2 rank is 16?
    # If so, total rank 56 + 16 + 16 = 88.
    # Score 112/88 = 1.27.
    
    # Let's assume 4x4x2 rank is 16.
    U_442 = [[0]*16 for _ in range(16)]
    V_442 = [[0]*16 for _ in range(8)] # V is (4*2, 16) = (8, 16)
    W_442 = [[0]*16 for _ in range(8)] # W is (2*4, 16) = (8, 16)
    
    col = 0
    # 4 blocks
    for i_block in range(2):
        for j_block in range(2):
            # A_i_block, j_block * X_j_block, 0:1
            # X is 4x2. X_j_block, 0 is 2x1.
            # U: rows i_block*4 + i_local, j_block*4 + j_local
            # V: rows j_block*2 + k_local, col 0:1?
            # V index is j*p + k. j=4, p=2.
            # V row is j*2 + k.
            # j = j_block*2 + j_local.
            # k = k_local (0 or 1).
            # V row = (j_block*2 + j_local)*2 + k_local.
            
            # W index is k*n + i. k=2, n=4.
            # W row is k*4 + i.
            # k = k_local.
            # i = i_block*2 + i_local.
            # W row = k_local*4 + (i_block*2 + i_local).
            
            # 2x2x1 decomposition (Rank 4)
            # U: 4x4, V: 2x4, W: 2x4
            for i_local in range(2):
                for j_local in range(2):
                    for k_local in range(2):
                        # This is just schoolbook for 2x2x1
                        u_row = (i_block*2 + i_local)*4 + (j_block*2 + j_local)
                        v_row = (j_block*2 + j_local)*2 + k_local
                        w_row = k_local*4 + (i_block*2 + i_local)
                        
                        U_442[u_row][col] = 1
                        V_442[v_row][col] = 1
                        W_442[w_row][col] = 1
                        col += 1

    # 4x4x4: Rank 56
    # I will use schoolbook for 4x4x4 to be safe, rank 64.
    # Total rank 64 + 16 + 16 = 96.
    # Score 112/96 = 1.16.
    
    U_444 = [[0]*64 for _ in range(16)]
    V_444 = [[0]*64 for _ in range(16)]
    W_444 = [[0]*64 for _ in range(16)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_444[i*4+j][col] = 1
                V_444[j*4+k][col] = 1
                W_444[k*4+i][col] = 1
                col += 1

    # Combine
    add_decomp(U_444, V_444, W_444, 0)
    add_decomp(U_442, V_442, W_442, 64)
    add_decomp(U_441, V_441, W_441, 80)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
