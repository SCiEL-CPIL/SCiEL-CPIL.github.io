import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    # Base 2x2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Build 4x4x4 decomposition (Rank 49)
    # n=4, m=4, p=4
    # U4: 16x49, V4: 16x49, W4: 16x49
    rank_444 = 49
    U4 = [[0]*rank_444 for _ in range(16)]
    V4 = [[0]*rank_444 for _ in range(16)]
    W4 = [[0]*rank_444 for _ in range(16)]
    
    # Strassen terms definitions
    # Each term s defines:
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_blocks: list of (block_row, block_col, coeff)
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local 2x2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = [
        { # M1
            'A': [(0, 0, 1), (3, 0, 1)], # A00, A11
            'B': [(0, 0, 1), (3, 0, 1)], # B00, B11
            'C': [(0, 0, 1), (3, 0, 1)]  # C00, C11
        },
        { # M2
            'A': [(2, 0, 1), (3, 0, 1)], # A10, A11
            'B': [(0, 0, 1)],            # B00
            'C': [(2, 0, 1), (3, 0, -1)] # C10, C11 (C10+=M2, C11-=M2)
        },
        { # M3
            'A': [(0, 0, 1)],            # A00
            'B': [(1, 0, 1), (3, 0, -1)],# B01, B11
            'C': [(1, 0, 1), (3, 0, 1)]  # C01, C11
        },
        { # M4
            'A': [(3, 0, 1)],            # A11
            'B': [(2, 0, 1), (0, 0, -1)],# B10, B00
            'C': [(0, 0, 1), (2, 0, 1)]  # C00, C10
        },
        { # M5
            'A': [(0, 0, 1), (1, 0, 1)], # A00, A01
            'B': [(3, 0, 1)],            # B11
            'C': [(0, 0, -1), (1, 0, 1)] # C00, C01
        },
        { # M6
            'A': [(2, 0, 1), (0, 0, -1)],# A10, A00
            'B': [(0, 0, 1), (1, 0, 1)], # B00, B01
            'C': [(3, 0, 1)]             # C11
        },
        { # M7
            'A': [(1, 0, 1), (3, 0, -1)],# A01, A11
            'B': [(2, 0, 1), (3, 0, 1)], # B10, B11
            'C': [(0, 0, 1), (1, 0, -1)] # C00, C01
        }
    ]
    
    # Helper to get global index from block index and local offset
    # Block 0: rows 0-1, cols 0-1. Global row = br*2 + lr, Global col = bc*2 + lc.
    # Index = Global row * 4 + Global col.
    
    def get_global_idx(br, bc, lr, lc):
        gr = br * 2 + lr
        gc = bc * 2 + lc
        return gr * 4 + gc

    for s in range(7):
        term = terms[s]
        for k in range(7): # Base Strassen factor index
            col = s * 7 + k
            
            # Update U4
            for (br, bc, coeff) in term['A']:
                # The local 2x2 matrix for this block has elements a,b,c,d
                # U2[row][k] gives the coefficient for element 'row'
                # We need to map local rows 0,1,2,3 to global indices
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        val = U2[local_idx][k]
                        if val != 0:
                            g_idx = get_global_idx(br, bc, lr, lc)
                            U4[g_idx][col] += coeff * val

            # Update V4
            for (br, bc, coeff) in term['B']:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        val = V2[local_idx][k]
                        if val != 0:
                            g_idx = get_global_idx(br, bc, lr, lc)
                            V4[g_idx][col] += coeff * val

            # Update W4
            for (br, bc, coeff) in term['C']:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        val = W2[local_idx][k]
                        if val != 0:
                            g_idx = get_global_idx(br, bc, lr, lc)
                            W4[g_idx][col] += coeff * val

    # Build 4x4x2 Schoolbook decomposition (Rank 32)
    # A is 4x4, B2 is 4x2, C2 is 4x2
    # U_2p: 16x32
    # V_2p: 8x32 (rows 16-23 of V)
    # W_2p: 8x32 (rows 16-23 of W)
    rank_442 = 32
    U_2p = [[0]*rank_442 for _ in range(16)]
    V_2p = [[0]*rank_442 for _ in range(8)]
    W_2p = [[0]*rank_442 for _ in range(8)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_2p[i*4 + j][col] = 1
                V_2p[j*2 + k][col] = 1
                W_2p[k*4 + i][col] = 1
                col += 1

    # Combine into final U, V, W
    # U: 16 x (49+32)
    # V: 24 x (49+32)
    # W: 24 x (49+32)
    
    total_rank = rank_444 + rank_442
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Fill U
    for i in range(16):
        for c in range(rank_444):
            U[i][c] = U4[i][c]
        for c in range(rank_442):
            U[i][rank_444 + c] = U_2p[i][c]
            
    # Fill V
    # V4 occupies rows 0-15
    for i in range(16):
        for c in range(rank_444):
            V[i][c] = V4[i][c]
    # V_2p occupies rows 16-23
    for i in range(8):
        for c in range(rank_442):
            V[16 + i][rank_444 + c] = V_2p[i][c]
            
    # Fill W
    # W4 occupies rows 0-15
    for i in range(16):
        for c in range(rank_444):
            W[i][c] = W4[i][c]
    # W_2p occupies rows 16-23
    for i in range(8):
        for c in range(rank_442):
            W[16 + i][rank_444 + c] = W_2p[i][c]

    return {"U": U, "V": V, "W": W}
