"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    # Shapes:
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Rank 36 decomposition:
    # 8 terms for A (2x4) * B_last_col (4x1)
    # 14 terms for A_left (2x2) * B_top (2x4) using two Strassen blocks
    # 14 terms for A_right (2x2) * B_bot (2x4) using two Strassen blocks
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_indices: indices in U for the 2x2 A block
    # b_indices: indices in V for the 2x2 B block
    # c_indices: indices in W for the 2x2 C block
    # The Strassen formulas:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        
        # M1
        U[a_idx[0], col] += 1
        U[a_idx[3], col] += 1
        V[b_idx[0], col] += 1
        V[b_idx[3], col] += 1
        W[c_idx[0], col] += 1  # C00
        W[c_idx[3], col] += 1  # C11
        col += 1
        
        # M2
        U[a_idx[2], col] += 1
        U[a_idx[3], col] += 1
        V[b_idx[0], col] += 1
        W[c_idx[2], col] += 1  # C10
        W[c_idx[3], col] -= 1  # C11
        col += 1
        
        # M3
        U[a_idx[0], col] += 1
        V[b_idx[1], col] += 1
        V[b_idx[3], col] -= 1
        W[c_idx[1], col] += 1  # C01
        W[c_idx[3], col] += 1  # C11
        col += 1
        
        # M4
        U[a_idx[3], col] += 1
        V[b_idx[2], col] += 1
        V[b_idx[0], col] -= 1
        W[c_idx[0], col] += 1  # C00
        W[c_idx[2], col] += 1  # C10
        col += 1
        
        # M5
        U[a_idx[0], col] += 1
        U[a_idx[1], col] += 1
        V[b_idx[3], col] += 1
        W[c_idx[0], col] -= 1  # C00
        W[c_idx[1], col] += 1  # C01
        col += 1
        
        # M6
        U[a_idx[2], col] += 1
        U[a_idx[0], col] -= 1
        V[b_idx[0], col] += 1
        V[b_idx[1], col] += 1
        W[c_idx[3], col] += 1  # C11
        col += 1
        
        # M7
        U[a_idx[1], col] += 1
        U[a_idx[3], col] -= 1
        V[b_idx[2], col] += 1
        V[b_idx[3], col] += 1
        W[c_idx[0], col] += 1  # C00
        col += 1

    # 1. Last column of B (4x1)
    # A is 2x4. B_last is 4x1.
    # A[i,j] * B[j,4] contributes to C[i,4]
    # C[i,4] index in W: k=4, i=i -> 4*2 + i = 8+i
    for i in range(2):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 2 + i, col] = 1
            col += 1
    # col is now 8

    # 2. A_left (2x2) * B_top (2x4)
    # A_left indices:
    # a00: A[0,0] -> 0
    # a01: A[0,1] -> 1
    # a10: A[1,0] -> 4
    # a11: A[1,1] -> 5
    a_left = [0, 1, 4, 5]
    
    # B_top is rows 0,1 of B. Cols 0,1 and 2,3.
    # B_top_left (cols 0,1):
    # b00: B[0,0] -> 0
    # b01: B[0,1] -> 1
    # b10: B[1,0] -> 5
    # b11: B[1,1] -> 6
    b_top_left = [0, 1, 5, 6]
    
    # C output for A_left * B_top_left:
    # Contributes to C cols 0,1.
    # C[0,0] -> W[0*2+0] = 0
    # C[0,1] -> W[1*2+0] = 2
    # C[1,0] -> W[0*2+1] = 1
    # C[1,1] -> W[1*2+1] = 3
    c_top_left = [0, 2, 1, 3]
    
    add_strassen(a_left, b_top_left, c_top_left)
    # col is now 8 + 7 = 15

    # B_top_right (cols 2,3):
    # b00: B[0,2] -> 2
    # b01: B[0,3] -> 3
    # b10: B[1,2] -> 7
    # b11: B[1,3] -> 8
    b_top_right = [2, 3, 7, 8]
    
    # C output for A_left * B_top_right:
    # Contributes to C cols 2,3.
    # C[0,2] -> W[2*2+0] = 4
    # C[0,3] -> W[3*2+0] = 6
    # C[1,2] -> W[2*2+1] = 5
    # C[1,3] -> W[3*2+1] = 7
    c_top_right = [4, 6, 5, 7]
    
    add_strassen(a_left, b_top_right, c_top_right)
    # col is now 15 + 7 = 22

    # 3. A_right (2x2) * B_bot (2x4)
    # A_right indices:
    # a00: A[0,2] -> 2
    # a01: A[0,3] -> 3
    # a10: A[1,2] -> 6
    # a11: A[1,3] -> 7
    a_right = [2, 3, 6, 7]
    
    # B_bot is rows 2,3 of B. Cols 0,1 and 2,3.
    # B_bot_left (cols 0,1):
    # b00: B[2,0] -> 10
    # b01: B[2,1] -> 11
    # b10: B[3,0] -> 15
    # b11: B[3,1] -> 16
    b_bot_left = [10, 11, 15, 16]
    
    # C output for A_right * B_bot_left:
    # Contributes to C cols 0,1.
    c_bot_left = [0, 2, 1, 3]
    
    add_strassen(a_right, b_bot_left, c_bot_left)
    # col is now 22 + 7 = 29

    # B_bot_right (cols 2,3):
    # b00: B[2,2] -> 12
    # b01: B[2,3] -> 13
    # b10: B[3,2] -> 17
    # b11: B[3,3] -> 18
    b_bot_right = [12, 13, 17, 18]
    
    # C output for A_right * B_bot_right:
    # Contributes to C cols 2,3.
    c_bot_right = [4, 6, 5, 7]
    
    add_strassen(a_right, b_bot_right, c_bot_right)
    # col is now 29 + 7 = 36

    # Convert to lists for return format if needed, but numpy arrays are usually accepted
    # The prompt says "factor matrices of shapes... with integer/half-integer entries"
    # Returning numpy arrays is standard.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
