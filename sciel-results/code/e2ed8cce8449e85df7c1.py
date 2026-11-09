import time
import sys

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 66
    # 3 blocks of 4x4x3, each 4x4x3 is two 2x2x3 (rank 11 each) -> 22 per block.
    total_rank = 66
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_idx: 4 indices for A (i*m+j)
    # b_idx: 4 indices for B (j*p+k)
    # c_idx: 4 indices for C (k*n+i)
    def add_strassen_2x2x2(a_idx, b_idx, c_idx, offset):
        # Strassen terms
        # M1 = (A00+A11)(B00+B11) -> C00 + C11 - C10 + C01 is wrong mapping
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: (A00+A11) * (B00+B11)
        U[a_idx[0]][offset] = 1
        U[a_idx[1]][offset] = 1 # A11 is a_idx[1]? No, a_idx is [A00, A01, A10, A11]
        # Let's define a_idx mapping: [A00, A01, A10, A11]
        # b_idx mapping: [B00, B01, B10, B11]
        # c_idx mapping: [C00, C01, C10, C11]
        
        # M1
        U[a_idx[0]][offset] += 1
        U[a_idx[3]][offset] += 1
        V[b_idx[0]][offset] += 1
        V[b_idx[3]][offset] += 1
        W[c_idx[0]][offset] += 1
        W[c_idx[3]][offset] += 1
        
        # M2: (A10+A11) * B00
        U[a_idx[2]][offset+1] += 1
        U[a_idx[3]][offset+1] += 1
        V[b_idx[0]][offset+1] += 1
        W[c_idx[2]][offset+1] += 1 # C10
        W[c_idx[3]][offset+1] -= 1 # C11
        
        # M3: A00 * (B01-B11)
        U[a_idx[0]][offset+2] += 1
        V[b_idx[1]][offset+2] += 1
        V[b_idx[3]][offset+2] -= 1
        W[c_idx[1]][offset+2] += 1 # C01
        W[c_idx[3]][offset+2] += 1 # C11
        
        # M4: A11 * (B10-B00)
        U[a_idx[3]][offset+3] += 1
        V[b_idx[2]][offset+3] += 1
        V[b_idx[0]][offset+3] -= 1
        W[c_idx[0]][offset+3] += 1 # C00
        W[c_idx[2]][offset+3] += 1 # C10
        
        # M5: (A00+A01) * B11
        U[a_idx[0]][offset+4] += 1
        U[a_idx[1]][offset+4] += 1
        V[b_idx[3]][offset+4] += 1
        W[c_idx[0]][offset+4] -= 1 # C00
        W[c_idx[1]][offset+4] += 1 # C01
        
        # M6: (A10-A00) * (B00+B01)
        U[a_idx[2]][offset+5] += 1
        U[a_idx[0]][offset+5] -= 1
        V[b_idx[0]][offset+5] += 1
        V[b_idx[1]][offset+5] += 1
        W[c_idx[3]][offset+5] += 1 # C11
        
        # M7: (A01-A11) * (B10+B11)
        U[a_idx[1]][offset+6] += 1
        U[a_idx[3]][offset+6] -= 1
        V[b_idx[2]][offset+6] += 1
        V[b_idx[3]][offset+6] += 1
        W[c_idx[0]][offset+6] += 1 # C00
        W[c_idx[2]][offset+6] -= 1 # C10
        
        return offset + 7

    # Helper to add a 2x2x1 Schoolbook decomposition
    def add_schoolbook_2x2x1(a_idx, b_idx, c_idx, offset):
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        # Note: b_idx here is for the single column, so b_idx[0]=B00, b_idx[1]=B10
        # c_idx[0]=C00, c_idx[1]=C10
        
        # M8: A00 * B00
        U[a_idx[0]][offset] += 1
        V[b_idx[0]][offset] += 1
        W[c_idx[0]][offset] += 1
        
        # M9: A01 * B10
        U[a_idx[1]][offset+1] += 1
        V[b_idx[1]][offset+1] += 1
        W[c_idx[0]][offset+1] += 1
        
        # M10: A10 * B00
        U[a_idx[2]][offset+2] += 1
        V[b_idx[0]][offset+2] += 1
        W[c_idx[1]][offset+2] += 1
        
        # M11: A11 * B10
        U[a_idx[3]][offset+3] += 1
        V[b_idx[1]][offset+3] += 1
        W[c_idx[1]][offset+3] += 1
        
        return offset + 4

    # Main Loop: 3 blocks of 4x4x3
    for block in range(3):
        # Block b handles columns 3b, 3b+1, 3b+2 of B and C
        col_start = block * 3
        
        # Top 2x2x3: C_top (rows 0,1) = A11 * B_top + A12 * B_bot
        # A11 indices: i in [0,1], j in [0,1] -> i*m+j
        a11_idx = [0*m+0, 0*m+1, 1*m+0, 1*m+1]
        a12_idx = [0*m+2, 0*m+3, 1*m+2, 1*m+3]
        
        # B_top: j in [0,1], k in [col_start, col_start+2]
        # B_bot: j in [2,3], k in [col_start, col_start+2]
        
        # We need to decompose 2x2x3.
        # Split into 2x2x2 (cols 0,1 of block) and 2x2x1 (col 2 of block)
        
        # 2x2x2 part for C_top (cols 0,1 of block)
        # B_top_2x2: j in [0,1], k in [col_start, col_start+1]
        b_top_2x2_idx = [
            0*p + (col_start), 0*p + (col_start+1),
            1*p + (col_start), 1*p + (col_start+1)
        ]
        b_bot_2x2_idx = [
            2*p + (col_start), 2*p + (col_start+1),
            3*p + (col_start), 3*p + (col_start+1)
        ]
        
        # C_top_2x2: k in [col_start, col_start+1], i in [0,1]
        c_top_2x2_idx = [
            (col_start)*n + 0, (col_start+1)*n + 0,
            (col_start)*n + 1, (col_start+1)*n + 1
        ]
        
        # We have two 2x2x2 multiplications:
        # 1. A11 * B_top_2x2
        # 2. A12 * B_bot_2x2
        # Both contribute to C_top_2x2
        
        # Add Strassen for A11 * B_top_2x2
        rank_offset = add_strassen_2x2x2(a11_idx, b_top_2x2_idx, c_top_2x2_idx, rank_offset)
        
        # Add Strassen for A12 * B_bot_2x2
        rank_offset = add_strassen_2x2x2(a12_idx, b_bot_2x2_idx, c_top_2x2_idx, rank_offset)
        
        # 2x2x1 part for C_top (col 2 of block)
        b_top_1_idx = [
            0*p + (col_start+2),
            1*p + (col_start+2)
        ]
        b_bot_1_idx = [
            2*p + (col_start+2),
            3*p + (col_start+2)
        ]
        c_top_1_idx = [
            (col_start+2)*n + 0,
            (col_start+2)*n + 1
        ]
        
        # Add Schoolbook for A11 * B_top_1
        rank_offset = add_schoolbook_2x2x1(a11_idx, b_top_1_idx, c_top_1_idx, rank_offset)
        
        # Add Schoolbook for A12 * B_bot_1
        rank_offset = add_schoolbook_2x2x1(a12_idx, b_bot_1_idx, c_top_1_idx, rank_offset)
        
        # Bottom 2x2x3: C_bot (rows 2,3) = A21 * B_top + A22 * B_bot
        a21_idx = [2*m+0, 2*m+1, 3*m+0, 3*m+1]
        a22_idx = [2*m+2, 2*m+3, 3*m+2, 3*m+3]
        
        # C_bot_2x2: k in [col_start, col_start+1], i in [2,3]
        c_bot_2x2_idx = [
            (col_start)*n + 2, (col_start+1)*n + 2,
            (col_start)*n + 3, (col_start+1)*n + 3
        ]
        
        # Add Strassen for A21 * B_top_2x2
        rank_offset = add_strassen_2x2x2(a21_idx, b_top_2x2_idx, c_bot_2x2_idx, rank_offset)
        
        # Add Strassen for A22 * B_bot_2x2
        rank_offset = add_strassen_2x2x2(a22_idx, b_bot_2x2_idx, c_bot_2x2_idx, rank_offset)
        
        # C_bot_1: k in [col_start+2], i in [2,3]
        c_bot_1_idx = [
            (col_start+2)*n + 2,
            (col_start+2)*n + 3
        ]
        
        # Add Schoolbook for A21 * B_top_1
        rank_offset = add_schoolbook_2x2x1(a21_idx, b_top_1_idx, c_bot_1_idx, rank_offset)
        
        # Add Schoolbook for A22 * B_bot_1
        rank_offset = add_schoolbook_2x2x1(a22_idx, b_bot_1_idx, c_bot_1_idx, rank_offset)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
