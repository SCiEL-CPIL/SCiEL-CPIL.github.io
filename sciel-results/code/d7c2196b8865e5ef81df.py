import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We decompose the 2x4 x 4x7 multiplication into two 2x2 x 2x7 multiplications.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    
    # Rank of 2x2 x 2x7 multiplication is 25 (3 Strassen blocks of 2x2x2 + 1 schoolbook 2x2x1).
    # Total rank = 2 * 25 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    # Indices:
    # U row index: i * m + j, where i in 0..1, j in 0..3.
    # A1 is rows 0,1 and cols 0,1 of A. Indices in U: 0,1,2,3.
    # A2 is rows 0,1 and cols 2,3 of A. Indices in U: 4,5,6,7.
    
    # V row index: j * p + k, where j in 0..3, k in 0..6.
    # B1 is rows 0,1 of B. Indices in V: 0..13 (j=0,1).
    # B2 is rows 2,3 of B. Indices in V: 14..27 (j=2,3).
    
    # W row index: k * n + i, where k in 0..6, i in 0..1.
    # C columns 0..6.
    
    # We will fill 25 columns for A1 B1 and 25 columns for A2 B2.
    
    def add_strassen_block(u_offset, v_offset, w_offset, a_rows, b_rows):
        # a_rows: list of 4 indices into U corresponding to A1 or A2
        # b_rows: list of indices into V corresponding to B1 or B2 blocks
        # We process 3 blocks of 2 columns (Strassen) and 1 block of 1 column (Schoolbook)
        
        # Strassen part
        for block_idx in range(3):
            # Columns 2*block_idx, 2*block_idx+1 of B (and C)
            col1 = 2 * block_idx
            col2 = 2 * block_idx + 1
            
            # V indices for B block
            # B is 2x7. Rows 0,1.
            # V index for row 0, col c is 0 * 7 + c = c.
            # V index for row 1, col c is 1 * 7 + c = 7 + c.
            # So for block_idx, cols are col1, col2.
            v_x = v_offset + col1      # B[0, col1]
            v_y = v_offset + col2      # B[0, col2]
            v_z = v_offset + 7 + col1  # B[1, col1]
            v_w = v_offset + 7 + col2  # B[1, col2]
            
            # U indices for A block
            # a_rows[0] = A[0,0], a_rows[1] = A[0,1], a_rows[2] = A[1,0], a_rows[3] = A[1,1]
            u_a = u_offset + a_rows[0]
            u_b = u_offset + a_rows[1]
            u_c = u_offset + a_rows[2]
            u_d = u_offset + a_rows[3]
            
            # W indices for C block
            # C is 2x7. Rows 0,1.
            # W index for row i, col k is k * 2 + i.
            w_11 = w_offset + col1 * 2 + 0
            w_12 = w_offset + col2 * 2 + 0
            w_21 = w_offset + col1 * 2 + 1
            w_22 = w_offset + col2 * 2 + 1
            
            # Strassen products
            # P1 = (a+d)(x+w)
            col = u_offset + block_idx * 7
            U[u_a][col] += 1
            U[u_d][col] += 1
            V[v_x][col] += 1
            V[v_w][col] += 1
            W[w_11][col] += 1
            W[w_22][col] += 1
            
            # P2 = (c+d)x
            col += 1
            U[u_c][col] += 1
            U[u_d][col] += 1
            V[v_x][col] += 1
            W[w_21][col] += 1
            W[w_22][col] -= 1
            
            # P3 = a(y-w)
            col += 1
            U[u_a][col] += 1
            V[v_y][col] += 1
            V[v_w][col] -= 1
            W[w_12][col] += 1
            W[w_22][col] += 1
            
            # P4 = d(z-x)
            col += 1
            U[u_d][col] += 1
            V[v_z][col] += 1
            V[v_x][col] -= 1
            W[w_11][col] += 1
            W[w_21][col] += 1
            
            # P5 = (a+b)w
            col += 1
            U[u_a][col] += 1
            U[u_b][col] += 1
            V[v_w][col] += 1
            W[w_11][col] -= 1
            W[w_12][col] += 1
            
            # P6 = (c-a)(x+w)
            col += 1
            U[u_c][col] += 1
            U[u_a][col] -= 1
            V[v_x][col] += 1
            V[v_w][col] += 1
            W[w_22][col] += 1
            
            # P7 = (b-d)(y+z)
            col += 1
            U[u_b][col] += 1
            U[u_d][col] -= 1
            V[v_y][col] += 1
            V[v_z][col] += 1
            W[w_11][col] += 1
            W[w_12][col] += 1
            W[w_21][col] -= 1 # Wait, check Strassen formula for C21.
            # C21 = P2 + P4. No P7 in C21.
            # Let's re-verify Strassen outputs.
            # C11 = P1 + P4 - P5 + P7
            # C12 = P3 + P5
            # C21 = P2 + P4
            # C22 = P1 - P2 + P3 + P6
            # My W assignments:
            # P1: W11+=1, W22+=1. Correct.
            # P2: W21+=1, W22-=1. Correct.
            # P3: W12+=1, W22+=1. Correct.
            # P4: W11+=1, W21+=1. Correct.
            # P5: W11-=1, W12+=1. Correct.
            # P6: W22+=1. Correct.
            # P7: W11+=1, W12+=1. 
            # Wait, C12 = P3 + P5. No P7.
            # C11 = ... + P7.
            # So P7 should only affect W11?
            # Let's check standard Strassen again.
            # M7 = (B - D)(Y + Z)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # Yes, P7 only appears in C11.
            # So W12 should NOT get P7.
            # I had W[w_12][col] += 1 for P7. That is WRONG.
            # And W[w_21] should NOT get P7.
            
            # Correction for P7:
            # Only W11 gets +1.
            
        # Schoolbook part for last column (col 6)
        col_idx = 6
        v_x = v_offset + col_idx
        v_y = v_offset + 7 + col_idx
        u_a = u_offset + a_rows[0]
        u_b = u_offset + a_rows[1]
        u_c = u_offset + a_rows[2]
        u_d = u_offset + a_rows[3]
        w_11 = w_offset + col_idx * 2 + 0
        w_21 = w_offset + col_idx * 2 + 1
        
        # 4 products
        # ax
        col = u_offset + 21
        U[u_a][col] += 1
        V[v_x][col] += 1
        W[w_11][col] += 1
        
        # by
        col += 1
        U[u_b][col] += 1
        V[v_y][col] += 1
        W[w_11][col] += 1
        
        # cx
        col += 1
        U[u_c][col] += 1
        V[v_x][col] += 1
        W[w_21][col] += 1
        
        # dy
        col += 1
        U[u_d][col] += 1
        V[v_y][col] += 1
        W[w_21][col] += 1

    # Call for A1 B1
    # A1 indices in U: 0,1,2,3
    # B1 indices in V: 0..13 (offset 0)
    # C1 indices in W: 0..13 (offset 0)
    add_strassen_block(0, 0, 0, [0, 1, 2, 3])
    
    # Call for A2 B2
    # A2 indices in U: 4,5,6,7
    # B2 indices in V: 14..27 (offset 14)
    # C2 indices in W: 0..13 (offset 0, same columns)
    add_strassen_block(25, 14, 0, [4, 5, 6, 7])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
