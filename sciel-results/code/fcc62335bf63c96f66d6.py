import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists or numpy arrays.
    # The interface requires {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    # Note: The problem statement says "integer/half-integer entries".
    # Strassen uses integers. Schoolbook uses integers.
    
    # Total rank
    # 12 blocks of 2x2x2 (rank 7) -> 84
    # 2 blocks of 2x1x6 (rank 12) -> 24
    # Total rank = 108
    
    total_rank = 108
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(u_offset, v_offset, w_offset):
        nonlocal col
        # u_offset: index in U for the start of the 2x2 block of A (4 elements)
        # v_offset: index in V for the start of the 2x2 block of B (4 elements)
        # w_offset: index in W for the start of the 2x2 block of C (4 elements)
        
        # Strassen's 7 products for 2x2 matrices
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # Indices in local 2x2:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # We need to map these to global U, V, W indices.
        # U indices for A block:
        # u0 = u_offset + 0 (a)
        # u1 = u_offset + 1 (b)
        # u2 = u_offset + 2 (c)
        # u3 = u_offset + 3 (d)
        
        # V indices for B block:
        # v0 = v_offset + 0 (e)
        # v1 = v_offset + 1 (f)
        # v2 = v_offset + 2 (g)
        # v3 = v_offset + 3 (h)
        
        # W indices for C block:
        # w0 = w_offset + 0 (C00)
        # w1 = w_offset + 1 (C01)
        # w2 = w_offset + 2 (C10)
        # w3 = w_offset + 3 (C11)
        
        # M1
        U[u0][col] = 1
        U[u3][col] = 1
        V[v0][col] = 1
        V[v3][col] = 1
        W[w0][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M2
        U[u2][col] = 1
        U[u3][col] = 1
        V[v0][col] = 1
        W[w2][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M3
        U[u0][col] = 1
        V[v1][col] = 1
        V[v3][col] = -1
        W[w1][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M4
        U[u3][col] = 1
        V[v2][col] = 1
        V[v0][col] = -1
        W[w2][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M5
        U[u0][col] = 1
        U[u1][col] = 1
        V[v3][col] = 1
        W[w1][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M6
        U[u2][col] = 1
        U[u0][col] = -1
        V[v0][col] = 1
        V[v1][col] = 1
        W[w0][col] = 1
        W[w2][col] = 1
        col += 1
        
        # M7
        U[u1][col] = 1
        U[u3][col] = -1
        V[v2][col] = 1
        V[v3][col] = 1
        W[w0][col] = 1
        W[w1][col] = 1
        col += 1

    # Helper to add a 2x1x6 block (Schoolbook for vector outer product)
    # A is 2x1 (indices u0, u1), B is 1x6 (indices v0..v5)
    # C is 2x6 (indices w0..w11)
    def add_outer_block(u_offset, v_offset, w_offset):
        nonlocal col
        # u0 = u_offset, u1 = u_offset + 1
        # v0..v5
        # w0..w11 (row-major: w00, w01, w02, w03, w04, w05, w10, w11, w12, w13, w14, w15)
        
        for i in range(2):
            for j in range(6):
                U[u_offset + i][col] = 1
                V[v_offset + j][col] = 1
                W[w_offset + i * 6 + j][col] = 1
                col += 1

    # Partition A (4x5)
    # Rows 0,1 -> A_top (2x5)
    # Rows 2,3 -> A_bot (2x5)
    # Cols 0,1 -> Col01 (2x2)
    # Cols 2,3 -> Col23 (2x2)
    # Col 4 -> Col4 (2x1)
    
    # A_top_Col01: u_offset 0 (indices 0,1,2,3 in U? No, U is 20xR)
    # U index = i*m + k.
    # A is 4x5. m=5.
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3], A[0,4]=U[4]
    # A[1,0]=U[5], A[1,1]=U[6], A[1,2]=U[7], A[1,3]=U[8], A[1,4]=U[9]
    # A[2,0]=U[10]...
    # A[3,0]=U[15]...
    
    # Block A_top_Col01 (rows 0,1; cols 0,1):
    # U indices: 0,1, 5,6
    # This is not contiguous in U!
    # My add_strassen_block assumes contiguous U indices for the 2x2 block.
    # I need to handle non-contiguous indices.
    
    # Let's modify add_strassen_block to take explicit indices.
    
    def add_strassen_block_explicit(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: list of 4 indices
        # v_idx: list of 4 indices
        # w_idx: list of 4 indices
        
        u0, u1, u2, u3 = u_idx
        v0, v1, v2, v3 = v_idx
        w0, w1, w2, w3 = w_idx
        
        # M1
        U[u0][col] = 1
        U[u3][col] = 1
        V[v0][col] = 1
        V[v3][col] = 1
        W[w0][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M2
        U[u2][col] = 1
        U[u3][col] = 1
        V[v0][col] = 1
        W[w2][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M3
        U[u0][col] = 1
        V[v1][col] = 1
        V[v3][col] = -1
        W[w1][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M4
        U[u3][col] = 1
        V[v2][col] = 1
        V[v0][col] = -1
        W[w2][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M5
        U[u0][col] = 1
        U[u1][col] = 1
        V[v3][col] = 1
        W[w1][col] = 1
        W[w3][col] = 1
        col += 1
        
        # M6
        U[u2][col] = 1
        U[u0][col] = -1
        V[v0][col] = 1
        V[v1][col] = 1
        W[w0][col] = 1
        W[w2][col] = 1
        col += 1
        
        # M7
        U[u1][col] = 1
        U[u3][col] = -1
        V[v2][col] = 1
        V[v3][col] = 1
        W[w0][col] = 1
        W[w1][col] = 1
        col += 1

    def add_outer_block_explicit(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: list of 2 indices
        # v_idx: list of 6 indices
        # w_idx: list of 12 indices
        
        for i in range(2):
            for j in range(6):
                U[u_idx[i]][col] = 1
                V[v_idx[j]][col] = 1
                W[w_idx[i * 6 + j]][col] = 1
                col += 1

    # Define indices for A (4x5)
    # A[i, k] -> U[i*5 + k]
    # Block A_top_Col01: i in {0,1}, k in {0,1}
    # u_idx: [0*5+0, 0*5+1, 1*5+0, 1*5+1] = [0, 1, 5, 6]
    
    # Block A_top_Col23: i in {0,1}, k in {2,3}
    # u_idx: [2, 3, 7, 8]
    
    # Block A_top_Col4: i in {0,1}, k in {4}
    # u_idx: [4, 9]
    
    # Block A_bot_Col01: i in {2,3}, k in {0,1}
    # u_idx: [10, 11, 15, 16]
    
    # Block A_bot_Col23: i in {2,3}, k in {2,3}
    # u_idx: [12, 13, 17, 18]
    
    # Block A_bot_Col4: i in {2,3}, k in {4}
    # u_idx: [14, 19]

    # Define indices for B (5x6)
    # B[k, j] -> V[k*6 + j]
    # Block B_Col01_Row01: k in {0,1}, j in {0,1}
    # v_idx: [0*6+0, 0*6+1, 1*6+0, 1*6+1] = [0, 1, 6, 7]
    
    # Block B_Col23_Row01: k in {0,1}, j in {2,3}
    # v_idx: [2, 3, 8, 9]
    
    # Block B_Col45_Row01: k in {0,1}, j in {4,5}
    # v_idx: [4, 5, 10, 11]
    
    # Block B_Col01_Row23: k in {2,3}, j in {0,1}
    # v_idx: [12, 13, 18, 19]
    
    # Block B_Col23_Row23: k in {2,3}, j in {2,3}
    # v_idx: [14, 15, 20, 21]
    
    # Block B_Col45_Row23: k in {2,3}, j in {4,5}
    # v_idx: [16, 17, 22, 23]
    
    # Block B_Col01_Row4: k in {4}, j in {0,1}
    # v_idx: [24, 25]
    
    # Block B_Col23_Row4: k in {4}, j in {2,3}
    # v_idx: [26, 27]
    
    # Block B_Col45_Row4: k in {4}, j in {4,5}
    # v_idx: [28, 29]

    # Define indices for C (4x6)
    # C[i, j] -> W[j*4 + i] (Trace convention W[k*n+i, r] where k=j, i=i? 
    # "W[k*n+i,r]" -> k is column of C, i is row of C.
    # So W[j*4 + i]
    
    # Block C_Top_Col01: i in {0,1}, j in {0,1}
    # w_idx: [0*4+0, 0*4+1, 1*4+0, 1*4+1] = [0, 1, 4, 5]
    
    # Block C_Top_Col23: i in {0,1}, j in {2,3}
    # w_idx: [2*4+0, 2*4+1, 3*4+0, 3*4+1] = [8, 9, 12, 13]
    
    # Block C_Top_Col45: i in {0,1}, j in {4,5}
    # w_idx: [4*4+0, 4*4+1, 5*4+0, 5*4+1] = [16, 17, 20, 21]
    
    # Block C_Bot_Col01: i in {2,3}, j in {0,1}
    # w_idx: [0*4+2, 0*4+3, 1*4+2, 1*4+3] = [2, 3, 6, 7]
    
    # Block C_Bot_Col23: i in {2,3}, j in {2,3}
    # w_idx: [2*4+2, 2*4+3, 3*4+2, 3*4+3] = [10, 11, 14, 15]
    
    # Block C_Bot_Col45: i in {2,3}, j in {4,5}
    # w_idx: [4*4+2, 4*4+3, 5*4+2, 5*4+3] = [18, 19, 22, 23]

    # 12 Strassen blocks
    # 1. A_top_Col01 * B_Col01_Row01 -> C_Top_Col01
    add_strassen_block_explicit([0, 1, 5, 6], [0, 1, 6, 7], [0, 1, 4, 5])
    
    # 2. A_top_Col23 * B_Col23_Row01 -> C_Top_Col23
    add_strassen_block_explicit([2, 3, 7, 8], [2, 3, 8, 9], [8, 9, 12, 13])
    
    # 3. A_top_Col01 * B_Col45_Row01 -> C_Top_Col45
    add_strassen_block_explicit([0, 1, 5, 6], [4, 5, 10, 11], [16, 17, 20, 21])
    
    # 4. A_bot_Col01 * B_Col01_Row23 -> C_Bot_Col01
    add_strassen_block_explicit([10, 11, 15, 16], [12, 13, 18, 19], [2, 3, 6, 7])
    
    # 5. A_bot_Col23 * B_Col23_Row23 -> C_Bot_Col23
    add_strassen_block_explicit([12, 13, 17, 18], [14, 15, 20, 21], [10, 11, 14, 15])
    
    # 6. A_bot_Col01 * B_Col45_Row23 -> C_Bot_Col45
    add_strassen_block_explicit([10, 11, 15, 16], [16, 17, 22, 23], [18, 19, 22, 23])
    
    # 7. A_top_Col23 * B_Col01_Row23 -> C_Top_Col01 (Add to existing)
    # Wait, C_Top_Col01 is sum of A_top_Col01*B_Col01_Row01 and A_top_Col23*B_Col01_Row23?
    # No, B is partitioned by rows.
    # C_ij = sum_k A_ik B_kj.
    # C_Top_Col01 (i in 0,1; j in 0,1)
    # = A_top_Col01 * B_Row01_Col01 + A_top_Col23 * B_Row23_Col01 + A_top_Col4 * B_Row4_Col01
    # So we need 3 terms for each 2x2 block of C.
    
    # Let's re-list the 12 Strassen blocks properly.
    # C blocks:
    # C_Top_Col01: Terms from k=0,1 (Row01) and k=2,3 (Row23) and k=4 (Row4)
    # k=0,1: A_top_Col01 * B_Row01_Col01 (Strassen)
    # k=2,3: A_top_Col23 * B_Row23_Col01 (Strassen)
    # k=4: A_top_Col4 * B_Row4_Col01 (Outer)
    
    # So for C_Top_Col01, we have 2 Strassen blocks and 1 Outer block.
    # Similarly for all 6 C blocks.
    # Total Strassen blocks: 6 C blocks * 2 Strassen terms each = 12.
    # Total Outer blocks: 6 C blocks * 1 Outer term each = 6.
    
    # My previous count of 6 outer blocks was wrong.
    # A_top_Col4 is 2x1. B_Row4_Col01 is 1x2. Product is 2x2.
    # This is a 2x1x2 multiplication. Rank 4.
    # There are 6 such 2x1x2 multiplications.
    # 6 * 4 = 24 rank.
    # 12 * 7 = 84 rank.
    # Total 108.
    
    # I need to add the remaining Strassen blocks and the Outer blocks.
    
    # 7. A_top_Col23 * B_Row23_Col01 -> C_Top_Col01
    add_strassen_block_explicit([2, 3, 7, 8], [12, 13, 18, 19], [0, 1, 4, 5])
    
    # 8. A_top_Col23 * B_Row23_Col23 -> C_Top_Col23
    add_strassen_block_explicit([2, 3, 7, 8], [14, 15, 20, 21], [8, 9, 12, 13])
    
    # 9. A_top_Col23 * B_Row23_Col45 -> C_Top_Col45
    add_strassen_block_explicit([2, 3, 7, 8], [16, 17, 22, 23], [16, 17, 20, 21])
    
    # 10. A_bot_Col23 * B_Row01_Col01 -> C_Bot_Col01
    add_strassen_block_explicit([12, 13, 17, 18], [0, 1, 6, 7], [2, 3, 6, 7])
    
    # 11. A_bot_Col23 * B_Row01_Col23 -> C_Bot_Col23
    add_strassen_block_explicit([12, 13, 17, 18], [2, 3, 8, 9], [10, 11, 14, 15])
    
    # 12. A_bot_Col23 * B_Row01_Col45 -> C_Bot_Col45
    add_strassen_block_explicit([12, 13, 17, 18], [4, 5, 10, 11], [18, 19, 22, 23])
    
    # Now the 6 Outer blocks (2x1x2)
    # Each has rank 4.
    
    # 13. A_top_Col4 * B_Row4_Col01 -> C_Top_Col01
    # u_idx: [4, 9]
    # v_idx: [24, 25]
    # w_idx: [0, 1, 4, 5]
    add_outer_block_explicit([4, 9], [24, 25], [0, 1, 4, 5])
    # Wait, add_outer_block_explicit assumes 2x1x6.
    # I need a 2x1x2 version.
    
    def add_outer_2x1x2(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: 2, v_idx: 2, w_idx: 4
        for i in range(2):
            for j in range(2):
                U[u_idx[i]][col] = 1
                V[v_idx[j]][col] = 1
                W[w_idx[i * 2 + j]][col] = 1
                col += 1

    # Re-implementing the last 6 blocks with 2x1x2
    
    # 13. A_top_Col4 * B_Row4_Col01 -> C_Top_Col01
    add_outer_2x1x2([4, 9], [24, 25], [0, 1, 4, 5])
    
    # 14. A_top_Col4 * B_Row4_Col23 -> C_Top_Col23
    add_outer_2x1x2([4, 9], [26, 27], [8, 9, 12, 13])
    
    # 15. A_top_Col4 * B_Row4_Col45 -> C_Top_Col45
    add_outer_2x1x2([4, 9], [28, 29], [16, 17, 20, 21])
    
    # 16. A_bot_Col4 * B_Row4_Col01 -> C_Bot_Col01
    add_outer_2x1x2([14, 19], [24, 25], [2, 3, 6, 7])
    
    # 17. A_bot_Col4 * B_Row4_Col23 -> C_Bot_Col23
    add_outer_2x1x2([14, 19], [26, 27], [10, 11, 14, 15])
    
    # 18. A_bot_Col4 * B_Row4_Col45 -> C_Bot_Col45
    add_outer_2x1x2([14, 19], [28, 29], [18, 19, 22, 23])
    
    # Check col count
    # 12 * 7 = 84
    # 6 * 4 = 24
    # Total 108
    if col != total_rank:
        print(f"Warning: col {col} != total_rank {total_rank}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
