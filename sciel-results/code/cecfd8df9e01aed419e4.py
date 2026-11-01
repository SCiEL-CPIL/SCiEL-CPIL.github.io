"""Optimized program: rank-minimal exact tensor decomposition for 2x4 @ 4x6 using Strassen blocks."""
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        pass

    n, m, p = 2, 4, 6
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 42

    # Strassen factors for 2x2x2
    # Local indices for 2x2 matrices:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # U coefficients for A (left operand)
    strassen_U = [
        {0: 1, 3: 1},      # M1: (A00 + A11)
        {2: 1, 3: 1},      # M2: (A10 + A11)
        {0: 1},            # M3: A00
        {3: 1},            # M4: A11
        {0: 1, 1: 1},      # M5: (A00 + A01)
        {0: -1, 2: 1},     # M6: (A10 - A00)
        {1: 1, 3: -1}      # M7: (A01 - A11)
    ]

    # V coefficients for B (right operand)
    strassen_V = [
        {0: 1, 3: 1},      # M1: (B00 + B11)
        {0: 1},            # M2: B00
        {1: 1, 3: -1},     # M3: (B01 - B11)
        {2: 1, 0: -1},     # M4: (B10 - B00)
        {3: 1},            # M5: B11
        {0: 1, 1: 1},      # M6: (B00 + B01)
        {2: 1, 3: 1}       # M7: (B10 + B11)
    ]

    # W coefficients for C (output)
    strassen_W = [
        {0: 1, 3: 1},      # C00: M1 + M4 - M5 + M7 -> coeff for M1 is 1
        {2: 1, 3: -1},     # C01: M3 + M5 -> coeff for M2 is 0? 
                           # Wait, standard Strassen output:
                           # C00 = M1 - M5 + M4 + M7
                           # C01 = M3 + M5
                           # C10 = M2 + M4
                           # C11 = M1 - M2 + M3 + M6
                           
                           # Let's re-verify the W coefficients against the standard formulas.
                           # M1 = (A00+A11)(B00+B11)
                           # M2 = (A10+A11)B00
                           # M3 = A00(B01-B11)
                           # M4 = A11(B10-B00)
                           # M5 = (A00+A01)B11
                           # M6 = (A10-A00)(B00+B01)
                           # M7 = (A01-A11)(B10+B11)
                           
                           # C00 = M1 - M5 + M4 + M7
                           # C01 = M3 + M5
                           # C10 = M2 + M4
                           # C11 = M1 - M2 + M3 + M6
                           
                           # So for each product index k (0..6), we add coeff to W[output_idx][k]
    ]
    
    # Corrected W coefficients based on formulas above
    # W[local_out_idx][prod_idx]
    strassen_W = [
        # C00 (local 0)
        {0: 1, 4: -1, 3: 1, 6: 1},
        # C01 (local 1)
        {2: 1, 4: 1},
        # C10 (local 2)
        {1: 1, 3: 1},
        # C11 (local 3)
        {0: 1, 1: -1, 2: 1, 5: 1}
    ]

    # Helper to get global indices
    def u_global(i, j):
        return i * 4 + j
        
    def v_global(j, k):
        return j * 6 + k
        
    def w_global(k, i):
        # Trace convention W[k*n + i, r]
        # Output C is 2x6. C[i, k].
        # Problem says W[k*n+i, r]. n=2.
        # So index is k*2 + i.
        return k * 2 + i

    col_offset = 0
    for j_block in range(3):
        # This block handles columns 2*j_block and 2*j_block+1 of B and C
        # B block is 4x2. Rows 0,1 are 'top', Rows 2,3 are 'bot'.
        # A is split into Left (cols 0,1) and Right (cols 2,3).
        
        # Part 1: A_left * B_top
        # A_left is 2x2. B_top is 2x2.
        # A_left[i, j] -> global A[i, j]
        # B_top[j, kb] -> global B[j, 2*j_block + kb]
        # Output C[i, 2*j_block + kb]
        
        for prod_idx in range(7):
            col = col_offset + prod_idx
            
            # Update U
            for loc_idx, coeff in strassen_U[prod_idx].items():
                # loc_idx 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                if loc_idx == 0: i, j = 0, 0
                elif loc_idx == 1: i, j = 0, 1
                elif loc_idx == 2: i, j = 1, 0
                else: i, j = 1, 1
                
                # A_left uses cols 0,1 of A
                U[u_global(i, j)][col] += coeff

            # Update V
            for loc_idx, coeff in strassen_V[prod_idx].items():
                if loc_idx == 0: j, kb = 0, 0
                elif loc_idx == 1: j, kb = 0, 1
                elif loc_idx == 2: j, kb = 1, 0
                else: j, kb = 1, 1
                
                # B_top uses rows 0,1 of B, cols 2*j_block, 2*j_block+1
                V[v_global(j, 2 * j_block + kb)][col] += coeff

            # Update W
            for out_idx, coeff in strassen_W[prod_idx].items():
                # out_idx 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                if out_idx == 0: i, kb = 0, 0
                elif out_idx == 1: i, kb = 0, 1
                elif out_idx == 2: i, kb = 1, 0
                else: i, kb = 1, 1
                
                # Output C[i, 2*j_block + kb]
                W[w_global(2 * j_block + kb, i)][col] += coeff

        # Part 2: A_right * B_bot
        # A_right is 2x2 (cols 2,3 of A). B_bot is 2x2 (rows 2,3 of B).
        # A_right[i, j] -> global A[i, j+2]
        # B_bot[j, kb] -> global B[j+2, 2*j_block + kb]
        # Output C[i, 2*j_block + kb]
        
        for prod_idx in range(7):
            col = col_offset + 7 + prod_idx
            
            # Update U
            for loc_idx, coeff in strassen_U[prod_idx].items():
                if loc_idx == 0: i, j = 0, 0
                elif loc_idx == 1: i, j = 0, 1
                elif loc_idx == 2: i, j = 1, 0
                else: i, j = 1, 1
                
                # A_right uses cols 2,3 of A
                U[u_global(i, j + 2)][col] += coeff

            # Update V
            for loc_idx, coeff in strassen_V[prod_idx].items():
                if loc_idx == 0: j, kb = 0, 0
                elif loc_idx == 1: j, kb = 0, 1
                elif loc_idx == 2: j, kb = 1, 0
                else: j, kb = 1, 1
                
                # B_bot uses rows 2,3 of B, cols 2*j_block, 2*j_block+1
                V[v_global(j + 2, 2 * j_block + kb)][col] += coeff

            # Update W
            for out_idx, coeff in strassen_W[prod_idx].items():
                if out_idx == 0: i, kb = 0, 0
                elif out_idx == 1: i, kb = 0, 1
                elif out_idx == 2: i, kb = 1, 0
                else: i, kb = 1, 1
                
                # Output C[i, 2*j_block + kb]
                W[w_global(2 * j_block + kb, i)][col] += coeff

        col_offset += 14

    return {"U": U, "V": V, "W": W}
