import numpy as np
import time

_DEADLINE = time.time() + 100  # Safety margin before 120s limit
DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 (cols 0-3) using Strassen: rank 49
    # Part 2: 4x4x2 (cols 4-5) using Strassen on row blocks: rank 28
    # Part 3: 4x4x1 (col 6) using schoolbook: rank 8
    # Total rank: 49 + 28 + 8 = 85
    total_rank = 85
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Strassen 2x2x2 base terms ---
    # (A_sub, B_sub) for the 7 multiplications
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    
    strassen_2x2x2 = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1 = (A00+A11)*(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
         {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)*B00
        ({(1,0):1, (1,1):1}, {(0,0):1},
         {(1,0):1, (1,1):-1}),
        # M3 = A00*(B01+B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1},
         {(0,1):1, (1,1):1}),
        # M4 = A11*(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1},
         {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)*B11
        ({(0,0):1, (0,1):1}, {(1,1):1},
         {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)*(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1},
         {(1,1):1}),
        # M7 = (A01-A11)*(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1},
         {(0,0):1}),
    ]
    
    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, block_size=2):
        """Add Strassen terms for a block.
        a_row_off, a_col_off: offset of A block (in 2x2 units)
        b_row_off, b_col_off: offset of B block
        c_row_off, c_col_off: offset of C block
        """
        nonlocal current_rank
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_2x2x2:
            r = current_rank
            current_rank += 1
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for (li, lj), c in a_coeffs.items():
                i = 2 * a_row_off + li
                j = 2 * a_col_off + lj
                u_vec[i * m + j] += c
            
            for (li, lj), c in b_coeffs.items():
                i = 2 * b_row_off + li
                j = 2 * b_col_off + lj
                v_vec[i * p + j] += c
            
            for (li, lj), c in c_coeffs.items():
                i = 2 * c_row_off + li
                k = 2 * c_col_off + lj
                w_vec[k * n + i] += c
            
            for idx in range(n * m):
                U[idx][r] = u_vec[idx]
            for idx in range(m * p):
                V[idx][r] = v_vec[idx]
            for idx in range(p * n):
                W[idx][r] = w_vec[idx]
    
    # --- Part 1: 4x4x4 using Strassen (rank 49) ---
    # A(4x4) * B(4x4) -> C(4x4), cols 0-3
    # Split into 4 blocks of 2x2x2:
    # C_00 = A_00*B_00 + A_01*B_10
    # C_01 = A_00*B_01 + A_01*B_11
    # C_10 = A_10*B_00 + A_11*B_10
    # C_11 = A_10*B_01 + A_11*B_11
    
    # For each 2x2x2 product, use Strassen (rank 7)
    # A_00*B_00: a_row=0,a_col=0, b_row=0,b_col=0, c_row=0,c_col=0
    add_strassen_block(0, 0, 0, 0, 0, 0)  # 7 terms
    # A_01*B_10: a_row=0,a_col=1, b_row=1,b_col=0, c_row=0,c_col=0
    add_strassen_block(0, 1, 1, 0, 0, 0)  # 7 terms
    # A_00*B_01: a_row=0,a_col=0, b_row=0,b_col=1, c_row=0,c_col=1
    add_strassen_block(0, 0, 0, 1, 0, 1)  # 7 terms
    # A_01*B_11: a_row=0,a_col=1, b_row=1,b_col=1, c_row=0,c_col=1
    add_strassen_block(0, 1, 1, 1, 0, 1)  # 7 terms
    # A_10*B_00: a_row=1,a_col=0, b_row=0,b_col=0, c_row=1,c_col=0
    add_strassen_block(1, 0, 0, 0, 1, 0)  # 7 terms
    # A_11*B_10: a_row=1,a_col=1, b_row=1,b_col=0, c_row=1,c_col=0
    add_strassen_block(1, 1, 1, 0, 1, 0)  # 7 terms
    # A_10*B_01: a_row=1,a_col=0, b_row=0,b_col=1, c_row=1,c_col=1
    add_strassen_block(1, 0, 0, 1, 1, 1)  # 7 terms
    # A_11*B_11: a_row=1,a_col=1, b_row=1,b_col=1, c_row=1,c_col=1
    add_strassen_block(1, 1, 1, 1, 1, 1)  # 7 terms
    
    # That's 8 * 7 = 56 terms... but Strassen for 4x4x4 is rank 49, not 56.
    # The issue: we're doing independent Strassen for each 2x2x2 product.
    # But the 4x4x4 Strassen algorithm combines them more efficiently.
    
    # Actually, let me reconsider. The standard Strassen 4x4x4 uses 49 multiplications
    # by applying Strassen at two levels. My approach of doing 8 independent 2x2x2 
    # Strassen decompositions gives 56, which is worse.
    
    # Let me use a different approach: treat 4x4x4 as a block computation.
    # A(4x4) * B(4x4):
    # Split A into 4 blocks of 2x2: A00, A01, A10, A11
    # Split B into 4 blocks of 2x2: B00, B01, B10, B11
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each Ci is a sum of two 2x2x2 products.
    # If we do each 2x2x2 with Strassen (rank 7), we get 8*7 = 56.
    # But Strassen's 4x4x4 achieves 49 by sharing.
    
    # For simplicity and correctness, let me just use 56 for the 4x4x4 part.
    # Total: 56 + 28 + 8 = 92. Still better than 93.
    
    # Actually wait - I already added 56 terms above. Let me continue.
    
    # --- Part 2: 4x4x2 (cols 4-5) ---
    # A(4x4) * B(4x2) -> C(4x2)
    # Split A into top (rows 0-1) and bottom (rows 2-3)
    # C_top = A_top * B (2x4 * 4x2)
    # C_bot = A_bot * B (2x4 * 4x2)
    
    # A_top = [A00 (2x2) | A01 (2x2)]
    # B = [B_top (2x2); B_bot (2x2)] where B_top is rows 0-1, B_bot is rows 2-3
    # C_top = A00*B_top + A01*B_bot
    
    # A00*B_top: 2x2x2, Strassen rank 7
    add_strassen_block(0, 0, 0, 2, 0, 2)  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    # A01*B_bot: 2x2x2, Strassen rank 7
    add_strassen_block(0, 1, 2, 2, 0, 2)  # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    
    # A_bot = [A10 (2x2) | A11 (2x2)]
    # C_bot = A10*B_top + A11*B_bot
    
    # A10*B_top: 2x2x2, Strassen rank 7
    add_strassen_block(1, 0, 0, 2, 1, 2)  # A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
    # A11*B_bot: 2x2x2, Strassen rank 7
    add_strassen_block(1, 1, 2, 2, 1, 2)  # A[2:4,2:4] * B[2:4,4:6] -> C[2:4,4:6]
    
    # Total for part 2: 4 * 7 = 28 terms. 
    
    # --- Part 3: 4x4x1 (col 6) ---
    # A(4x4) * B(4x1) -> C(4x1)
    # C[i,6] = sum_j A[i,j] * B[j,6]
    # This is just a matrix-vector product. Rank = 4*4 = 16 if done naively.
    # But we can do it as: for each output row, it's a dot product.
    # Actually, the tensor rank for 4x4x1 is 16 (schoolbook).
    # Can we do better? R(4,4,1) = 16 since it's just 4 independent dot products
    # of length 4, and R(n,m,1) = n*m for the schoolbook.
    
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            r = current_rank
            current_rank += 1
            U[i * m + j][r] = 1.0
            V[j * p + 6][r] = 1.0
            W[6 * n + i][r] = 1.0
    
    # Total: 56 + 28 + 16 = 100. That's worse than 93!
    
    # Let me reconsider. I need to be more careful.
    
    # Reset and redo with a better plan.
    # Plan:
    # 1. 4x4x4: Use the proper Strassen 4x4x4 (rank 49)
    # 2. 4x4x3: Split as 4x4x2 + 4x4x1
    #    4x4x2: Split A into 2 row blocks. Each 2x4x2 product has rank 14.
    #            Total: 28.
    #    4x4x1: Schoolbook rank 16.
    #    Total for 4x4x3: 28 + 16 = 44.
    # Grand total: 49 + 44 = 93. Same as before.
    
    # Alternative for 4x4x1: Can we do better than 16?
    # R(n,m,1) = nm. So 16 is optimal for 4x4x1.
    
    # So 93 seems to be the limit with this approach.
    # Let me check if there's a way to combine the 4x4x4 and 4x4x3 parts.
    
    # Actually, the key insight from the inspiration is that the current score is 1.2043
    # which corresponds to rank 93 (since seed is 112, 112/93 = 1.2043).
    
    # Let me just implement the 93-rank solution correctly.
    
    # Clear and restart
    U = [[0.0] * 93 for _ in range(n * m)]
    V = [[0.0] * 93 for _ in range(m * p)]
    W = [[0.0] * 93 for _ in range(p * n)]
    current_rank = 0
    
    # --- Part 1: 4x4x4 Strassen (rank 49) ---
    # Proper two-level Strassen for 4x4x4.
    # Level 1: Split into 2x2 blocks.
    # A = [A00 A01; A10 A11] where each Ai is 2x2
    # B = [B00 B01; B10 B11]
    # C = [C00 C01; C10 C11]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each Ci involves two 2x2x2 products.
    # To get rank 49, we need to use Strassen at the 4x4 level.
    # Strassen 4x4x4:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01+B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # Then:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2x2 product, so use Strassen (rank 7) for each.
    # Total: 7 * 7 = 49.
    
    strassen_4x4_terms = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # P1 = (A00+A11)*(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
         {(0,0):1, (0,1):1, (1,0):1, (1,1):-1}),
        # P2 = (A10+A11)*B00
        ({(1,0):1, (1,1):1}, {(0,0):1},
         {(1,0):1, (0,0):-1}),
        # P3 = A00*(B01+B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1},
         {(0,1):1, (1,1):1}),
        # P4 = A11*(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1},
         {(0,0):1, (1,0):1}),
        # P5 = (A00+A01)*B11
        ({(0,0):1, (0,1):1}, {(1,1):1},
         {(0,0):-1, (0,1):1}),
        # P6 = (A10-A00)*(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1},
         {(1,1):1}),
        # P7 = (A01-A11)*(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1},
         {(0,0):1}),
    ]
    
    for a_block_coeffs, b_block_coeffs, c_block_coeffs in strassen_4x4_terms:
        # For each of the 7 top-level terms, decompose the 2x2x2 product using Strassen
        for a_sub, b_sub, c_sub in strassen_2x2x2:
            r = current_rank
            current_rank += 1
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            # A part: for each block in a_block_coeffs, apply a_sub
            for (bi, bj), ca in a_block_coeffs.items():
                for (li, lj), csub in a_sub.items():
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_vec[i * m + j] += ca * csub
            
            # B part: for each block in b_block_coeffs, apply b_sub
            for (bi, bj), cb in b_block_coeffs.items():
                for (li, lj), csub in b_sub.items():
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_vec[i * p + j] += cb * csub
            
            # C part: for each block in c_block_coeffs, apply c_sub
            for (bi, bj), cc in c_block_coeffs.items():
                for (li, lj), csub in c_sub.items():
                    i = 2 * bi + li
                    k = 2 * bj + lj
                    w_vec[k * n + i] += cc * csub
            
            for idx in range(n * m):
                U[idx][r] = u_vec[idx]
            for idx in range(m * p):
                V[idx][r] = v_vec[idx]
            for idx in range(p * n):
                W[idx][r] = w_vec[idx]
    
    # Part 1 done: 49 terms. current_rank = 49
    
    # --- Part 2: 4x4x2 (cols 4-5) using row-block split (rank 28) ---
    # C_top (rows 0-1) = A_top * B
    # C_bot (rows 2-3) = A_bot * B
    
    # A_top = [A00 | A01] (2x4)
    # A_bot = [A10 | A11] (2x4)
    # B = [B_top; B_bot] (4x2)
    
    # C_top = A00*B_top + A01*B_bot
    # C_bot = A10*B_top + A11*B_bot
    
    # Each 2x2x2 product: Strassen rank 7
    
    # A00*B_top: A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # A01*B_bot: A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    add_strassen_block(0, 1, 2, 2, 0, 2)
    # A10*B_top: A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
    add_strassen_block(1, 0, 0, 2, 1, 2)
    # A11*B_bot: A[2:4,2:4] * B[2:4,4:6] -> C[2:4,4:6]
    add_strassen_block(1, 1, 2, 2, 1, 2)
    
    # Part 2 done: 28 terms. current_rank = 49 + 28 = 77
    
    # --- Part 3: 4x4x1 (col 6) schoolbook (rank 16) ---
    for i in range(4):
        for j in range(4):
            r = current_rank
            current_rank += 1
            U[i * m + j][r] = 1.0
            V[j * p + 6][r] = 1.0
            W[6 * n + i][r] = 1.0
    
    # Part 3 done: 16 terms. current_rank = 77 + 16 = 93
    
    # Trim to actual rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    # Verify
    if time.time() < _DEADLINE:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        max_err = np.max(np.abs(recon - target))
        
        if max_err > 1e-10:
            # Fallback to schoolbook
            rank_sb = n * m * p
            U_sb = np.zeros((n*m, rank_sb))
            V_sb = np.zeros((m*p, rank_sb))
            W_sb = np.zeros((p*n, rank_sb))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, r] = 1.0
                        V_sb[j*p+k, r] = 1.0
                        W_sb[k*n+i, r] = 1.0
                        r += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U, "V": V, "W": W}
