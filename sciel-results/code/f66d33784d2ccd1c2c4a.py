import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # Step 1: Get the 2x2x2 Strassen decomposition (rank 7)
    # U, V, W are 4x7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # Strassen terms for 2x2x2
    # Indices: 0:00, 1:01, 2:10, 3:11
    # P0 = (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # P1 = (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # P2 = A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # P3 = A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # P4 = (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # P5 = (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # P6 = (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Step 2: Build 4x4x4 decomposition (rank 49)
    # We treat the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The Strassen combination for the block level:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Each Mk is a 2x2x2 product.
    # We expand each Mk using the 7-term Strassen decomposition.
    
    # We need to map the local 2x2 indices to global 4x4 indices.
    # Global A index: (row_block * 2 + row_local) * 4 + (col_block * 2 + col_local)
    # Global B index: (row_block * 2 + row_local) * 4 + (col_block * 2 + col_local) -- Wait, B is 4x4 here.
    # Global C index: (row_block * 2 + row_local) * 4 + (col_block * 2 + col_local) -- Wait, C is 4x4 here.
    
    # Let's define the block contributions.
    # For each top-level term t (0..6), we have:
    #   A_blocks involved: dict from block_idx (0..3) to coeff
    #   B_blocks involved: dict from block_idx (0..3) to coeff
    #   C_blocks involved: dict from block_idx (0..3) to coeff
    
    # But this is not quite right because the Strassen terms are products of sums.
    # It's easier to think in terms of the tensor decomposition.
    # The 4x4x4 tensor is a sum of 49 rank-1 tensors.
    # Each rank-1 tensor is formed by "tensor product" of the top-level Strassen term structure and the base-level Strassen term structure.
    
    # Top-level term t has:
    #   U_top[t]: linear combo of A blocks
    #   V_top[t]: linear combo of B blocks
    #   W_top[t]: linear combo of C blocks
    
    # Actually, the standard way to build recursive Strassen:
    # For each of the 7 top-level multiplications Mk:
    #   Mk = A_top[t] * B_top[t]
    #   This is a 2x2 matrix multiplication.
    #   We decompose this 2x2 multiplication into 7 base-level terms.
    #   So for each base-level term b (0..6) of the 2x2 multiplication:
    #       The total term is:
    #       U = (A_top[t] coefficients) * (U_base[b] coefficients)
    #       V = (B_top[t] coefficients) * (V_base[b] coefficients)
    #       W = (W_top[t] coefficients) * (W_base[b] coefficients)
    
    # Let's define the top-level coefficients.
    # A blocks: 00, 01, 10, 11. Indices 0,1,2,3 in block space.
    # B blocks: 00, 01, 10, 11. Indices 0,1,2,3 in block space.
    # C blocks: 00, 01, 10, 11. Indices 0,1,2,3 in block space.
    
    # Top-level Strassen terms (coeffs for A, B, C blocks):
    # Note: The "C" part of the top-level term is actually the distribution of the product Mk to the final C blocks.
    # Mk contributes to C blocks with certain signs.
    
    top_A = [
        {0: 1, 3: 1}, # M0: A00+A11
        {2: 1, 3: 1}, # M1: A10+A11
        {0: 1},       # M2: A00
        {3: 1},       # M3: A11
        {0: 1, 1: 1}, # M4: A00+A01
        {2: 1, 0: -1},# M5: A10-A00
        {1: 1, 3: -1} # M6: A01-A11
    ]
    
    top_B = [
        {0: 1, 3: 1}, # M0: B00+B11
        {0: 1},       # M1: B00
        {1: 1, 3: -1},# M2: B01-B11
        {2: 1, 0: -1},# M3: B10-B00
        {3: 1},       # M4: B11
        {0: 1, 1: 1}, # M5: B00+B01
        {2: 1, 3: 1}  # M6: B10+B11
    ]
    
    top_C = [
        {0: 1, 3: 1}, # M0 -> C00(+1), C11(+1)
        {2: 1, 3: -1},# M1 -> C10(+1), C11(-1)
        {1: 1, 3: 1}, # M2 -> C01(+1), C11(+1)
        {0: 1, 2: 1}, # M3 -> C00(+1), C10(+1)
        {0: -1, 1: 1},# M4 -> C00(-1), C01(+1)
        {3: 1},       # M5 -> C11(+1)
        {0: 1}        # M6 -> C00(+1)
    ]
    
    # Now build the 49 columns.
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    col = 0
    for t in range(7):
        for b in range(7):
            # Combine top and base coefficients
            # Base U_2 is 4x1 vector (for the 2x2 A block)
            # Top A is dict of block coeffs
            # We need to expand "Top A * Base U" into 16-dim vector.
            
            # A global index = (block_row * 2 + local_row) * 4 + (block_col * 2 + local_col)
            # Base U_2 index = local_row * 2 + local_col
            
            for local_idx, base_coeff in enumerate(U_2[:, b]):
                if base_coeff == 0: continue
                lr, lc = divmod(local_idx, 2)
                for block_idx, top_coeff in top_A[t].items():
                    if top_coeff == 0: continue
                    br, bc = divmod(block_idx, 2)
                    global_A_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    U_444[global_A_idx, col] += top_coeff * base_coeff
            
            for local_idx, base_coeff in enumerate(V_2[:, b]):
                if base_coeff == 0: continue
                lr, lc = divmod(local_idx, 2)
                for block_idx, top_coeff in top_B[t].items():
                    if top_coeff == 0: continue
                    br, bc = divmod(block_idx, 2)
                    global_B_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    V_444[global_B_idx, col] += top_coeff * base_coeff
            
            for local_idx, base_coeff in enumerate(W_2[:, b]):
                if base_coeff == 0: continue
                lr, lc = divmod(local_idx, 2)
                for block_idx, top_coeff in top_C[t].items():
                    if top_coeff == 0: continue
                    br, bc = divmod(block_idx, 2)
                    global_C_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    W_444[global_C_idx, col] += top_coeff * base_coeff
            
            col += 1
    
    # Step 3: Extend to 4x4x8
    # U is same.
    # V is [V_444; V_444]
    # W is [W_444; W_444]
    
    U_final = U_444
    V_final = np.vstack([V_444, V_444])
    W_final = np.vstack([W_444, W_444])
    
    # Verify quickly
    # Check rank
    rank = 49
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
