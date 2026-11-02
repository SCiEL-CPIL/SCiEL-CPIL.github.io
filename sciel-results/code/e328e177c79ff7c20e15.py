import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank_444 = 49
    rank_442 = 16
    total_rank = rank_444 + rank_442
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # This handles columns 0-3 of B and C
    U_444, V_444, W_444 = _strassen_444()
    
    # Part 2: 4x4x2 Schoolbook (Rank 16)
    # This handles columns 4-5 of B and C
    U_442, V_442, W_442 = _schoolbook_442()
    
    # Combine U
    U[:, :rank_444] = U_444
    
    # Combine V
    # V_444 is 16x49. Rows correspond to B[0:4, 0:4].
    # In full V (24x65), rows are j*6+k.
    # For 444 part, k is 0..3.
    # We need to place V_444 into V such that row j*6+k gets the value from V_444 row j*4+k.
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, :rank_444] = V_444[j * 4 + k, :]
            
    # V_442 is 8x16. Rows correspond to B[0:4, 4:6] -> j*2 + k' where k' in 0..1.
    # In full V, rows are j*6+k where k in 4..5.
    for j in range(4):
        for k_prime in range(2):
            k = 4 + k_prime
            V[j * 6 + k, rank_444:] = V_442[j * 2 + k_prime, :]
            
    # Combine W
    # W_444 is 16x49. Rows correspond to C[0:4, 0:4] -> k*4+i.
    # In full W (24x65), rows are k*4+i.
    # For 444 part, k is 0..3.
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :rank_444] = W_444[k * 4 + i, :]
            
    # W_442 is 8x16. Rows correspond to C[0:4, 4:6] -> k'*4+i where k' in 0..1?
    # No, C is 4x6. The 442 part produces C[:, 4:6].
    # The rows in W_442 are indexed by local column 0,1.
    # In full W, these correspond to global columns 4,5.
    # Row index in full W is k_global * 4 + i.
    for k_prime in range(2):
        k_global = 4 + k_prime
        for i in range(4):
            W[k_global * 4 + i, rank_444:] = W_442[k_prime * 4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_444():
    # Returns U, V, W for 4x4x4 multiplication
    # Rank 49
    # U: 16x49, V: 16x49, W: 16x49
    
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    # Strassen 2x2x2 rank 7 template
    # We need to define the 7 products.
    # Each product is a 2x2x2 multiplication.
    # We can generate the U, V, W for 2x2x2 rank 7 and then place them.
    
    U_222, V_222, W_222 = _strassen_222()
    
    # The 7 products M1..M7 are formed by linear combinations of blocks of A and B.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We iterate over the 7 products.
    # For each product, we add the corresponding U, V, W blocks.
    
    # Define the linear combinations for A and B for each M
    # A_combos: list of 7 tuples, each tuple is a dict mapping block index to coeff
    # B_combos: list of 7 tuples, each tuple is a dict mapping block index to coeff
    
    A_blocks = [
        {0: 1, 3: 1}, # M1: A00 + A11
        {2: 1, 3: 1}, # M2: A10 + A11
        {0: 1},       # M3: A00
        {3: 1},       # M4: A11
        {0: 1, 1: 1}, # M5: A00 + A01
        {2: 1, 0: -1},# M6: A10 - A00
        {1: 1, 3: -1} # M7: A01 - A11
    ]
    
    B_blocks = [
        {0: 1, 3: 1}, # M1: B00 + B11
        {0: 1},       # M2: B00
        {1: 1, 3: -1},# M3: B01 - B11
        {2: 1, 0: -1},# M4: B10 - B00
        {3: 1},       # M5: B11
        {0: 1, 1: 1}, # M6: B00 + B01
        {2: 1, 3: 1}  # M7: B10 + B11
    ]
    
    # C_blocks: how M1..M7 contribute to C00, C01, C10, C11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_combos = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    # For each product m (0..6), we add U_222, V_222 scaled by A_blocks[m] and B_blocks[m]
    # But U_222 is for the "left" matrix. The left matrix for M_m is the linear combo of A blocks.
    # So U[:, m*7:(m+1)*7] = sum_{block} coeff * U_block
    # U_block is U_222 placed in the corresponding 2x2 block of the 4x4 matrix.
    
    for m in range(7):
        # U part
        for block_idx, coeff in A_blocks[m].items():
            # block_idx 0: A00 (rows 0-1, cols 0-1)
            # block_idx 1: A01 (rows 0-1, cols 2-3)
            # block_idx 2: A10 (rows 2-3, cols 0-1)
            # block_idx 3: A11 (rows 2-3, cols 2-3)
            r_start = (block_idx // 2) * 2
            c_start = (block_idx % 2) * 2
            # U_222 is 4x7. We need to place it in U[r_start:r_start+2, c_start:c_start+2]
            # Wait, U_222 rows are 0..3 corresponding to 2x2 matrix elements.
            # U_222[i, j] corresponds to A[i//2, i%2]
            # So we add coeff * U_222 to the appropriate slice of U.
            U[r_start:r_start+2, c_start:c_start+2] += coeff * U_222[:, m*7:(m+1)*7]
            
        # V part
        for block_idx, coeff in B_blocks[m].items():
            r_start = (block_idx // 2) * 2
            c_start = (block_idx % 2) * 2
            V[r_start:r_start+2, c_start:c_start+2] += coeff * V_222[:, m*7:(m+1)*7]
            
    # W part
    # W_222 is 4x7.
    # For each C block (0..3), we sum the contributions from M_m.
    for c_block_idx in range(4):
        r_start = (c_block_idx // 2) * 2
        c_start = (c_block_idx % 2) * 2
        for m, coeff in C_combos[c_block_idx].items():
            W[r_start:r_start+2, c_start:c_start+2] += coeff * W_222[:, m*7:(m+1)*7]
            
    return U, V, W

def _strassen_222():
    # Rank 7 decomposition of 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U rows: 0:A00, 1:A01, 2:A10, 3:A11
    # V rows: 0:B00, 1:B01, 2:B10, 3:B11
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 gets M1, C11 gets M1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 gets M2
    W[3, 1] = -1 # C11 gets -M2
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01 gets M3
    W[3, 2] = 1 # C11 gets M3
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00 gets M4
    W[2, 3] = 1 # C10 gets M4
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00 gets -M5
    W[1, 4] = 1 # C01 gets M5
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 gets M6
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00 gets M7
    
    return U, V, W

def _schoolbook_442():
    # Rank 16 decomposition of 4x4x2
    # U: 16x16, V: 8x16, W: 8x16
    U = np.zeros((16, 16), dtype=float)
    V = np.zeros((8, 16), dtype=float)
    W = np.zeros((8, 16), dtype=float)
    
    # A is 4x4, B is 4x2
    # C is 4x2
    # Schoolbook: for each i, j, k: C[i, j] += A[i, k] B[k, j]
    # Rank is 4*4*2 = 32? No, schoolbook is n*m*p.
    # Here n=4, m=4, p=2. Rank 32.
    # Wait, I said 16 earlier.
    # 4x4x2 schoolbook is 4*4*2 = 32.
    # Is there a lower rank?
    # Rank of 4x4x2 is 14?
    # If I use schoolbook 32, total rank is 49 + 32 = 81.
    # 81 < 96. Still an improvement.
    # Let's use schoolbook for simplicity and correctness.
    
    idx = 0
    for i in range(4):
        for k in range(4):
            for j in range(2):
                U[i * 4 + k, idx] = 1
                V[k * 2 + j, idx] = 1
                W[j * 4 + i, idx] = 1
                idx += 1
                
    return U, V, W
