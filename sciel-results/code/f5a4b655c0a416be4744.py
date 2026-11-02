import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    # Strategy: Decompose into three 2x2x2 blocks using Strassen's algorithm.
    # Rank = 3 * 7 = 21.
    
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 coefficients
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # P1 = (A00 + A11) * (B00 + B11) -> C00 += P1, C11 += P1
    # P2 = (A10 + A11) * B00 -> C10 += P2, C11 -= P2
    # P3 = A00 * (B01 - B11) -> C01 += P3, C11 += P3
    # P4 = A11 * (B10 - B00) -> C00 += P4, C10 += P4
    # P5 = (A00 + A01) * B11 -> C00 -= P5, C01 += P5
    # P6 = (A10 - A00) * (B00 + B01) -> C11 += P6
    # P7 = (A01 - A11) * (B10 + B11) -> C00 += P7
    
    # Define the coefficient vectors for A, B, C for each of the 7 terms.
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Term 0: P1
    # A: A00 + A11 => [1, 0, 0, 1]
    # B: B00 + B11 => [1, 0, 0, 1]
    # C: C00 + C11 => [1, 0, 0, 1]
    
    # Term 1: P2
    # A: A10 + A11 => [0, 0, 1, 1]
    # B: B00 => [1, 0, 0, 0]
    # C: C10 - C11 => [0, 0, 1, -1]
    
    # Term 2: P3
    # A: A00 => [1, 0, 0, 0]
    # B: B01 - B11 => [0, 1, 0, -1]
    # C: C01 + C11 => [0, 1, 0, 1]
    
    # Term 3: P4
    # A: A11 => [0, 0, 0, 1]
    # B: B10 - B00 => [-1, 0, 1, 0]
    # C: C00 + C10 => [1, 0, 1, 0]
    
    # Term 4: P5
    # A: A00 + A01 => [1, 1, 0, 0]
    # B: B11 => [0, 0, 0, 1]
    # C: -C00 + C01 => [-1, 1, 0, 0]
    
    # Term 5: P6
    # A: A10 - A00 => [-1, 0, 1, 0]
    # B: B00 + B01 => [1, 1, 0, 0]
    # C: C11 => [0, 0, 0, 1]
    
    # Term 6: P7
    # A: A01 - A11 => [0, 1, 0, -1]
    # B: B10 + B11 => [0, 0, 1, 1]
    # C: C00 => [1, 0, 0, 0]
    
    coeff_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    coeff_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    coeff_C = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Populate the global matrices
    for block_idx in range(3):
        # Each block handles 2 columns of B and C
        # B columns: block_idx*2, block_idx*2+1
        # C columns: block_idx*2, block_idx*2+1
        
        for t in range(7):
            rank_idx = block_idx * 7 + t
            
            # Update U (A part)
            for a_idx in range(4):
                U[a_idx, rank_idx] = coeff_A[t][a_idx]
                
            # Update V (B part)
            for b_idx in range(4):
                # Map local b_idx (0..3) to global B index
                # Local 0: (0,0) -> global col 0
                # Local 1: (0,1) -> global col 1
                # Local 2: (1,0) -> global col 0 (row 1)
                # Local 3: (1,1) -> global col 1 (row 1)
                # Global B index = row * p + col
                # For block_idx, the columns are offset by block_idx * 2
                local_row = b_idx // 2
                local_col = b_idx % 2
                global_col = block_idx * 2 + local_col
                global_b_idx = local_row * p + global_col
                V[global_b_idx, rank_idx] = coeff_B[t][b_idx]
                
            # Update W (C part)
            for c_idx in range(4):
                # Map local c_idx (0..3) to global C index
                # Local 0: (0,0) -> global (0, block_idx*2)
                # Local 1: (0,1) -> global (0, block_idx*2+1)
                # Local 2: (1,0) -> global (1, block_idx*2)
                # Local 3: (1,1) -> global (1, block_idx*2+1)
                # Global C index = col * n + row (trace convention W[k*n+i,r])
                # Wait, the task says W[k*n+i,r].
                # k is column index of C, i is row index of C.
                local_row = c_idx // 2
                local_col = c_idx % 2
                global_col = block_idx * 2 + local_col
                global_c_idx = global_col * n + local_row
                W[global_c_idx, rank_idx] = coeff_C[t][c_idx]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
