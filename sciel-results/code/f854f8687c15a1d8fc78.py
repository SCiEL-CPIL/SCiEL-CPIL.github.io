import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Generate 2x2x2 Strassen Tensors
    # U: 4x7, V: 4x7, W: 4x7
    U222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int).T
    
    V222 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int).T
    
    W222 = np.array([
        [1, 0, 0, 0, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, -1]
    ], dtype=int).T
    
    # Helper to place a small tensor into a block of a larger tensor
    def place_block(target, src, start_row, start_col, rank_offset, rows, cols):
        # src is (src_rows, rank_src)
        # target is (rows, total_rank)
        r_src, c_src = src.shape
        for r in range(r_src):
            for c in range(c_src):
                target[start_row + r, rank_offset + c] += src[r, c]
        
    # Build 4x4x4 Strassen Tensors
    rank_444 = 49
    U444 = np.zeros((16, rank_444), dtype=int)
    V444 = np.zeros((16, rank_444), dtype=int)
    W444 = np.zeros((16, rank_444), dtype=int)
    
    # Strassen 4x4x4 uses 7 block multiplications of 2x2x2
    # Each block multiplication is a 2x2x2 problem.
    # We map the 7 Strassen products to the 4x4 grid.
    
    # The 7 products for 4x4x4 Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We define the block positions for A, B, C in the 4x4x4 case.
    # A blocks: (0,0), (0,2), (2,0), (2,2) in 4x4 matrix.
    # Actually, A is 4x4. Blocks are 2x2.
    # Indices in flattened U (16x7?):
    # A[0:2, 0:2] -> rows 0,1,4,5? No, row-major.
    # A[0,0] -> 0, A[0,1] -> 1, A[1,0] -> 2, A[1,1] -> 3.
    # Block (0,0) uses rows 0,1,4,5? No.
    # A is 4x4.
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    # Row 3: 12, 13, 14, 15
    
    # Block (0,0) (top-left 2x2): rows 0,1; cols 0,1. Flattened indices: 0,1,4,5.
    # Block (0,1) (top-right 2x2): rows 0,1; cols 2,3. Flattened indices: 2,3,6,7.
    # Block (1,0) (bot-left 2x2): rows 2,3; cols 0,1. Flattened indices: 8,9,12,13.
    # Block (1,1) (bot-right 2x2): rows 2,3; cols 2,3. Flattened indices: 10,11,14,15.
    
    # We define mappings for A, B, C blocks.
    # A_blocks[i,j] = list of 4 row indices in U444
    def get_block_indices(row_start, col_start, dim=4, block_size=2):
        rows = []
        for r in range(block_size):
            for c in range(block_size):
                rows.append((row_start + r) * dim + (col_start + c))
        return rows

    A_BL = get_block_indices(0, 0) # 0,1,4,5
    A_BR = get_block_indices(0, 2) # 2,3,6,7
    A_BLL = get_block_indices(2, 0) # 8,9,12,13
    A_BRR = get_block_indices(2, 2) # 10,11,14,15
    
    # B blocks (for 4x4x4, B is 4x4)
    B_BL = get_block_indices(0, 0)
    B_BR = get_block_indices(0, 2)
    B_BLL = get_block_indices(2, 0)
    B_BRR = get_block_indices(2, 2)
    
    # C blocks (for 4x4x4, C is 4x4)
    C_BL = get_block_indices(0, 0)
    C_BR = get_block_indices(0, 2)
    C_BLL = get_block_indices(2, 0)
    C_BRR = get_block_indices(2, 2)
    
    # We define the 7 Strassen combinations for 4x4.
    # Each combination consists of sums/diffs of A blocks and B blocks.
    # We need to generate the U, V, W columns for each of the 7 block multiplications.
    # Since each block multiplication is a 2x2x2 multiplication, we can reuse the U222, V222, W222.
    
    # We iterate over the 7 products.
    # For each product, we determine which A blocks are involved and with what signs,
    # and which B blocks are involved and with what signs.
    
    # Product 1: (A_BL + A_BRR)(B_BL + B_BRR)
    # We need to add the U222 vectors for A_BL and A_BRR.
    # And V222 vectors for B_BL and B_BRR.
    # And W222 vectors for C_BL and C_BRR?
    # No, the result of (A_BL + A_BRR)(B_BL + B_BRR) contributes to C_BL and C_BRR?
    # No, Strassen computes 7 products, and then combines them to form the 4 blocks of C.
    # The products are scalars (if we were doing scalar 1x1x1).
    # Here, the "scalars" are 2x2 matrices.
    # So Product 1 is a 2x2 matrix.
    # It contributes to C_BL, C_BR, C_BLL, C_BRR with specific signs.
    
    # Standard Strassen Output Formulas:
    # C_BL = M1 + M4 - M2 - M5
    # C_BR = M3 + M5
    # C_BLL = M2 + M4
    # C_BRR = M1 - M3 + M6 - M7
    
    # So, for each of the 7 products, we need to add its contribution to the corresponding C blocks.
    # The product itself is computed as a 2x2x2 multiplication.
    
    # We will create a temporary list of operations.
    # Each operation is (prod_id, A_block_map, B_block_map, C_block_contribs)
    
    # This is getting complex to do dynamically with signs.
    # I will hardcode the 7 block mappings for 4x4x4 Strassen.
    
    # Product 1: A=(BL, BRR), B=(BL, BRR)
    # Product 2: A=(BLL, BRR), B=(BL)
    # Product 3: A=(BL), B=(BR, -BRR)
    # Product 4: A=(BRR), B=(BLL, -BL)
    # Product 5: A=(BL, BR), B=(BRR)
    # Product 6: A=(BLL, -BL), B=(BL, BR)
    # Product 7: A=(BR, -BRR), B=(BLL, BRR)
    
    # And C contributions:
    # M1: C_BL(+1), C_BRR(+1)
    # M2: C_BLL(+1), C_BL(-1)
    # M3: C_BR(+1), C_BRR(-1)
    # M4: C_BLL(+1), C_BL(+1)
    # M5: C_BR(+1), C_BL(-1)
    # M6: C_BRR(+1)
    # M7: C_BRR(-1)
    
    # We will iterate through these 7 products.
    for prod_id in range(7):
        rank_offset = prod_id * 7
        
        # Determine A blocks and signs
        A_blocks = []
        if prod_id == 0: A_blocks = [(A_BL, 1), (A_BRR, 1)]
        elif prod_id == 1: A_blocks = [(A_BLL, 1), (A_BRR, 1)]
        elif prod_id == 2: A_blocks = [(A_BL, 1)]
        elif prod_id == 3: A_blocks = [(A_BRR, 1)]
        elif prod_id == 4: A_blocks = [(A_BL, 1), (A_BR, 1)]
        elif prod_id == 5: A_blocks = [(A_BLL, 1), (A_BL, -1)]
        elif prod_id == 6: A_blocks = [(A_BR, 1), (A_BRR, -1)]
        
        for indices, sign in A_blocks:
            # Add sign * U222 to U444 at these rows
            for r, idx in enumerate(indices):
                U444[idx, rank_offset:rank_offset+7] += sign * U222[r]
                
        # Determine B blocks and signs
        B_blocks = []
        if prod_id == 0: B_blocks = [(B_BL, 1), (B_BRR, 1)]
        elif prod_id == 1: B_blocks = [(B_BL, 1)]
        elif prod_id == 2: B_blocks = [(B_BR, 1), (B_BRR, -1)]
        elif prod_id == 3: B_blocks = [(B_BLL, 1), (B_BL, -1)]
        elif prod_id == 4: B_blocks = [(B_BRR, 1)]
        elif prod_id == 5: B_blocks = [(B_BL, 1), (B_BR, 1)]
        elif prod_id == 6: B_blocks = [(B_BLL, 1), (B_BRR, 1)]
        
        for indices, sign in B_blocks:
            for r, idx in enumerate(indices):
                V444[idx, rank_offset:rank_offset+7] += sign * V222[r]
                
        # Determine C blocks and signs
        C_blocks = []
        if prod_id == 0: C_blocks = [(C_BL, 1), (C_BRR, 1)]
        elif prod_id == 1: C_blocks = [(C_BLL, 1), (C_BL, -1)]
        elif prod_id == 2: C_blocks = [(C_BR, 1), (C_BRR, -1)]
        elif prod_id == 3: C_blocks = [(C_BLL, 1), (C_BL, 1)]
        elif prod_id == 4: C_blocks = [(C_BR, 1), (C_BL, -1)]
        elif prod_id == 5: C_blocks = [(C_BRR, 1)]
        elif prod_id == 6: C_blocks = [(C_BRR, -1)]
        
        for indices, sign in C_blocks:
            for r, idx in enumerate(indices):
                W444[idx, rank_offset:rank_offset+7] += sign * W222[r]
    
    # Build 4x4x2 Strassen Tensors
    # 4 independent 2x2x2 multiplications
    rank_442 = 28
    U442 = np.zeros((16, rank_442), dtype=int)
    V442 = np.zeros((8, rank_442), dtype=int) # B is 4x2, so 8 elements
    W442 = np.zeros((8, rank_442), dtype=int) # C is 4x2, so 8 elements
    
    # B is 4x2. Blocks:
    # B_T: rows 0,1; cols 0,1. Indices: 0,1,4,5 (in 4x2 grid? No, 4x2 grid has 8 elements).
    # Row 0: 0,1
    # Row 1: 2,3
    # Row 2: 4,5
    # Row 3: 6,7
    # B_T (top 2x2): rows 0,1; cols 0,1. Indices: 0,1,2,3.
    # B_B (bot 2x2): rows 2,3; cols 0,1. Indices: 4,5,6,7.
    
    # C is 4x2. Blocks:
    # C_T (top 2x2): rows 0,1; cols 0,1. Indices: 0,1,2,3.
    # C_B (bot 2x2): rows 2,3; cols 0,1. Indices: 4,5,6,7.
    
    B_T = [0,1,2,3]
    B_B = [4,5,6,7]
    
    C_T = [0,1,2,3]
    C_B = [4,5,6,7]
    
    # Products:
    # P1: A_BL * B_T -> C_T
    # P2: A_BR * B_B -> C_T
    # P3: A_BLL * B_T -> C_B
    # P4: A_BRR * B_B -> C_B
    
    prod_defs_442 = [
        (A_BL, B_T, C_T),
        (A_BR, B_B, C_T),
        (A_BLL, B_T, C_B),
        (A_BRR, B_B, C_B)
    ]
    
    for i, (A_idx, B_idx, C_idx) in enumerate(prod_defs_442):
        rank_offset = i * 7
        
        for r, idx in enumerate(A_idx):
            U442[idx, rank_offset:rank_offset+7] += U222[r]
            
        for r, idx in enumerate(B_idx):
            V442[idx, rank_offset:rank_offset+7] += V222[r]
            
        for r, idx in enumerate(C_idx):
            W442[idx, rank_offset:rank_offset+7] += W222[r]
            
    # Assemble Global U, V, W for 4x4x6
    rank = rank_444 + rank_442 # 49 + 28 = 77
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int) # 4x6 = 24
    W = np.zeros((24, rank), dtype=int) # 6x4 = 24
    
    # Copy U444 and U442
    U[:, :rank_444] = U444
    U[:, rank_444:] = U442
    
    # Copy V444 and V442
    # V444 is 16x49. Corresponds to B_L (first 4 cols of B, 4x4).
    # Global V rows for B_L: k in [0,4).
    # Index j*6 + k.
    # For j in 0..3, k in 0..3.
    # We place V444 into these rows.
    for j in range(4):
        for k in range(4):
            global_row = j * 6 + k
            V444_local_row = j * 4 + k
            V[global_row, :rank_444] = V444[V444_local_row, :rank_444]
            
    # V442 is 8x28. Corresponds to B_R (last 2 cols of B, 4x2).
    # Global V rows for B_R: k in [4,6).
    # B_R indices:
    # B_R is 4x2.
    # Row 0: cols 4,5.
    # Row 1: cols 4,5.
    # etc.
    # Local V442 indices:
    # 0,1,2,3 correspond to B_T (rows 0,1 of B_R).
    # 4,5,6,7 correspond to B_B (rows 2,3 of B_R).
    
    for j in range(4):
        for k in range(2):
            global_row = j * 6 + (4 + k)
            # Map local row
            if j < 2:
                local_row = j * 2 + k # 0,1,2,3
            else:
                local_row = 4 + (j-2) * 2 + k # 4,5,6,7
            V[global_row, rank_444:rank_444+rank_442] = V442[local_row, rank_442] if False else V442[local_row, :]
            
    # Copy W444 and W442
    # W444 is 16x49. Corresponds to C_L (first 4 cols of C, 4x4).
    # Global W rows for C_L: k in [0,4).
    # Index k*4 + i.
    for k in range(4):
        for i in range(4):
            global_row = k * 4 + i
            W444_local_row = k * 4 + i
            W[global_row, :rank_444] = W444[W444_local_row, :rank_444]
            
    # W442 is 8x28. Corresponds to C_R (last 2 cols of C, 4x2).
    # Global W rows for C_R: k in [4,6).
    for k in range(2):
        for i in range(4):
            global_row = (4 + k) * 4 + i
            # Map local row
            # C_R is 4x2.
            # Local indices 0-3 are C_T (top 2 rows of C_R, i.e., i=0,1? No, i is column in C? No, i is row in C?).
            # W index is k*n + i.
            # C_R is 4 rows (i=0..3) by 2 cols (k=4,5).
            # Local C_T is top 2 rows (i=0,1).
            # Local C_B is bot 2 rows (i=2,3).
            
            if i < 2:
                local_row = i * 2 + k
            else:
                local_row = 4 + (i-2) * 2 + k
            W[global_row, rank_444:rank_444+rank_442] = W442[local_row, :]
            
    # Convert to list of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
