import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # Rank will be 49 (for 4x4x4) + 16 (for 4x4x1) = 65
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # --- Part 1: 4x4x4 Multiplication (Rank 49) ---
    # We use recursive Strassen.
    # We need to generate linear forms for A (16 vars), B (16 vars for first 4 cols), C (16 vars for first 4 cols)
    
    # We will build the 49 terms by iterating over the 7 block multiplications
    # and expanding each into 7 scalar multiplications using 2x2 Strassen.
    
    # Indices for A: i*4+j
    # Indices for B: j*4+k (k in 0..3)
    # Indices for C: k*4+i (k in 0..3)
    
    # Define the 7 block operations for 4x4x4
    # Each operation is defined by:
    # A_block_linear_combos: list of (block_row, block_col, coeff)
    # B_block_linear_combos: list of (block_row, block_col, coeff)
    # C_block_contributions: list of (block_row, block_col, coeff)
    
    # Block indices: (r, c) where r,c in 0..1
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    ops = [
        # M1 = (A00 + A11) (B00 + B11)
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)] # C00 += M1, C11 += M1
        },
        # M2 = (A10 + A11) B00
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)] # C10 += M2, C11 -= M2
        },
        # M3 = A00 (B01 - B11)
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)] # C01 += M3, C11 += M3
        },
        # M4 = A11 (B10 - B00)
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)] # C00 += M4, C10 += M4
        },
        # M5 = (A00 + A01) B11
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)] # C00 -= M5, C01 += M5
        },
        # M6 = (A10 - A00) (B00 + B01)
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)] # C11 += M6
        },
        # M7 = (A01 - A11) (B10 + B11)
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)] # C00 += M7
        }
    ]
    
    # For each op, we compute the 2x2 product using Strassen (7 terms)
    # The inputs to the 2x2 Strassen are linear combinations of the 2x2 blocks.
    # We need to map the 2x2 Strassen terms to the global indices.
    
    # 2x2 Strassen terms definitions (local indices 0,1 for rows/cols)
    # M1_loc = (A00_loc + A11_loc) (B00_loc + B11_loc)
    # M2_loc = (A10_loc + A11_loc) B00_loc
    # M3_loc = A00_loc (B01_loc - B11_loc)
    # M4_loc = A11_loc (B10_loc - B00_loc)
    # M5_loc = (A00_loc + A01_loc) B11_loc
    # M6_loc = (A10_loc - A00_loc) (B00_loc + B01_loc)
    # M7_loc = (A01_loc - A11_loc) (B10_loc + B11_loc)
    
    # Local C contributions:
    # C00_loc = M1_loc + M4_loc - M5_loc + M7_loc
    # C01_loc = M3_loc + M5_loc
    # C10_loc = M2_loc + M4_loc
    # C11_loc = M1_loc - M2_loc + M3_loc + M6_loc
    
    # We will iterate over the 7 global ops.
    # For each global op, we have linear forms for the 4 A-blocks and 4 B-blocks.
    # We construct the linear forms for the 7 local A-terms and 7 local B-terms.
    # Then we iterate over the 7 local Strassen terms.
    # For each local term, we have a linear form for A (sum of 2x2 block elements) and B.
    # We add a global term with these forms.
    # The W form is determined by which local C element this term contributes to,
    # and which global C block that local C element belongs to.
    
    for op in ops:
        # Build linear forms for the 4 A-blocks involved in this op
        # A_blocks is a dict: (r,c) -> list of (global_i, global_j, coeff)
        # Actually, we just need the linear form for the whole 2x2 matrix of the op.
        # Let's represent the linear form for a 2x2 matrix as a dict of (local_r, local_c) -> list of global indices
        # Or simpler: just build the final U, V vectors for each of the 7 local terms.
        
        # First, determine the global linear forms for the 4 A-blocks and 4 B-blocks
        # A_block(r,c) corresponds to global A rows 2r..2r+1, cols 2c..2c+1
        # B_block(r,c) corresponds to global B rows 2r..2r+1, cols 2c..2c+1
        
        # We need to combine these to form the inputs for the 7 local Strassen terms.
        
        # Let's define the 7 local terms' A and B linear forms in terms of the 4 blocks.
        local_ops = [
            # (A_blocks, B_blocks)
            # A is list of (block_rc, coeff), B is list of (block_rc, coeff)
            {
                'A': [((0,0), 1), ((1,1), 1)],
                'B': [((0,0), 1), ((1,1), 1)]
            },
            {
                'A': [((1,0), 1), ((1,1), 1)],
                'B': [((0,0), 1)]
            },
            {
                'A': [((0,0), 1)],
                'B': [((0,1), 1), ((1,1), -1)]
            },
            {
                'A': [((1,1), 1)],
                'B': [((1,0), 1), ((0,0), -1)]
            },
            {
                'A': [((0,0), 1), ((0,1), 1)],
                'B': [((1,1), 1)]
            },
            {
                'A': [((1,0), 1), ((0,0), -1)],
                'B': [((0,0), 1), ((0,1), 1)]
            },
            {
                'A': [((0,1), 1), ((1,1), -1)],
                'B': [((1,0), 1), ((1,1), 1)]
            }
        ]
        
        # Local C contributions for each of the 7 local terms
        # local_term_idx -> list of (local_c_rc, coeff)
        local_c_contribs = [
            [((0,0), 1), ((1,1), 1)], # M1_loc
            [((1,0), 1), ((1,1), -1)], # M2_loc
            [((0,1), 1), ((1,1), 1)], # M3_loc
            [((0,0), 1), ((1,0), 1)], # M4_loc
            [((0,0), -1), ((0,1), 1)], # M5_loc
            [((1,1), 1)], # M6_loc
            [((0,0), 1)] # M7_loc
        ]
        
        # For each local term (0..6)
        for l_idx in range(7):
            l_op = local_ops[l_idx]
            
            # Build U vector (size 16)
            u_vec = [0] * 16
            for (br, bc), coeff in l_op['A']:
                # This block is combined with the global op's A blocks
                # We need to find which global blocks are in l_op['A'] and combine with op['A']
                # Wait, the structure is:
                # Global Op defines linear combos of 4 blocks.
                # Local Op defines linear combos of those 4 blocks (conceptually).
                # Actually, the Local Op acts on the *result* of the Global Op's block selection?
                # No.
                # The Global Op M_k = A'_k B'_k.
                # A'_k is a 2x2 matrix formed by linear combos of the 4 A-blocks.
                # B'_k is a 2x2 matrix formed by linear combos of the 4 B-blocks.
                # Then we apply Strassen to A'_k B'_k.
                # So the inputs to the local Strassen are the entries of A'_k and B'_k.
                
                # We need to expand A'_k and B'_k into global indices.
                # A'_k[r][c] = sum_{(br,bc) in op['A']} coeff * A_block[br][bc][r][c]
                # This is getting complex to track symbolically in nested loops.
                
                # Alternative: Just build the full linear form for each of the 7 local A-terms and 7 local B-terms.
                # The 7 local A-terms are linear combos of the 4 entries of A'_k.
                # The 4 entries of A'_k are linear combos of the 16 global A entries.
                
                pass # We will do this in the next step

    # Let's restart the generation logic to be clearer.
    # We will generate the 49 terms directly.
    
    U = []
    V = []
    W = []
    
    for op_idx, op in enumerate(ops):
        # Determine the linear forms for the 4 entries of A'_op and B'_op
        # A'_op is a 2x2 matrix.
        # A'_op[r][c] = sum over global A blocks
        
        # We will create 4 vectors for A'_op entries (r,c) in 0..1
        A_prime_entries = [[0]*16 for _ in range(4)] # index r*2+c
        B_prime_entries = [[0]*20 for _ in range(4)] # index r*2+c
        
        for (br, bc), coeff in op['A']:
            for r in range(2):
                for c in range(2):
                    global_i = 2*br + r
                    global_j = 2*bc + c
                    idx = global_i * 4 + global_j
                    A_prime_entries[r*2+c][idx] += coeff
        
        for (br, bc), coeff in op['B']:
            for r in range(2):
                for c in range(2):
                    global_j = 2*br + r
                    global_k = 2*bc + c
                    idx = global_j * 4 + global_k # B is 4x5, but we only use first 4 cols
                    B_prime_entries[r*2+c][idx] += coeff
        
        # Now apply 2x2 Strassen to A_prime and B_prime
        # 7 local terms
        local_strassen = [
            # (A_entries, B_entries)
            # A is list of (entry_idx, coeff), B is list of (entry_idx, coeff)
            # entry_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            {
                'A': [(0, 1), (3, 1)],
                'B': [(0, 1), (3, 1)]
            },
            {
                'A': [(2, 1), (3, 1)],
                'B': [(0, 1)]
            },
            {
                'A': [(0, 1)],
                'B': [(1, 1), (3, -1)]
            },
            {
                'A': [(3, 1)],
                'B': [(2, 1), (0, -1)]
            },
            {
                'A': [(0, 1), (1, 1)],
                'B': [(3, 1)]
            },
            {
                'A': [(2, 1), (0, -1)],
                'B': [(0, 1), (1, 1)]
            },
            {
                'A': [(1, 1), (3, -1)],
                'B': [(2, 1), (3, 1)]
            }
        ]
        
        # Local C contributions for these 7 terms
        # Term 0 (M1): C00 += 1, C11 += 1
        # Term 1 (M2): C10 += 1, C11 -= 1
        # Term 2 (M3): C01 += 1, C11 += 1
        # Term 3 (M4): C00 += 1, C10 += 1
        # Term 4 (M5): C00 -= 1, C01 += 1
        # Term 5 (M6): C11 += 1
        # Term 6 (M7): C00 += 1
        local_c_map = [
            [((0,0), 1), ((1,1), 1)],
            [((1,0), 1), ((1,1), -1)],
            [((0,1), 1), ((1,1), 1)],
            [((0,0), 1), ((1,0), 1)],
            [((0,0), -1), ((0,1), 1)],
            [((1,1), 1)],
            [((0,0), 1)]
        ]
        
        for l_idx in range(7):
            l_op = local_strassen[l_idx]
            
            # Build U vector
            u_vec = [0] * 16
            for (e_idx, coeff) in l_op['A']:
                for i in range(16):
                    u_vec[i] += coeff * A_prime_entries[e_idx][i]
            
            # Build V vector
            v_vec = [0] * 20
            for (e_idx, coeff) in l_op['B']:
                for i in range(20):
                    v_vec[i] += coeff * B_prime_entries[e_idx][i]
            
            # Build W vector
            w_vec = [0] * 20
            for (c_r, c_c), coeff in local_c_map[l_idx]:
                # This local C entry (c_r, c_c) belongs to global C block (op's block? No)
                # Wait. The global op M_k produces a 2x2 matrix.
                # This 2x2 matrix is added to global C blocks.
                # The local C entry (c_r, c_c) is an element of the 2x2 result of M_k.
                # Which global C block does M_k contribute to?
                # M_k is the k-th term in the 4x4 Strassen expansion.
                # The 4x4 Strassen expansion results are:
                # C00_block = M1 + M4 - M5 + M7
                # C01_block = M3 + M5
                # C10_block = M2 + M4
                # C11_block = M1 - M2 + M3 + M6
                
                # So, for a given global op index (0..6 corresponding to M1..M7),
                # we need to know which global C blocks it contributes to.
                
                # Let's define global_c_map for op_idx (0..6)
                # op 0 (M1): C00 (+1), C11 (+1)
                # op 1 (M2): C10 (+1), C11 (-1)
                # op 2 (M3): C01 (+1), C11 (+1)
                # op 3 (M4): C00 (+1), C10 (+1)
                # op 4 (M5): C00 (-1), C01 (+1)
                # op 5 (M6): C11 (+1)
                # op 6 (M7): C00 (+1)
                
                global_c_map = [
                    [((0,0), 1), ((1,1), 1)],
                    [((1,0), 1), ((1,1), -1)],
                    [((0,1), 1), ((1,1), 1)],
                    [((0,0), 1), ((1,0), 1)],
                    [((0,0), -1), ((0,1), 1)],
                    [((1,1), 1)],
                    [((0,0), 1)]
                ]
                
                # The local C entry (c_r, c_c) is at position (c_r, c_c) in the 2x2 block.
                # The global C block is determined by op_idx.
                # But wait, the global C block is a 2x2 block itself.
                # So the global C entry is at (2*gc_r + c_r, 2*gc_c + c_c).
                
                for (gc_r, gc_c), g_coeff in global_c_map[op_idx]:
                    global_i = 2*gc_r + c_r
                    global_k = 2*gc_c + c_c
                    # W index: k*n + i
                    w_idx = global_k * 4 + global_i
                    w_vec[w_idx] += coeff * g_coeff
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)

    # --- Part 2: 4x4x1 Multiplication (Rank 16) ---
    # C_i4 = sum_j A_ij B_j4
    # 16 terms: for each i, j
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 20
            w_vec = [0] * 20
            
            u_idx = i * 4 + j
            v_idx = j * 4 + 4 # B row j, col 4
            w_idx = 4 * 4 + i # C row i, col 4 -> k=4, i=i -> 4*4+i
            
            u_vec[u_idx] = 1
            v_vec[v_idx] = 1
            w_vec[w_idx] = 1
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
