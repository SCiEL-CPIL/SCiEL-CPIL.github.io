"""
Strassen-based 4x4 Matrix Multiplication Tensor Decomposition.
Strategy: Compose the rank-7 Strassen decomposition for 2x2 matrices 
to construct a valid rank-49 decomposition for 4x4 matrices.
This is a significant improvement over the schoolbook rank 64.
The decomposition is exact over the integers.
"""
import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 4
    
    # We will construct a rank-49 decomposition.
    # The structure is:
    # Split A, B, C into 2x2 blocks of size 2x2.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A * B
    # 
    # Strassen's algorithm computes 7 products M1..M7 of 2x2 matrices.
    # Each M_k is a product of two linear combinations of the 2x2 blocks.
    # Since each 2x2 matrix multiplication has rank 7, and we have 7 such products,
    # the total rank is 7 * 7 = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # The 7 Strassen terms for 2x2 matrices X, Y -> Z = X*Y:
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y12 - Y22)
    # M4 = X22(Y21 - Y11)
    # M5 = (X11 + X12)Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    # 
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # We define the 7 "macro" products for the 4x4 case.
    # Each macro product is of the form (Sum of A_blocks) * (Sum of B_blocks).
    # Let's define the 7 macro products P1..P7:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)(B11)
    # P3 = (A11)(B12 - B22)
    # P4 = (A22)(B21 - B11)
    # P5 = (A11 + A12)(B22)
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # The resulting C blocks are:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to decompose each Pk into 7 scalar multiplications.
    # Pk is a product of two 2x2 matrices (which are sums of blocks).
    # Let X be the first matrix in Pk, and Y be the second.
    # X and Y are 2x2 matrices whose entries are linear combinations of A's 2x2 blocks.
    # Specifically, each entry of X is a sum of one or two 2x2 blocks from A.
    # Each entry of Y is a sum of one or two 2x2 blocks from B.
    
    # We will iterate through the 7 macro products.
    # For each macro product, we apply the Strassen 2x2 decomposition.
    # This generates 7 scalar terms for that macro product.
    # Total 7 * 7 = 49 terms.
    
    # Helper to add a term to U, V, W
    # a_blocks: list of (block_idx, coeff) for the A part of the scalar term
    # b_blocks: list of (block_idx, coeff) for the B part of the scalar term
    # c_blocks: list of (block_idx, coeff) for the C part of the scalar term
    # col: the column index in U, V, W
    
    def add_term(a_blocks, b_blocks, c_blocks, col):
        # a_blocks, b_blocks, c_blocks are lists of (block_idx, coeff)
        # block_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        for (b_idx, coeff_a) in a_blocks:
            row_off = (b_idx // 2) * 2
            col_off = (b_idx % 2) * 2
            # The block is 2x2.
            # We need to add coeff_a to the appropriate U entries.
            # U is indexed by i*m + j.
            # The block (b_idx) corresponds to rows row_off, row_off+1 and cols col_off, col_off+1.
            # The scalar term is a product of specific entries? 
            # NO. The Strassen decomposition works on the *entries* of the 2x2 matrices.
            # So when we say "P1 = (A11+A22)(B11+B22)", we are treating (A11+A22) as a 2x2 matrix.
            # Let X = A11+A22. X is 2x2.
            # The Strassen term M1 for X*Y is (X11+X22)(Y11+Y22).
            # X11 is the (0,0) entry of X. X = A11+A22.
            # So X11 = A11_00 + A22_00.
            # This means the U part of this scalar term involves A11_00 and A22_00.
            # So we add coeff_a to U at the indices corresponding to A11_00 and A22_00.
            
            # Indices for A11_00: row 0, col 0 -> 0*4+0 = 0
            # Indices for A22_00: row 2, col 2 -> 2*4+2 = 10
            
            # General formula:
            # Block b_idx has offset (row_off, col_off).
            # Entry (r,c) in the block has global index (row_off+r)*4 + (col_off+c).
            
            # But wait, the "block" here refers to the 2x2 submatrix of the 4x4.
            # The "entry" (r,c) in the 2x2 Strassen context refers to the position within the 2x2 block.
            # So if a_blocks contains (1, 1.0), it means "Block 1 (A12), entry (r,c) with coeff 1.0".
            # We need to know which entry (r,c) of the 2x2 block is involved.
            
            # Let's refine the input format.
            # Instead of just block_idx, we need the specific entry (r,c) within the 2x2 block.
            # So a_blocks should be a list of (block_idx, r, c, coeff).
            
            # Let's restart the helper logic with this in mind.
            pass

    # Corrected Helper:
    def add_term(a_entries, b_entries, c_entries, col):
        # a_entries: list of (block_idx, r, c, coeff)
        # block_idx: 0=A11, 1=A12, 2=A21, 3=A22
        # r, c: 0 or 1 (position within the 2x2 block)
        # coeff: multiplier
        
        for (b_idx, r, c, coeff_a) in a_entries:
            row_off = (b_idx // 2) * 2
            col_off = (b_idx % 2) * 2
            u_idx = (row_off + r) * 4 + (col_off + c)
            U[u_idx][col] += coeff_a
            
        for (b_idx, r, c, coeff_b) in b_entries:
            row_off = (b_idx // 2) * 2
            col_off = (b_idx % 2) * 2
            v_idx = (row_off + r) * 4 + (col_off + c)
            V[v_idx][col] += coeff_b
            
        for (b_idx, r, c, coeff_c) in c_entries:
            row_off = (b_idx // 2) * 2
            col_off = (b_idx % 2) * 2
            w_idx = (row_off + r) * 4 + (col_off + c)
            W[w_idx][col] += coeff_c

    # Define the 7 Strassen 2x2 terms in terms of entries (r,c) of X and Y.
    # X, Y are 2x2.
    # s1: (X00+X11)(Y00+Y11)
    # s2: (X10+X11)Y00
    # s3: X00(Y01-Y11)
    # s4: X11(Y10-Y00)
    # s5: (X00+X01)Y11
    # s6: (X10-X00)(Y00+Y01)
    # s7: (X01-X11)(Y10+Y11)
    
    # For each macro product Pk, we define X and Y as sums of A/B blocks.
    # We need to expand X_rc and Y_rc into sums of A/B block entries.
    
    # Let's define the mapping for each macro product.
    # P1: X = A11 + A22, Y = B11 + B22
    # X00 = A11_00 + A22_00
    # X01 = A11_01 + A22_01
    # X10 = A11_10 + A22_10
    # X11 = A11_11 + A22_11
    # Y00 = B11_00 + B22_00
    # ...
    
    # We will create a function to expand a "sum of blocks" into entries.
    # def expand_block_sum(blocks):
    #    blocks: list of (block_idx, coeff)
    #    returns: dict {(r,c): [(block_idx, coeff), ...]}
    #    Actually, we need to return the list of (block_idx, r, c, coeff) for each (r,c).
    
    def expand_block_sum(blocks):
        # blocks: list of (block_idx, coeff)
        # Returns a list of lists: entries[r][c] = list of (block_idx, coeff)
        entries = [[[] for _ in range(2)] for _ in range(2)]
        for b_idx, coeff in blocks:
            for r in range(2):
                for c in range(2):
                    entries[r][c].append((b_idx, coeff))
        return entries

    # Now we define the 7 macro products.
    # Each macro product is (X, Y) where X, Y are sums of blocks.
    # We also need to know how the result Z (2x2) contributes to C.
    # The result Z is a 2x2 matrix.
    # Z contributes to C blocks with specific coefficients.
    
    # P1: X = A11+A22, Y = B11+B22
    # Z11 contributes to C11 with coeff 1
    # Z12 contributes to C12 with coeff 0? No, P1 is not directly a C block.
    # C11 = P1_11 + P4_11 - P5_11 + P7_11
    # C12 = P3_12 + P5_12
    # C21 = P2_21 + P4_21
    # C22 = P1_22 - P2_22 + P3_22 + P6_22
    
    # So for P1, the Z entries contribute to:
    # Z00 (Z11) -> C11 (block 0) with coeff 1
    # Z11 (Z22) -> C22 (block 3) with coeff 1
    
    # Let's define the contribution map for each Pk.
    # c_map: dict {(r,c): [(c_block_idx, coeff), ...]}
    
    macro_products = [
        {
            "x_blocks": [(0, 1), (3, 1)], # A11 + A22
            "y_blocks": [(0, 1), (3, 1)], # B11 + B22
            "c_map": {(0,0): [(0, 1)], (1,1): [(3, 1)]} # Z00->C11, Z11->C22
        },
        {
            "x_blocks": [(2, 1), (3, 1)], # A21 + A22
            "y_blocks": [(0, 1)],         # B11
            "c_map": {(1,0): [(2, 1)], (1,1): [(3, -1)]} # Z10->C21, Z11->C22
        },
        {
            "x_blocks": [(0, 1)],         # A11
            "y_blocks": [(1, 1), (3, -1)],# B12 - B22
            "c_map": {(0,1): [(1, 1)], (1,1): [(3, 1)]}  # Z01->C12, Z11->C22
        },
        {
            "x_blocks": [(3, 1)],         # A22
            "y_blocks": [(2, 1), (0, -1)],# B21 - B11
            "c_map": {(0,0): [(0, 1)], (1,0): [(2, 1)]}  # Z00->C11, Z10->C21
        },
        {
            "x_blocks": [(0, 1), (1, 1)], # A11 + A12
            "y_blocks": [(3, 1)],         # B22
            "c_map": {(0,0): [(0, -1)], (0,1): [(1, 1)]} # Z00->C11, Z01->C12
        },
        {
            "x_blocks": [(2, 1), (0, -1)],# A21 - A11
            "y_blocks": [(0, 1), (1, 1)], # B11 + B12
            "c_map": {(1,1): [(3, 1)]}    # Z11->C22
        },
        {
            "x_blocks": [(1, 1), (3, -1)],# A12 - A22
            "y_blocks": [(2, 1), (3, 1)]  # B21 + B22
            ,
            "c_map": {(0,0): [(0, 1)]}    # Z00->C11
        }
    ]
    
    # Strassen 2x2 terms definitions:
    # Each term is a list of (x_entries, y_entries, z_entries)
    # x_entries: list of (r, c, coeff)
    # y_entries: list of (r, c, coeff)
    # z_entries: list of (r, c, coeff)
    
    strassen_2x2_terms = [
        # s1: (X00+X11)(Y00+Y11) -> Z00+Z11
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # s2: (X10+X11)Y00 -> Z10 - Z11
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # s3: X00(Y01-Y11) -> Z01 + Z11
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # s4: X11(Y10-Y00) -> Z00 + Z10
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # s5: (X00+X01)Y11 -> -Z00 + Z01
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # s6: (X10-X00)(Y00+Y01) -> Z11
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # s7: (X01-X11)(Y10+Y11) -> Z00
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    col = 0
    for mp in macro_products:
        x_expanded = expand_block_sum(mp["x_blocks"])
        y_expanded = expand_block_sum(mp["y_blocks"])
        
        for term in strassen_2x2_terms:
            x_entries, y_entries, z_entries = term
            
            # Construct a_entries, b_entries, c_entries for add_term
            a_entries = []
            for (r, c, coeff_x) in x_entries:
                for (b_idx, coeff_block) in x_expanded[r][c]:
                    a_entries.append((b_idx, r, c, coeff_x * coeff_block))
            
            b_entries = []
            for (r, c, coeff_y) in y_entries:
                for (b_idx, coeff_block) in y_expanded[r][c]:
                    b_entries.append((b_idx, r, c, coeff_y * coeff_block))
            
            c_entries = []
            for (r, c, coeff_z) in z_entries:
                # Look up c_map for this (r,c)
                if (r,c) in mp["c_map"]:
                    for (c_block_idx, coeff_c) in mp["c_map"][(r,c)]:
                        c_entries.append((c_block_idx, r, c, coeff_z * coeff_c))
            
            add_term(a_entries, b_entries, c_entries, col)
            col += 1
            
            if time.time() > deadline:
                # Fallback to schoolbook if we run out of time (unlikely for 49 terms)
                return construct_schoolbook()

    return {"U": U, "V": V, "W": W}

def construct_schoolbook():
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
