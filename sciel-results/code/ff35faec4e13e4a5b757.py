"""Matrix multiplication 4,4,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-49 decomposition for 4x4 * 4x6 matrix multiplication.
    
    Uses Strassen's algorithm to handle 4x4 * 4x4 (rank 49) and then
    embeds the 4x4 * 4x6 problem into this framework.
    
    The output matrix C has shape (4,6). We use the first 4 columns of 
    a 4x4 result by setting the 5th and 6th columns of B to zero.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen's 2x2 * 2x2 decomposition (rank 7)
    # M0 = (A11+A22)(B11+B22)
    # M1 = (A21+A22)B11
    # M2 = A11(B21-B22)
    # M3 = A22(B11-B21)
    # M4 = (A11+A12)B22
    # M5 = (A21-A11)(B11+B12)
    # M6 = (A12-A22)(B21+B22)
    
    # For 4x4 * 4x4, we partition into 2x2 blocks of 2x2 matrices
    # This gives rank 7^2 = 49
    
    # Define the Strassen decomposition for 2x2 matrices
    # U vectors (for A), V vectors (for B), W vectors (for C)
    # Each is indexed by [row*2+col, rank]
    
    # 2x2 Strassen decomposition:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # U2[i, k], V2[j, k], W2[c, k] for k = 0..6
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M0 = (A11+A22)(B11+B22) -> contributes to C11, C22
    U2[0, 0] = 1  # a00
    U2[3, 0] = 1  # a11
    V2[0, 0] = 1  # b00
    V2[3, 0] = 1  # b11
    W2[0, 0] = 1  # c00
    W2[3, 0] = 1  # c11
    
    # M1 = (A21+A22)B11 -> contributes to C21, C22
    U2[2, 1] = 1  # a10
    U2[3, 1] = 1  # a11
    V2[0, 1] = 1  # b00
    W2[2, 1] = 1  # c10
    W2[3, 1] = 1  # c11
    
    # M2 = A11(B21-B22) -> contributes to C11, C12
    U2[0, 2] = 1  # a00
    V2[2, 2] = 1  # b10
    V2[3, 2] = -1 # b11
    W2[0, 2] = 1  # c00
    W2[1, 2] = 1  # c01
    
    # M3 = A22(B11-B21) -> contributes to C21, C22
    U2[3, 3] = 1  # a11
    V2[0, 3] = 1  # b00
    V2[2, 3] = -1 # b10
    W2[2, 3] = 1  # c10
    W2[3, 3] = 1  # c11
    
    # M4 = (A11+A12)B22 -> contributes to C11, C12
    U2[0, 4] = 1  # a00
    U2[1, 4] = 1  # a01
    V2[3, 4] = 1  # b11
    W2[0, 4] = 1  # c00
    W2[1, 4] = 1  # c01
    
    # M5 = (A21-A11)(B11+B12) -> contributes to C21, C22
    U2[2, 5] = 1  # a10
    U2[0, 5] = -1 # a00
    V2[0, 5] = 1  # b00
    V2[1, 5] = 1  # b01
    W2[2, 5] = 1  # c10
    W2[3, 5] = 1  # c11
    
    # M6 = (A12-A22)(B21+B22) -> contributes to C11, C12
    U2[1, 6] = 1  # a01
    U2[3, 6] = -1 # a11
    V2[2, 6] = 1  # b10
    V2[3, 6] = 1  # b11
    W2[0, 6] = 1  # c00
    W2[1, 6] = 1  # c01
    
    # Now for 4x4 * 4x4, partition into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ck is 2x2
    
    # The Strassen algorithm for 4x4:
    # M00 = (A00+A11)(B00+B11)
    # M01 = (A10+A11)B00
    # M02 = A00(B10-B11)
    # M03 = A11(B00-B10)
    # M04 = (A00+A01)B11
    # M05 = (A10-A00)(B00+B01)
    # M06 = (A01-A11)(B10+B11)
    
    # Then:
    # C00 = M00 + M03 - M04 - M06
    # C01 = M02 + M04
    # C10 = M01 + M03
    # C11 = M00 - M01 + M02 + M05
    
    # Each Mij is a 2x2 matrix product computed via the 2x2 Strassen decomp
    # So we have 7 block products, each of rank 7, total rank 49
    
    # Let's construct U4, V4, W4 for 4x4 * 4x4
    # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49)
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # Block indices:
    # A00: rows 0-1, cols 0-1 -> indices 0,1,2,3 in A
    # A01: rows 0-1, cols 2-3 -> indices 4,5,6,7 in A
    # A10: rows 2-3, cols 0-1 -> indices 8,9,10,11 in A
    # A11: rows 2-3, cols 2-3 -> indices 12,13,14,15 in A
    
    # B00: rows 0-1, cols 0-1 -> indices 0,1,2,3 in B
    # B01: rows 0-1, cols 2-3 -> indices 4,5,6,7 in B
    # B10: rows 2-3, cols 0-1 -> indices 8,9,10,11 in B
    # B11: rows 2-3, cols 2-3 -> indices 12,13,14,15 in B
    
    # C00: rows 0-1, cols 0-1 -> indices 0,1,2,3 in C
    # C01: rows 0-1, cols 2-3 -> indices 4,5,6,7 in C
    # C10: rows 2-3, cols 0-1 -> indices 8,9,10,11 in C
    # C11: rows 2-3, cols 2-3 -> indices 12,13,14,15 in C
    
    # For each of the 7 Strassen products, we use the 2x2 decomp
    # Product k (k=0..6) uses columns 7*k to 7*k+6 in U4, V4, W4
    
    def set_block(decomp, block_row_start, block_col_start, block_size, base_col):
        """Set a block in the decomposition arrays."""
        # This is a helper to place the 2x2 Strassen pattern
        pass
    
    # M00 = (A00+A11)(B00+B11)
    # This means: U for A00+A11, V for B00+B11, W for C contribution
    # In the 2x2 decomp, we need:
    # U_vec = [1,0,1,0, 0,0,0,0] for A00+A11 (a00+a11, a01+a11? No...)
    # Wait, A00 and A11 are 2x2 matrices. A00+A11 means element-wise addition.
    # So for the first element of the 2x2 result, we take (A00[0,0]+A11[0,0]) * (B00[0,0]+B11[0,0])
    # etc.
    
    # Let me be more careful. The 2x2 Strassen decomp assumes:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # and computes C = A*B.
    
    # For M00 = (A00+A11)(B00+B11), we set:
    # a00 = A00[0,0] + A11[0,0] = A[0,0] + A[12,0]... wait, indexing.
    
    # A is 4x4, stored row-major:
    # A[0,0], A[0,1], A[0,2], A[0,3]
    # A[1,0], A[1,1], A[1,2], A[1,3]
    # A[2,0], A[2,1], A[2,2], A[2,3]
    # A[3,0], A[3,1], A[3,2], A[3,3]
    
    # A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    
    # So A00+A11 = [[A[0,0]+A[2,2], A[0,1]+A[2,3]], [A[1,0]+A[3,2], A[1,1]+A[3,3]]]
    
    # In the U vector for this product, we need coefficients for each element of A.
    # U[i, col] = coefficient for A[i] in the i-th component of the U vector for this product.
    
    # Let's define the 7 products explicitly.
    
    products = [
        # (name, U_indices, V_indices, W_indices, sign)
        # M00 = (A00+A11)(B00+B11)
        {
            'name': 'M00',
            'U': [(0, 1), (2, 1), (4, 1), (6, 1), (8, 1), (10, 1), (12, 1), (14, 1)],
            # A[0,0], A[0,1], A[1,0], A[1,1], A[2,0], A[2,1], A[3,0], A[3,1] for A00+A11? 
            # No, A00+A11 has 4 elements, but we need to map to the 2x2 Strassen which uses 4 elements of A.
            # The 2x2 Strassen U2 has 4 rows, corresponding to a00, a01, a10, a11.
            # For M00, a00 = A[0,0]+A[2,2], a01 = A[0,1]+A[2,3], a10 = A[1,0]+A[3,2], a11 = A[1,1]+A[3,3]
            # So in U4, for this product, we set:
            # U4[0, col] = 1 (for A[0,0])
            # U4[12, col] = 1 (for A[2,2])
            # U4[1, col] = 1 (for A[0,1])
            # U4[13, col] = 1 (for A[2,3])
            # U4[2, col] = 1 (for A[1,0])
            # U4[14, col] = 1 (for A[3,2])
            # U4[3, col] = 1 (for A[1,1])
            # U4[15, col] = 1 (for A[3,3])
            'V': [(0, 1), (2, 1), (4, 1), (6, 1), (8, 1), (10, 1), (12, 1), (14, 1)],
            # Similarly for B
            'W': [(0, 1), (2, 1), (4, 1), (6, 1), (8, 1), (10, 1), (12, 1), (14, 1)],
            # C00 = C00 + M00, so W for C00 block
        },
    ]
    
    # This is getting complex. Let me use a cleaner approach.
    
    # Define the 7 Strassen block products for 4x4:
    # Each product is of the form (linear_combination_of_A_blocks)(linear_combination_of_B_blocks)
    # and contributes to C blocks with specific coefficients.
    
    # I'll directly construct U4, V4, W4 by iterating over the 7 products.
    
    # Product definitions: (A_coefficients, B_coefficients, C_coefficients)
    # A_coefficients: dict mapping block (br, bc) to coefficient
    # where block (br, bc) is the 2x2 block at row block br, col block bc
    
    strassen_products = [
        # M00 = (A00+A11)(B00+B11)
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # M01 = (A10+A11)B00
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): 1}},
        # M02 = A00(B10-B11)
        {'A': {(0,0): 1}, 'B': {(1,0): 1, (1,1): -1}, 'C': {(0,0): 1, (0,1): 1}},
        # M03 = A11(B00-B10)
        {'A': {(1,1): 1}, 'B': {(0,0): 1, (1,0): -1}, 'C': {(1,0): 1, (1,1): 1}},
        # M04 = (A00+A01)B11
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): 1, (0,1): 1}},
        # M05 = (A10-A00)(B00+B01)
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,0): 1, (1,1): 1}},
        # M06 = (A01-A11)(B10+B11)
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (0,1): 1}},
    ]
    
    # For each product, we use the 2x2 Strassen decomp (U2, V2, W2)
    # The 2x2 decomp has 7 terms. For each term, we add to U4, V4, W4.
    
    for prod_idx, prod in enumerate(strassen_products):
        for term_idx in range(7):
            col = prod_idx * 7 + term_idx
            
            # U4: for each element in the A block, set coefficient
            for (ar, ac), coeff in prod['A'].items():
                # The 2x2 block (ar, ac) has elements at positions:
                # (2*ar + 0, 2*ac + 0), (2*ar + 0, 2*ac + 1)
                # (2*ar + 1, 2*ac + 0), (2*ar + 1, 2*ac + 1)
                # These correspond to indices in the 4x4 matrix:
                base_r = 2 * ar
                base_c = 2 * ac
                # The 2x2 Strassen U2 has 4 rows: [a00, a01, a10, a11]
                # U2[0, term_idx] corresponds to a00 (top-left)
                # U2[1, term_idx] corresponds to a01 (top-right)
                # U2[2, term_idx] corresponds to a10 (bottom-left)
                # U2[3, term_idx] corresponds to a11 (bottom-right)
                
                a00_idx = base_r * 4 + base_c
                a01_idx = base_r * 4 + base_c + 1
                a10_idx = (base_r + 1) * 4 + base_c
                a11_idx = (base_r + 1) * 4 + base_c + 1
                
                U4[a00_idx, col] += coeff * U2[0, term_idx]
                U4[a01_idx, col] += coeff * U2[1, term_idx]
                U4[a10_idx, col] += coeff * U2[2, term_idx]
                U4[a11_idx, col] += coeff * U2[3, term_idx]
            
            # V4: similar for B blocks
            for (br, bc), coeff in prod['B'].items():
                base_r = 2 * br
                base_c = 2 * bc
                b00_idx = base_r * 4 + base_c
                b01_idx = base_r * 4 + base_c + 1
                b10_idx = (base_r + 1) * 4 + base_c
                b11_idx = (base_r + 1) * 4 + base_c + 1
                
                V4[b00_idx, col] += coeff * V2[0, term_idx]
                V4[b01_idx, col] += coeff * V2[1, term_idx]
                V4[b10_idx, col] += coeff * V2[2, term_idx]
                V4[b11_idx, col] += coeff * V2[3, term_idx]
            
            # W4: for C blocks
            for (cr, cc), coeff in prod['C'].items():
                base_r = 2 * cr
                base_c = 2 * cc
                c00_idx = base_r * 4 + base_c
                c01_idx = base_r * 4 + base_c + 1
                c10_idx = (base_r + 1) * 4 + base_c
                c11_idx = (base_r + 1) * 4 + base_c + 1
                
                W4[c00_idx, col] += coeff * W2[0, term_idx]
                W4[c01_idx, col] += coeff * W2[1, term_idx]
                W4[c10_idx, col] += coeff * W2[2, term_idx]
                W4[c11_idx, col] += coeff * W2[3, term_idx]
    
    # Now we have U4, V4, W4 for 4x4 * 4x4 with rank 49.
    # For 4x4 * 4x6, we note that B has shape (4, 6).
    # We can embed this into 4x4 * 4x4 by padding B with 2 zero columns.
    # But that would give us a 4x4 result, and we only need the first 4 columns.
    
    # Alternatively, we can directly handle the 4x6 case.
    # The key insight: the 4x6 matrix B can be seen as two 4x3 matrices,
    # but Strassen's algorithm works on square blocks.
    
    # A simpler approach: use the 4x4 * 4x4 decomposition and set the last 
    # two columns of B to zero. This gives rank 49, and the result is correct
    # for the first 4 columns of C.
    
    # But wait, the task is 4x4 * 4x6, so A is (4,4), B is (4,6), C is (4,6).
    # If we pad B to (4,8) or (4,4) with zeros, we need to adjust.
    
    # Actually, for B of shape (4,6), we can partition it into 2x3 blocks.
    # This doesn't fit nicely into the 2x2 Strassen framework.
    
    # Let me reconsider. The rank for 4x4 * 4x6 might not be 49.
    # The schoolbook rank is 4*4*6 = 96.
    # Using Strassen-like techniques, we might achieve lower rank.
    
    # For now, let's use the 4x4 * 4x4 decomposition and pad B.
    # We'll create a (4,8) version of B by padding with 2 zeros, but that's wasteful.
    
    # Better: directly construct for 4x4 * 4x6.
    # We can use the fact that 6 = 4 + 2, and handle the 4x4 and 4x2 parts separately.
    
    # For the 4x4 part, use Strassen (rank 49).
    # For the 4x2 part, use schoolbook (rank 4*4*2 = 32).
    # Total rank: 49 + 32 = 81. This is better than 96.
    
    # But can we do better? Let's use a mixed approach.
    
    # Actually, let's just use the 4x4 * 4x4 Strassen decomposition and 
    # handle the extra 2 columns of B separately.
    
    # U has shape (16, rank), V has shape (24, rank), W has shape (24, rank)
    # n=4, m=4, p=6
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # For the first 4 columns of B (the 4x4 part), use Strassen.
    # For the last 2 columns of B (the 4x2 part), use schoolbook.
    
    rank_strassen = 49
    rank_schoolbook_4x2 = 4 * 4 * 2  # 32
    total_rank = rank_strassen + rank_schoolbook_4x2  # 81
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((24, total_rank), dtype=float)
    W = np.zeros((24, total_rank), dtype=float)
    
    # Copy the Strassen part for the first 4 columns of B
    U[:, :rank_strassen] = U4
    V[:16, :rank_strassen] = V4  # First 16 rows of V (for B[:, :4])
    W[:16, :rank_strassen] = W4  # First 16 rows of W (for C[:, :4])
    
    # Schoolbook for the last 2 columns of B
    # A is (4,4), B[:, 4:6] is (4,2), C[:, 4:6] is (4,2)
    # For each i in 0..3, j in 0..3, k in 0..1:
    #   C[i, 4+k] += A[i, j] * B[j, 4+k]
    
    col_offset = rank_strassen
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i * 4 + j, col_offset] = 1
                V[j * 6 + (4 + k), col_offset] = 1  # B index: row j, col 4+k
                W[(4 + k) * 4 + i, col_offset] = 1  # C index: row 4+k, col i... wait
                col_offset += 1
    
    # Wait, the output convention is W[k*n+i, r] where k is the column index of C, i is the row index.
    # C has shape (n, p) = (4, 6). So C[i, k] where i in 0..3, k in 0..5.
    # W index = k * n + i = k * 4 + i.
    
    # For C[i, 4+k] where k in 0..1: W index = (4+k) * 4 + i.
    # That's what I have.
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Snap to half-integers
    def snap_to_half(x):
        return round(x * 2) / 2
    
    U_list = [[snap_to_half(v) for v in row] for row in U_list]
    V_list = [[snap_to_half(v) for v in row] for row in V_list]
    W_list = [[snap_to_half(v) for v in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
