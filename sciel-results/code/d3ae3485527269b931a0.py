"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: 
    1. Split B (4x5) into B1 (4x4) and B2 (4x1).
    2. Use Recursive Strassen (Kronecker product of Strassen 2x2x2 with itself) for the 4x4x4 part.
       Rank = 7 * 7 = 49.
    3. Use direct multiplication for the 4x4x1 part.
       Rank = 4 * 4 * 1 = 16.
    4. Total Rank = 49 + 16 = 65.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    rank_total = 65
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    state = [0]
    
    def add_recursive_strassen_4x4():
        """Add 49 terms for 4x4 * 4x4 using recursive Strassen."""
        # Strassen formulas for 4x4 matrices split into 2x2 blocks.
        # Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]].
        # The 7 Strassen products are:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B11)
        # M4 = A22(B12 - B21)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to map each 2x2 block multiplication to a rank-7 Strassen decomposition.
        # Since the blocks are 2x2 matrices, we can't just "multiply" them in the tensor sense directly 
        # without expanding. However, the Kronecker product structure means we can compose the indices.
        
        # Let's define the 2x2 Strassen basis vectors for the "inner" 2x2 multiplication.
        # For a 2x2 multiplication A_block * B_block = C_block, the rank-7 decomposition is defined by:
        # M1: (a+d)(e+h)
        # M2: (c+d)e
        # M3: a(f-h)
        # M4: d(g-e)
        # M5: (a+b)h
        # M6: (c-a)(e+f)
        # M7: (b-d)(g+h)
        
        # Let's define the U, V, W patterns for a single 2x2 Strassen term.
        # We will define these as lists of (index, value) tuples for U, V, W.
        # Indices are local to the 2x2 block.
        
        # Helper to add a 2x2 Strassen term given global offsets for A, B, C blocks.
        # A block: rows [r0, r0+1], cols [c0, c0+1]
        # B block: rows [r0, r0+1], cols [c0, c0+1]
        # C block: rows [r0, r0+1], cols [c0, c0+1]
        
        def add_strassen_2x2(ra, ca, rb, cb, rc, cc, sign_a, sign_b, sign_c):
            """
            Add one Strassen term for 2x2 * 2x2.
            ra, ca: row, col offset of A block (0 or 1)
            rb, cb: row, col offset of B block (0 or 1)
            rc, cc: row, col offset of C block (0 or 1)
            sign_a, sign_b, sign_c: multipliers for the whole term (usually 1, but could be -1 if we factor out signs)
            Actually, it's easier to just encode the specific Strassen formula directly.
            """
            pass # We won't use this generic helper, we'll just hardcode the 49 terms.
            
        # Let's generate the 49 terms systematically.
        # The 4x4 Strassen algorithm computes 7 "block" multiplications.
        # Each block multiplication is a 2x2 * 2x2 multiplication.
        # Each 2x2 * 2x2 multiplication is decomposed into 7 scalar multiplications (Strassen).
        # Total 7 * 7 = 49 scalar multiplications.
        
        # Let's define the 7 Strassen formulas for 2x2 matrices as functions that return
        # the coefficients for the input elements (A and B) and output elements (C).
        
        # Strassen 2x2:
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
        # M3 = A00(B10 - B00) -> C01 += M3, C11 += M3
        # M4 = A11(B01 - B10) -> C00 += M4, C10 += M4
        # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
        # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
        # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
        
        # We need to map the "block" indices to the "element" indices.
        # For the 4x4 matrices:
        # A11 is rows 0,1 cols 0,1
        # A12 is rows 0,1 cols 2,3
        # A21 is rows 2,3 cols 0,1
        # A22 is rows 2,3 cols 2,3
        
        # Let's define a function to add a 2x2 Strassen decomposition for a specific block multiplication.
        # Inputs:
        #   a_ra, a_ca: top-left corner of A block (row, col) in 4x4 matrix
        #   b_ra, b_ca: top-left corner of B block (row, col) in 4x4 matrix
        #   c_ra, c_ca: top-left corner of C block (row, col) in 4x4 matrix
        #   scale_a, scale_b, scale_c: multipliers for the block (from the 4x4 Strassen formula)
        
        def add_block_strassen(a_ra, a_ca, b_ra, b_ca, c_ra, c_ca, scale_a, scale_b, scale_c):
            """
            Add 7 Strassen terms for a 2x2 * 2x2 block multiplication.
            The blocks are located at (a_ra, a_ca) in A, (b_ra, b_ca) in B, (c_ra, c_ca) in C.
            scale_a, scale_b, scale_c are multipliers applied to the A, B, C parts of the term.
            """
            # M1: (A00 + A11)(B00 + B11) -> C00 += 1, C11 += 1
            col = state[0]
            # A: (0,0) and (1,1) relative to block
            U[(a_ra+0)*m + (a_ca+0), col] = scale_a
            U[(a_ra+1)*m + (a_ca+1), col] = scale_a
            # B: (0,0) and (1,1) relative to block
            V[(b_ra+0)*p + (b_ca+0), col] = scale_b
            V[(b_ra+1)*p + (b_ca+1), col] = scale_b
            # C: (0,0) and (1,1) relative to block
            W[(c_ca+0)*n + (c_ra+0), col] = scale_c
            W[(c_ca+1)*n + (c_ra+1), col] = scale_c
            state[0] += 1
            
            # M2: (A10 + A11)B00 -> C10 += 1, C11 -= 1
            col = state[0]
            U[(a_ra+1)*m + (a_ca+0), col] = scale_a
            U[(a_ra+1)*m + (a_ca+1), col] = scale_a
            V[(b_ra+0)*p + (b_ca+0), col] = scale_b
            W[(c_ca+0)*n + (c_ra+1), col] = scale_c
            W[(c_ca+1)*n + (c_ra+1), col] = -scale_c
            state[0] += 1
            
            # M3: A00(B10 - B00) -> C01 += 1, C11 += 1
            col = state[0]
            U[(a_ra+0)*m + (a_ca+0), col] = scale_a
            V[(b_ra+1)*p + (b_ca+0), col] = scale_b
            V[(b_ra+0)*p + (b_ca+0), col] = -scale_b
            W[(c_ca+1)*n + (c_ra+0), col] = scale_c
            W[(c_ca+1)*n + (c_ra+1), col] = scale_c
            state[0] += 1
            
            # M4: A11(B01 - B10) -> C00 += 1, C10 += 1
            col = state[0]
            U[(a_ra+1)*m + (a_ca+1), col] = scale_a
            V[(b_ra+0)*p + (b_ca+1), col] = scale_b
            V[(b_ra+1)*p + (b_ca+0), col] = -scale_b
            W[(c_ca+0)*n + (c_ra+0), col] = scale_c
            W[(c_ca+0)*n + (c_ra+1), col] = scale_c
            state[0] += 1
            
            # M5: (A00 + A01)B11 -> C00 -= 1, C01 += 1
            col = state[0]
            U[(a_ra+0)*m + (a_ca+0), col] = scale_a
            U[(a_ra+0)*m + (a_ca+1), col] = scale_a
            V[(b_ra+1)*p + (b_ca+1), col] = scale_b
            W[(c_ca+0)*n + (c_ra+0), col] = -scale_c
            W[(c_ca+1)*n + (c_ra+0), col] = scale_c
            state[0] += 1
            
            # M6: (A10 - A00)(B00 + B01) -> C11 += 1
            col = state[0]
            U[(a_ra+1)*m + (a_ca+0), col] = scale_a
            U[(a_ra+0)*m + (a_ca+0), col] = -scale_a
            V[(b_ra+0)*p + (b_ca+0), col] = scale_b
            V[(b_ra+0)*p + (b_ca+1), col] = scale_b
            W[(c_ca+1)*n + (c_ra+1), col] = scale_c
            state[0] += 1
            
            # M7: (A01 - A11)(B10 + B11) -> C00 += 1
            col = state[0]
            U[(a_ra+0)*m + (a_ca+1), col] = scale_a
            U[(a_ra+1)*m + (a_ca+1), col] = -scale_a
            V[(b_ra+1)*p + (b_ca+0), col] = scale_b
            V[(b_ra+1)*p + (b_ca+1), col] = scale_b
            W[(c_ca+0)*n + (c_ra+0), col] = scale_c
            state[0] += 1
        
        # Now, apply the 4x4 Strassen formulas using add_block_strassen.
        # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
        # This is a sum of 4 block multiplications:
        # A11*B11, A11*B22, A22*B11, A22*B22
        # All contribute to C11 and C22.
        
        # Let's break down M1:
        # (A11 + A22)(B11 + B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # Each term is a 2x2 * 2x2 multiplication.
        # We add them all with scale_a=1, scale_b=1, scale_c=1 for C11 and C22.
        
        # Term 1.1: A11 * B11
        # A11 is at (0,0), B11 is at (0,0), C11 is at (0,0)
        add_block_strassen(0, 0, 0, 0, 0, 0, 1, 1, 1)
        # A11 * B11 also contributes to C22? No, the Strassen formula says M1 contributes to C11 and C22.
        # So we need to add the same 7 terms again, but with C block at (0,2) and (2,0)?
        # Wait, C11 is top-left 2x2, C22 is bottom-right 2x2.
        # So for C11, c_ra=0, c_ca=0. For C22, c_ra=2, c_ca=2.
        # We must add the 7 terms for C11, and then the 7 terms for C22.
        
        # Let's refactor: add_block_strassen adds 7 terms for ONE C block location.
        # So for M1, we need to add 4 * 2 * 7 = 56 terms? No, that's not right.
        # The Strassen formula computes M1 = (A11+A22)(B11+B22).
        # This is ONE block multiplication result.
        # But (A11+A22) is a 2x2 matrix, (B11+B22) is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # We compute this product using Strassen (7 scalar mults).
        # So we call add_block_strassen ONCE for the "virtual" A block (A11+A22) and "virtual" B block (B11+B22).
        # But our add_block_strassen assumes the A and B blocks are at fixed locations in the original matrices.
        # We need to modify it to accept linear combinations?
        
        # Alternative approach:
        # Instead of modifying add_block_strassen, we can just expand the 4x4 Strassen formulas
        # into 49 scalar multiplications directly.
        
        # Let's list the 49 scalar multiplications.
        # M1 = (A11 + A22)(B11 + B22)
        # Let X = A11 + A22, Y = B11 + B22.
        # X is a 2x2 matrix: [[A00+A20, A01+A21], [A10+A30, A11+A31]] (using 0-indexed)
        # Y is a 2x2 matrix: [[B00+B20, B01+B21], [B10+B30, B11+B31]]
        # We compute X*Y using Strassen.
        # This gives 7 terms.
        # Each term is a sum of products of elements of X and Y.
        # Each element of X is a sum of 2 elements of A.
        # Each element of Y is a sum of 2 elements of B.
        # So each "scalar" multiplication in the 2x2 Strassen of X*Y is actually a product of sums.
        # (A_i + A_j)(B_k + B_l) = A_i B_k + A_i B_l + A_j B_k + A_j B_l.
        # This expands to 4 scalar multiplications.
        # So each of the 7 Strassen terms expands to 4 scalar multiplications.
        # Total 7 * 4 = 28 scalar multiplications for M1.
        # Similarly for M2...M7.
        # Total 7 * 28 = 196 scalar multiplications?
        # No, that's not right. The rank is 49.
        
        # Let's re-evaluate.
        # The Kronecker product of the Strassen algorithm (rank 7) with itself (rank 7) gives rank 49.
        # This means we can compute the 4x4 product with 49 scalar multiplications.
        # How?
        # The Strassen algorithm for 2x2 matrices uses 7 multiplications.
        # For 4x4 matrices, we view them as 2x2 matrices of 2x2 blocks.
        # We apply the 2x2 Strassen algorithm to the blocks.
        # This gives 7 "block multiplications".
        # Each block multiplication is a 2x2 * 2x2 multiplication.
        # We compute each block multiplication using the 2x2 Strassen algorithm.
        # This gives 7 * 7 = 49 scalar multiplications.
        
        # So the structure is:
        # For each of the 7 Strassen formulas (M1..M7) for the 4x4 blocks:
        #   For each of the 7 Strassen formulas (m1..m7) for the 2x2 blocks:
        #     Add 1 scalar multiplication term.
        
        # Let's implement this by defining the 7 4x4 Strassen formulas and the 7 2x2 Strassen formulas.
        
        # 4x4 Strassen formulas (block level):
        # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
        # M2 = (A21 + A22)B11 -> C21 += M2, C22 -= M2
        # M3 = A11(B21 - B11) -> C12 += M3, C22 += M3
        # M4 = A22(B12 - B21) -> C11 += M4, C21 += M4
        # M5 = (A11 + A12)B22 -> C11 -= M5, C12 += M5
        # M6 = (A21 - A11)(B11 + B12) -> C22 += M6
        # M7 = (A12 - A22)(B21 + B22) -> C11 += M7
        
        # 2x2 Strassen formulas (element level):
        # m1 = (a00 + a11)(b00 + b11) -> c00 += m1, c11 += m1
        # m2 = (a10 + a11)b00 -> c10 += m2, c11 -= m2
        # m3 = a00(b10 - b00) -> c01 += m3, c11 += m3
        # m4 = a11(b01 - b10) -> c00 += m4, c10 += m4
        # m5 = (a00 + a01)b11 -> c00 -= m5, c01 += m5
        # m6 = (a10 - a00)(b00 + b01) -> c11 += m6
        # m7 = (a01 - a11)(b10 + b11) -> c00 += m7
        
        # We need to combine these.
        # For each 4x4 term (e.g., M1), we have a "virtual" A block and "virtual" B block.
        # The "virtual" A block is a linear combination of 2x2 blocks of A.
        # The "virtual" B block is a linear combination of 2x2 blocks of B.
        # We apply the 2x2 Strassen algorithm to these virtual blocks.
        # Each "scalar" multiplication in the 2x2 Strassen algorithm is a product of two "virtual" elements.
        # Each "virtual" element is a linear combination of 2x2 elements (i.e., 4 scalars).
        # So each "scalar" multiplication is a product of two linear combinations, which expands to 4 scalar multiplications.
        # Wait, this would give 7 * 7 * 4 = 196 scalar multiplications.
        # But the rank is 49.
        
        # I think I am confusing the "rank" of the block algorithm with the "rank" of the scalar algorithm.
        # The rank of the 4x4x4 tensor is 49.
        # This means there exists a decomposition with 49 terms.
        # The recursive Strassen algorithm provides such a decomposition.
        
        # Let's look at the indices.
        # The 4x4 Strassen algorithm has 7 "intermediate" products.
        # Each intermediate product is a 2x2 matrix.
        # We compute each intermediate product using the 2x2 Strassen algorithm, which has 7 terms.
        # So total 7 * 7 = 49 terms.
        
        # Each term is a product of:
        #   - A linear combination of 2x2 elements of A (from the 4x4 Strassen formula)
        #   - A linear combination of 2x2 elements of B (from the 4x4 Strassen formula)
        #   - And then the 2x2 Strassen formula combines these "virtual" elements.
        
        # Let's define the 49 terms explicitly.
        # We will use a nested loop over the 7 4x4 Strassen terms and the 7 2x2 Strassen terms.
        
        # Define the 4x4 Strassen terms as (a_blocks, b_blocks, c_blocks, signs)
        # a_blocks: list of (block_row, block_col, sign)
        # b_blocks: list of (block_row, block_col, sign)
        # c_blocks: list of (block_row, block_col, sign)
        
        # M1: (A11 + A22)(B11 + B22) -> C11 += 1, C22 += 1
        # a_blocks: [(0,0,1), (1,1,1)]
        # b_blocks: [(0,0,1), (1,1,1)]
        # c_blocks: [(0,0,1), (1,1,1)]
        
        # M2: (A21 + A22)B11 -> C21 += 1, C22 -= 1
        # a_blocks: [(1,0,1), (1,1,1)]
        # b_blocks: [(0,0,1)]
        # c_blocks: [(1,0,1), (1,1,-1)]
        
        # M3: A11(B21 - B11) -> C12 += 1, C22 += 1
        # a_blocks: [(0,0,1)]
        # b_blocks: [(1,0,1), (0,0,-1)]
        # c_blocks: [(0,1,1), (1,1,1)]
        
        # M4: A22(B12 - B21) -> C11 += 1, C21 += 1
        # a_blocks: [(1,1,1)]
        # b_blocks: [(0,1,1), (1,0,-1)]
        # c_blocks: [(0,0,1), (1,0,1)]
        
        # M5: (A11 + A12)B22 -> C11 -= 1, C12 += 1
        # a_blocks: [(0,0,1), (0,1,1)]
        # b_blocks: [(1,1,1)]
        # c_blocks: [(0,0,-1), (0,1,1)]
        
        # M6: (A21 - A11)(B11 + B12) -> C22 += 1
        # a_blocks: [(1,0,1), (0,0,-1)]
        # b_blocks: [(0,0,1), (0,1,1)]
        # c_blocks: [(1,1,1)]
        
        # M7: (A12 - A22)(B21 + B22) -> C11 += 1
        # a_blocks: [(0,1,1), (1,1,-1)]
        # b_blocks: [(1,0,1), (1,1,1)]
        # c_blocks: [(0,0,1)]
        
        # 2x2 Strassen terms (m1..m7) as (a_elems, b_elems, c_elems, signs)
        # a_elems: list of (row, col, sign) relative to the 2x2 block
        # b_elems: list of (row, col, sign) relative to the 2x2 block
        # c_elems: list of (row, col, sign) relative to the 2x2 block
        
        # m1: (a00 + a11)(b00 + b11) -> c00 += 1, c11 += 1
        # a_elems: [(0,0,1), (1,1,1)]
        # b_elems: [(0,0,1), (1,1,1)]
        # c_elems: [(0,0,1), (1,1,1)]
        
        # m2: (a10 + a11)b00 -> c10 += 1, c11 -= 1
        # a_elems: [(1,0,1), (1,1,1)]
        # b_elems: [(0,0,1)]
        # c_elems: [(1,0,1), (1,1,-1)]
        
        # m3: a00(b10 - b00) -> c01 += 1, c11 += 1
        # a_elems: [(0,0,1)]
        # b_elems: [(1,0,1), (0,0,-1)]
        # c_elems: [(0,1,1), (1,1,1)]
        
        # m4: a11(b01 - b10) -> c00 += 1, c10 += 1
        # a_elems: [(1,1,1)]
        # b_elems: [(0,1,1), (1,0,-1)]
        # c_elems: [(0,0,1), (1,0,1)]
        
        # m5: (a00 + a01)b11 -> c00 -= 1, c01 += 1
        # a_elems: [(0,0,1), (0,1,1)]
        # b_elems: [(1,1,1)]
        # c_elems: [(0,0,-1), (0,1,1)]
        
        # m6: (a10 - a00)(b00 + b01) -> c11 += 1
        # a_elems: [(1,0,1), (0,0,-1)]
        # b_elems: [(0,0,1), (0,1,1)]
        # c_elems: [(1,1,1)]
        
        # m7: (a01 - a11)(b10 + b11) -> c00 += 1
        # a_elems: [(0,1,1), (1,1,-1)]
        # b_elems: [(1,0,1), (1,1,1)]
        # c_elems: [(0,0,1)]
        
        # Now, for each 4x4 term (M1..M7) and each 2x2 term (m1..m7):
        # We add a scalar multiplication term.
        # The U vector is the product of the "virtual" A element and the "virtual" B element.
        # Wait, no. The U, V, W vectors are for the scalar multiplication.
        # U is the coefficient of the A elements.
        # V is the coefficient of the B elements.
        # W is the coefficient of the C elements.
        
        # For a given 4x4 term M and 2x2 term m:
        # The "virtual" A element is a linear combination of A block elements.
        # The "virtual" B element is a linear combination of B block elements.
        # The product is (sum_a A_a) * (sum_b B_b) = sum_a sum_b A_a B_b.
        # This is not a single scalar multiplication, it's a sum of scalar multiplications.
        
        # I think the recursive Strassen algorithm works differently.
        # It doesn't expand the linear combinations.
        # It uses the fact that the tensor for 4x4x4 is the Kronecker product of the tensor for 2x2x2 with itself.
        # So the decomposition is the Kronecker product of the 2x2x2 decomposition with itself.
        
        # Let's implement the Kronecker product of the 2x2x2 Strassen decomposition.
        # The 2x2x2 Strassen decomposition has 7 terms.
        # Each term has U, V, W vectors of length 4.
        # The 4x4x4 decomposition will have 7*7 = 49 terms.
        # Each term will have U, V, W vectors of length 16.
        # The U vector for term (i,j) is the Kronecker product of U_i and U_j.
        # Similarly for V and W.
        
        # Let's define the 2x2x2 Strassen decomposition.
        # U, V, W are 4x7 matrices.
        
        U_2x2 = np.zeros((4, 7))
        V_2x2 = np.zeros((4, 7))
        W_2x2 = np.zeros((4, 7))
        
        # M1: (a00 + a11)(b00 + b11) -> c00 += 1, c11 += 1
        U_2x2[0, 0] = 1; U_2x2[3, 0] = 1
        V_2x2[0, 0] = 1; V_2x2[3, 0] = 1
        W_2x2[0, 0] = 1; W_2x2[3, 0] = 1
        
        # M2: (a10 + a11)b00 -> c10 += 1, c11 -= 1
        U_2x2[2, 1] = 1; U_2x2[3, 1] = 1
        V_2x2[0, 1] = 1
        W_2x2[2, 1] = 1; W_2x2[3, 1] = -1
        
        # M3: a00(b10 - b00) -> c01 += 1, c11 += 1
        U_2x2[0, 2] = 1
        V_2x2[2, 2] = 1; V_2x2[0, 2] = -1
        W_2x2[1, 2] = 1; W_2x2[3, 2] = 1
        
        # M4: a11(b01 - b10) -> c00 += 1, c10 += 1
        U_2x2[3, 3] = 1
        V_2x2[1, 3] = 1; V_2x2[2, 3] = -1
        W_2x2[0, 3] = 1; W_2x2[2, 3] = 1
        
        # M5: (a00 + a01)b11 -> c00 -= 1, c01 += 1
        U_2x2[0, 4] = 1; U_2x2[1, 4] = 1
        V_2x2[3, 4] = 1
        W_2x2[0, 4] = -1; W_2x2[1, 4] = 1
        
        # M6: (a10 - a00)(b00 + b01) -> c11 += 1
        U_2x2[2, 5] = 1; U_2x2[0, 5] = -1
        V_2x2[0, 5] = 1; V_2x2[1, 5] = 1
        W_2x2[3, 5] = 1
        
        # M7: (a01 - a11)(b10 + b11) -> c00 += 1
        U_2x2[1, 6] = 1; U_2x2[3, 6] = -1
        V_2x2[2, 6] = 1; V_2x2[3, 6] = 1
        W_2x2[0, 6] = 1
        
        # Now, compute the Kronecker product.
        # U_4x4 = np.kron(U_2x2, U_2x2)
        # V_4x4 = np.kron(V_2x2, V_2x2)
        # W_4x4 = np.kron(W_2x2, W_2x2)
        
        # But we need to map the indices correctly.
        # The 4x4 matrix is stored in row-major order.
        # The 2x2 blocks are:
        # A11: rows 0,1 cols 0,1 -> indices 0,1,4,5
        # A12: rows 0,1 cols 2,3 -> indices 2,3,6,7
        # A21: rows 2,3 cols 0,1 -> indices 8,9,12,13
        # A22: rows 2,3 cols 2,3 -> indices 10,11,14,15
        
        # The Kronecker product np.kron(U_2x2, U_2x2) will give a 16x49 matrix.
        # The row indices will be 0..15.
        # We need to permute the rows to match the 4x4 row-major order.
        
        # Let's check the row indices of np.kron.
        # np.kron(A, B) where A is 4x7 and B is 4x7.
        # The row index of the result is i*4 + j, where i is the row index of A and j is the row index of B.
        # A's row indices correspond to the 2x2 block indices.
        # A's row 0 -> A11, row 1 -> A12, row 2 -> A21, row 3 -> A22.
        # B's row indices correspond to the 2x2 element indices within the block.
        # B's row 0 -> (0,0), row 1 -> (0,1), row 2 -> (1,0), row 3 -> (1,1).
        
        # So the row index i*4 + j corresponds to:
        # Block i, element j.
        # i=0 (A11), j=0 ((0,0)) -> global index 0*4+0 = 0. Correct.
        # i=0 (A11), j=1 ((0,1)) -> global index 0*4+1 = 1. But A11's (0,1) is global index 1. Correct.
        # i=0 (A11), j=2 ((1,0)) -> global index 0*4+2 = 2. But A11's (1,0) is global index 4. Incorrect.
        
        # So we need to permute the rows.
        # The mapping from (block, element) to global index is:
        # block i, element j -> global index = (i//2)*4 + (j//2) + 2*(i%2) + 2*(j%2) ?
        # Let's define the mapping explicitly.
        
        # Block 0 (A11): elements 0,1,4,5
        # Block 1 (A12): elements 2,3,6,7
        # Block 2 (A21): elements 8,9,12,13
        # Block 3 (A22): elements 10,11,14,15
        
        # Element indices within a block:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # So for block i and element j:
        # row = i // 2, col = i % 2
        # erow = j // 2, ecol = j % 2
        # global_row = row * 2 + erow
        # global_col = col * 2 + ecol
        # global_index = global_row * 4 + global_col
        
        # Let's create the permutation.
        perm = np.zeros(16, dtype=int)
        for i in range(4):
            for j in range(4):
                kron_idx = i * 4 + j
                row = i // 2
                col = i % 2
                erow = j // 2
                ecol = j % 2
                global_row = row * 2 + erow
                global_col = col * 2 + ecol
                global_idx = global_row * 4 + global_col
                perm[kron_idx] = global_idx
        
        # Now, U_4x4 = np.kron(U_2x2, U_2x2)
        # U_4x4_permuted = U_4x4[perm, :]
        
        # We will fill the U, V, W matrices for the 4x4x4 part.
        # The 4x4x4 part uses the first 49 columns of U, V, W.
        
        U_kron = np.kron(U_2x2, U_2x2)
        V_kron = np.kron(V_2x2, V_2x2)
        W_kron = np.kron(W_2x2, W_2x2)
        
        U[:, :49] = U_kron[perm, :]
        V[:, :49] = V_kron[perm, :]
        W[:, :49] = W_kron[perm, :]
        
        state[0] += 49
    
    def add_direct_4x4_4x1():
        """Add 16 terms for 4x4 * 4x1 direct multiplication."""
        # A (4x4) * B (4x1) = C (4x1)
        # C[i] = sum_j A[i,j] B[j]
        # For each i, j:
        #   U[i*4+j, col] = 1
        #   V[j*5+4, col] = 1  # B is 4x5, col 4 is the 5th column
        #   W[4*4+i, col] = 1  # C is 4x4, col 4 is the 5th column, row i
        
        for i in range(4):
            for j in range(4):
                col = state[0]
                U[i * m + j, col] = 1.0
                V[j * p + 4, col] = 1.0
                W[4 * n + i, col] = 1.0
                state[0] += 1
    
    # Add the 4x4x4 part
    add_recursive_strassen_4x4()
    
    # Add the 4x4x1 part
    add_direct_4x4_4x1()
    
    used_rank = state[0]
    
    U = U[:, :used_rank]
    V = V[:, :used_rank]
    W = W[:, :used_rank]
    
    # Verify the decomposition
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition (rank n*m*p = 80)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
