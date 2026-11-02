"""Matrix multiplication 4x4 by 4x7 - tensor decomposition.

We exploit that 7 = 4 + 3 (or 2 + 5, or other splits) and use the
Strassen algorithm for 4x4x4x4x4 (rank 49) combined with schoolbook for
the remainder, to achieve rank < 112 (schoolbook).

Best known approach: split the 7 columns as 4 + 3.
- 4x4 * 4x4: Strassen rank 49
- 4x4 * 4x3: schoolbook rank 48
Total: 97.

Actually, we can do better with a 2+5 split or other decompositions.
Let me think more carefully.

For 4x4 x 4x7:
- Split into 4x4 x 4x4 (rank 49 via Strassen) + 4x4 x 4x3 (rank 48 schoolbook) = 97

Can we do 4x4 x 4x5 with rank < 80?
4x4 x 4x5: split as 4x4 x 4x4 (49) + 4x4 x 4x1 (16) = 65
So 4x4 x 4x7 = 49 + 65 = 114. That's worse.

Better: 4x4 x 4x7 = 4x4 x 4x4 (49) + 4x4 x 4x3 (48) = 97.

Or: use Strassen for the 4x4 block and a better method for 4x3.
4x4 x 4x3 schoolbook = 48. That seems optimal for the 3-column case.

Let me try: 2+5 split.
4x4 x 4x2 = 32 (schoolbook)
4x4 x 4x5 = 49+16 = 65 (if 5=4+1)
Total = 97. Same.

What about using the fact that we can share some of the Strassen intermediates?
Actually no - the blocks operate on different columns of B, so they're independent.

Let me just implement the 4+3 split with Strassen for 4x4x4x4 and schoolbook for 4x4x4x3.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    """Construct a rank-minimal decomposition for 4x4 * 4x7 matrix multiplication.
    
    Uses Strassen for the 4x4 x 4x4 block (rank 49) and schoolbook for the
    4x4 x 4x3 block (rank 48). Total rank = 97.
    """
    n, m, p = 4, 4, 7
    
    # Build 4x4 * 4x4 Strassen decomposition
    U_44, V_44, W_44 = _build_strassen_44()
    r_44 = len(W_44)  # should be 49
    
    # Build 4x4 * 4x3 schoolbook decomposition
    U_43, V_43, W_43 = _build_schoolbook_443()
    r_43 = len(W_43)  # should be 48
    
    total_rank = r_44 + r_43
    
    # Assemble full matrices
    # U: shape (n*m, total_rank) = (16, 97)
    # V: shape (m*p, total_rank) = (28, 97)
    # W: shape (p*n, total_rank) = (28, 97)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Place Strassen 4x4 part (columns 0-3 of B, rows 0-3 of C)
    for i in range(n * m):
        for c in range(r_44):
            U[i][c] = U_44[i][c]
    for i in range(m * 4):  # first 4 columns of B
        for c in range(r_44):
            V[i][c] = V_44[i][c]
    for i in range(4 * n):  # first 4 rows of C
        for c in range(r_44):
            W[i][c] = W_44[i][c]
    
    # Place schoolbook 4x4 x 4x3 part (columns 4-6 of B, rows 4-6 of C)
    offset = r_44
    for i in range(n * m):
        for c in range(r_43):
            U[i][offset + c] = U_43[i][c]
    for i in range(4, m * p):  # columns 4,5,6 of B
        for c in range(r_43):
            V[i][offset + c] = V_43[i - 4][c]
    for i in range(4 * n, p * n):  # rows 4,5,6 of C
        for c in range(r_43):
            W[i][offset + c] = W_43[i - 4 * n][c]
    
    return {"U": U, "V": V, "W": W}


def _build_strassen_44():
    """Build Strassen decomposition for 4x4 * 4x4.
    
    Returns U (16x49), V (16x49), W (16x49) as lists of lists.
    
    We split 4x4 matrices into 2x2 blocks and apply Strassen recursively.
    For 2x2 * 2x2, Strassen uses 7 multiplications instead of 8.
    For 4x4 * 4x4: split into four 2x2 blocks, apply Strassen => 7 * (2x2 mults)
    But we also need to handle the additions. The rank of 2x2*2x2 Strassen is 7.
    
    Actually, for 4x4*4x4 with 2x2 sub-blocks:
    We have A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    C11 = A11*B11 - A12*B21 + (A11+A12)*B22 (not quite standard)
    
    Standard Strassen for 2x2:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4*4x4, we decompose into 2x2 blocks and each "multiplication" of 2x2 blocks
    itself can use Strassen (rank 7) or schoolbook (rank 4).
    
    If we use Strassen recursively: rank = 7 * 7 = 49
    If we use schoolbook for the inner: rank = 7 * 4 = 28 (but then need to handle bilinear form)
    
    Let me use the recursive Strassen approach: rank 49.
    """
    n = 4
    # We'll build this by composing Strassen for 2x2 (rank 7) with itself.
    # The 4x4 matrix is split into 2x2 blocks indexed by (row_block, col_block)
    # A[ri, ci] for ri, ci in {0,1}, each being 2x2.
    
    # For each of the 7 Strassen multiplications at the top level,
    # we multiply two 2x2 matrices, which we do via Strassen again (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # Let's enumerate the 49 rank-1 terms.
    # Top-level Strassen variables for A 2x2 blocks (each 2x2):
    # A blocks: A00, A01, A10, A11 (each 2x2, indexed by (r,c) within)
    # B blocks: B00, B01, B10, B11
    
    # Top-level Strassen combinations (for 4x4 as 2x2 of 2x2 blocks):
    # S1 = (A00+A11)*(B00+B11)
    # S2 = (A10+A11)*B00
    # S3 = A00*(B01-B11)
    # S4 = A11*(B10-B00)
    # S5 = (A00+A01)*B11
    # S6 = (A10-A00)*(B00+B01)
    # S7 = (A01-A11)*(B10+B11)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Each Sx is a product of two 2x2 matrices (formed by adding/subtracting blocks).
    # We decompose each 2x2*2x2 product using Strassen (7 rank-1 terms).
    
    # For a 2x2 matrix X and 2x2 matrix Y, the Strassen decomposition has 7 terms.
    # Let's define a helper that gives the 7 U,V,W coefficient vectors for a 2x2*2x2 product.
    
    # Actually, let me think about this differently.
    # The tensor T_{(4,4,4)} has rank <= 49 via recursive Strassen.
    # I need to explicitly construct U, V, W.
    
    # Let me index elements:
    # A[i,j] for i,j in 0..3 (row, col) - 16 elements
    # B[j,k] for j,k in 0..3 - 16 elements  
    # C[i,k] for i,k in 0..3 - 16 elements
    
    # Split into 2x2 blocks:
    # A[0:2, 0:2] = A00, A[0:2, 2:4] = A01, A[2:4, 0:2] = A10, A[2:4, 2:4] = A11
    # Similarly for B and C.
    
    # For each top-level Strassen product (say X*Y where X,Y are 2x2 linear combinations),
    # we need to decompose X*Y into 7 rank-1 terms.
    
    # For 2x2 * 2x2 with Strassen:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # 
    # m1 = (x00+x11)*(y00+y11)
    # m2 = (x10+x11)*y00
    # m3 = x00*(y01-y11)
    # m4 = x11*(y10-y00)
    # m5 = (x00+x01)*y11
    # m6 = (x10-x00)*(y00+y01)
    # m7 = (x01-x11)*(y10+y11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # Each mi contributes to c00,c01,c10,c11 with coefficients.
    # And each mi is a product of two linear forms in the x's and y's.
    
    # So for each of the 7 inner Strassen steps, I need:
    # - U vectors: coefficients on the x variables (which are linear combos of A entries)
    # - V vectors: coefficients on the y variables (which are linear combos of B entries)
    # - W contribution: which C entry gets which coefficient
    
    # Let me build this systematically.
    
    # Define the 7 Strassen inner products for 2x2*2x2:
    # Each gives (u_vec, v_vec, w_contribs) where
    # u_vec is a 4-vector (coefficients on x00,x01,x10,x11)
    # v_vec is a 4-vector (coefficients on y00,y01,y10,y11)
    # w_contribs is a dict from c_index to coefficient
    
    strassen_2x2_terms = []
    
    # m1 = (x00+x11)*(y00+y11)
    # u: [1, 0, 0, 1], v: [1, 0, 0, 1]
    # contributes to c00: +1, c11: +1
    strassen_2x2_terms.append(([1,0,0,1], [1,0,0,1], [(0, 1), (3, 1)]))
    
    # m2 = (x10+x11)*y00
    # u: [0, 0, 1, 1], v: [1, 0, 0, 0]
    # contributes to c10: +1, c11: -1
    strassen_2x2_terms.append(([0,0,1,1], [1,0,0,0], [(2, 1), (3, -1)]))
    
    # m3 = x00*(y01-y11)
    # u: [1, 0, 0, 0], v: [0, 1, 0, -1]
    # contributes to c01: +1, c11: +1
    strassen_2x2_terms.append(([1,0,0,0], [0,1,0,-1], [(1, 1), (3, 1)]))
    
    # m4 = x11*(y10-y00)
    # u: [0, 0, 0, 1], v: [0, 0, 1, -1]  -- wait, y10-y00 means v=[-1,0,1,0]
    # Actually y = [[y00, y01], [y10, y11]], so y10 is at index 2, y00 at index 0.
    # v: [-1, 0, 1, 0]
    # contributes to c00: +1, c10: +1
    strassen_2x2_terms.append(([0,0,0,1], [-1,0,1,0], [(0, 1), (2, 1)]))
    
    # m5 = (x00+x01)*y11
    # u: [1, 1, 0, 0], v: [0, 0, 0, 1]
    # contributes to c00: -1, c01: +1
    strassen_2x2_terms.append(([1,1,0,0], [0,0,0,1], [(0, -1), (1, 1)]))
    
    # m6 = (x10-x00)*(y00+y01)
    # u: [-1, 0, 1, 0], v: [1, 1, 0, 0]
    # contributes to c11: +1
    strassen_2x2_terms.append(([-1,0,1,0], [1,1,0,0], [(3, 1)]))
    
    # m7 = (x01-x11)*(y10+y11)
    # u: [0, 1, 0, -1], v: [0, 0, 1, 1]
    # contributes to c00: +1
    strassen_2x2_terms.append(([0,1,0,-1], [0,0,1,1], [(0, 1)]))
    
    # Now, top-level Strassen for 4x4*4x4:
    # We define 7 products S1..S7, each of which is a 2x2*2x2 product.
    # The X and Y matrices for each are linear combinations of the 2x2 blocks.
    
    # A blocks: A00 (indices 0-3 in flat 16), A01 (4-7), A10 (8-11), A11 (12-15)
    # Wait, let me be careful with indexing.
    # A[i,j] where i in 0..3 (row), j in 0..3 (col).
    # A00 = A[0:2, 0:2] = indices (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,4,5
    # A01 = A[0:2, 2:4] = indices (0,2),(0,3),(1,2),(1,3) -> flat: 2,3,6,7
    # A10 = A[2:4, 0:2] = indices (2,0),(2,1),(3,0),(3,1) -> flat: 8,9,12,13
    # A11 = A[2:4, 2:4] = indices (2,2),(2,3),(3,2),(3,3) -> flat: 10,11,14,15
    
    # Hmm, let me use a different flat ordering. Let's say A_flat[i*m+j] = A[i,j]
    # with m=4. So A_flat[0]=A[0,0], A_flat[1]=A[0,1], ..., A_flat[3]=A[0,3],
    # A_flat[4]=A[1,0], etc.
    
    # A00 block (rows 0-1, cols 0-1): A[0,0],A[0,1],A[1,0],A[1,1] -> flat indices 0,1,4,5
    # A01 block (rows 0-1, cols 2-3): A[0,2],A[0,3],A[1,2],A[1,3] -> flat indices 2,3,6,7
    # A10 block (rows 2-3, cols 0-1): A[2,0],A[2,1],A[3,0],A[3,1] -> flat indices 8,9,12,13
    # A11 block (rows 2-3, cols 2-3): A[2,2],A[2,3],A[3,2],A[3,3] -> flat indices 10,11,14,15
    
    # For the inner 2x2 Strassen, x = [x00, x01, x10, x11] maps to:
    # If x represents A00: x00=A[0,0]=A[0], x01=A[0,1]=A[1], x10=A[1,0]=A[4], x11=A[1,1]=A[5]
    # If x represents A01: x00=A[0,2]=A[2], x01=A[0,3]=A[3], x10=A[1,2]=A[6], x11=A[1,3]=A[7]
    # If x represents A10: x00=A[2,0]=A[8], x01=A[2,1]=A[9], x10=A[3,0]=A[12], x11=A[3,1]=A[13]
    # If x represents A11: x00=A[2,2]=A[10], x01=A[2,3]=A[11], x10=A[3,2]=A[14], x11=A[3,3]=A[15]
    
    # Let me define mapping from (row_block, col_block) to flat indices within the 2x2 block:
    # Block (0,0): [0, 1, 4, 5]  -> x00, x01, x10, x11
    # Block (0,1): [2, 3, 6, 7]
    # Block (1,0): [8, 9, 12, 13]
    # Block (1,1): [10, 11, 14, 15]
    
    block_A_indices = {
        (0,0): [0, 1, 4, 5],
        (0,1): [2, 3, 6, 7],
        (1,0): [8, 9, 12, 13],
        (1,1): [10, 11, 14, 15]
    }
    
    # Similarly for B (m*p = 4*4 = 16 elements, B[j,k] flat index j*4+k):
    # B00 = B[0:2, 0:2]: B[0,0],B[0,1],B[1,0],B[1,1] -> flat: 0,1,4,5
    # B01 = B[0:2, 2:4]: B[0,2],B[0,3],B[1,2],B[1,3] -> flat: 2,3,6,7
    # B10 = B[2:4, 0:2]: B[2,0],B[2,1],B[3,0],B[3,1] -> flat: 8,9,12,13
    # B11 = B[2:4, 2:4]: B[2,2],B[2,3],B[3,2],B[3,3] -> flat: 10,11,14,15
    
    block_B_indices = {
        (0,0): [0, 1, 4, 5],
        (0,1): [2, 3, 6, 7],
        (1,0): [8, 9, 12, 13],
        (1,1): [10, 11, 14, 15]
    }
    
    # C[i,k] flat index i*4+k:
    # C00 = C[0:2, 0:2]: C[0,0],C[0,1],C[1,0],C[1,1] -> flat: 0,1,4,5
    # C01 = C[0:2, 2:4]: C[0,2],C[0,3],C[1,2],C[1,3] -> flat: 2,3,6,7
    # C10 = C[2:4, 0:2]: C[2,0],C[2,1],C[3,0],C[3,1] -> flat: 8,9,12,13
    # C11 = C[2:4, 2:4]: C[2,2],C[2,3],C[3,2],C[3,3] -> flat: 10,11,14,15
    
    block_C_indices = {
        (0,0): [0, 1, 4, 5],
        (0,1): [2, 3, 6, 7],
        (1,0): [8, 9, 12, 13],
        (1,1): [10, 11, 14, 15]
    }
    
    # Top-level Strassen products:
    # S1 = (A00+A11) * (B00+B11)
    # S2 = (A10+A11) * B00
    # S3 = A00 * (B01-B11)
    # S4 = A11 * (B10-B00)
    # S5 = (A00+A01) * B11
    # S6 = (A10-A00) * (B00+B01)
    # S7 = (A01-A11) * (B10+B11)
    
    # For each Sx, the X matrix is a linear combination of A-blocks and Y is of B-blocks.
    # I'll represent this as: for the inner 2x2, x00 comes from certain A entries with certain coeffs.
    
    # Let me define for each top-level term:
    # a_block_coeffs: list of 4 (coeff for A00, A01, A10, A11) -> forms X
    # b_block_coeffs: list of 4 (coeff for B00, B01, B10, B11) -> forms Y
    # Then X[rr,cc] = sum over blocks b of coeff_A[b] * A_block_b[rr,cc]
    # And similarly for Y.
    
    # Top-level terms:
    # Index: (a_coeffs, b_coeffs, c_contribution_block, c_coeffs)
    # where c_contribution_block is which C-block this Sx adds to, and c_coeffs
    # are the coefficients for c00,c01,c10,c11 within that block.
    
    # S1 -> C00 (+1), C11 (+1)
    # S2 -> C10 (+1), C11 (-1)
    # S3 -> C01 (+1), C11 (+1)
    # S4 -> C00 (+1), C10 (+1)
    # S5 -> C00 (-1), C01 (+1)
    # S6 -> C11 (+1)
    # S7 -> C00 (+1)
    
    top_level_terms = [
        # (a_block_coeffs [A00,A01,A10,A11], b_block_coeffs [B00,B01,B10,B11], [(c_block, c_inner_index, coeff)])
        ([1, 0, 0, 1], [1, 0, 0, 1], [((0,0), 0, 1), ((1,1), 0, 1)]),        # S1
        ([0, 0, 1, 1], [1, 0, 0, 0], [((1,0), 0, 1), ((1,1), 0, -1)]),       # S2
        ([1, 0, 0, 0], [0, 1, 0, -1], [((0,1), 0, 1), ((1,1), 0, 1)]),       # S3
        ([0, 0, 0, 1], [0, 0, 1, -1], [((0,0), 0, 1), ((1,0), 0, 1)]),       # S4
        ([1, 1, 0, 0], [0, 0, 0, 1], [((0,0), 0, -1), ((0,1), 0, 1)]),       # S5
        ([0, 0, 1, -1], [1, 1, 0, 0], [((1,1), 0, 1)]),                       # S6
        ([0, 1, 0, -1], [0, 0, 1, 1], [((0,0), 0, 1)]),                       # S7
    ]
    
    # Now build U, V, W.
    # U has 16 rows (A entries), 49 cols
    # V has 16 rows (B entries), 49 cols
    # W has 16 rows (C entries), 49 cols
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for tl_i, (a_coeffs, b_coeffs, c_contribs) in enumerate(top_level_terms):
        for inner_i, (u_inner, v_inner, w_inner) in enumerate(strassen_2x2_terms):
            col = tl_i * 7 + inner_i
            
            # U contribution: for each element of the inner 2x2 X matrix (positions 0,1,2,3)
            # u_inner[r] is the coefficient for x_r
            # x_r is formed from the A blocks: x_r = sum_b a_coeffs[b] * A_block_b[r]
            # where r in 0,1,2,3 corresponds to positions within the 2x2 block
            for r in range(4):
                for b in range(4):
                    if a_coeffs[b] != 0:
                        # A_block_b[r] is at flat index block_A_indices[(b//2, b%2)][r]
                        row_block = b // 2
                        col_block = b % 2
                        flat_idx = block_A_indices[(row_block, col_block)][r]
                        U[flat_idx][col] += a_coeffs[b] * u_inner[r]
            
            # V contribution: similar for B
            for r in range(4):
                for b in range(4):
                    if b_coeffs[b] != 0:
                        row_block = b // 2
                        col_block = b % 2
                        flat_idx = block_B_indices[(row_block, col_block)][r]
                        V[flat_idx][col] += b_coeffs[b] * v_inner[r]
            
            # W contribution: 
            # The inner w_inner gives contributions to c00,c01,c10,c11 (positions 0,1,2,3 in inner 2x2)
            # But wait - I defined strassen_2x2_terms w as [(inner_c_idx, coeff)]
            # where inner_c_idx 0=c00, 1=c01, 2=c10, 3=c11
            # And c_contribs tells us which top-level C block this maps to and with what multiplier.
            
            # Hmm, I think I conflated things. Let me reconsider.
            # 
            # The top-level Sx product produces a 2x2 result. This 2x2 result contributes to
            # specific C blocks. For example, S1 contributes to C00 and C11.
            # The inner c00 of S1 goes to C00[0,0] and C11[0,0]? No...
            
            # Actually, I think the structure is:
            # C00 (as a 2x2 matrix) gets contributions from S1, S4, S5, S7
            # Each contribution is the FULL 2x2 result of that product, scaled.
            
            # Wait no. Let me re-examine. In the standard Strassen formula for block 2x2*2x2:
            # C_block00 = S1_block + S4_block - S5_block + S7_block  (matrix equation, each Sx is 2x2)
            
            # So each Sx produces a full 2x2 matrix, and C00 = S1 + S4 - S5 + S7 (matrix addition).
            # The inner Strassen decomposes each Sx (which is a 2x2*2x2 product) into 7 rank-1 terms.
            # Each rank-1 term contributes to all 4 entries of the resulting 2x2.
            
            # So for S1 with inner term m_k, the contribution to C00 is:
            #   C00[i,j] += (coefficient from w_inner for (i,j)) * (coefficient from top-level for S1 in C00)
            
            # Let me redo. The inner Strassen for 2x2*2x2 gives:
            # result[0,0] = m1 + m4 - m5 + m7
            # result[0,1] = m3 + m5
            # result[1,0] = m2 + m4
            # result[1,1] = m1 - m2 + m3 + m6
            
            # So for inner term k (0-indexed, m_k):
            # Its contribution to result[r,c] is given by a coefficient.
            
            # Let me define inner_w_matrix[k] = [[coeff_00, coeff_01], [coeff_10, coeff_11]]
            
            inner_w = [
                [[1, 0], [0, 1]],    # m1: c00+=1, c11+=1
                [[0, 0], [1, -1]],   # m2: c10+=1, c11-=1
                [[0, 1], [0, 1]],    # m3: c01+=1, c11+=1
                [[1, 0], [1, 0]],    # m4: c00+=1, c10+=1
                [[-1, 1], [0, 0]],   # m5: c00-=1, c01+=1
                [[0, 0], [0, 1]],    # m6: c11+=1
                [[1, 0], [0, 0]],    # m7: c00+=1
            ]
            
            # Wait, I already have w_inner from strassen_2x2_terms. Let me use a cleaner formulation.
            pass
    
    # OK this is getting complicated. Let me take a step back and use a cleaner approach.
    
    # I'll build the 49-term decomposition more carefully.
    
    # For the recursive Strassen, the total rank is 49.
    # I'll enumerate all 49 terms explicitly.
    
    # Top-level: 7 terms, each with 7 inner terms.
    # For top-level term t (0..6) and inner term i (0..6):
    #   col = t*7 + i
    #   
    #   U[:, col]: For each A-block position, determined by a_coeffs[t] and u_inner[i]
    #   V[:, col]: Similarly for B
    #   W[:, col]: For each C position, determined by the top-level C-block mapping and inner_w[i]
    
    # Let me define the top-level C-block mapping more carefully.
    # Each S_t contributes to certain C blocks with certain 2x2 coefficient matrices.
    
    # S1 = (A00+A11)*(B00+B11): contributes +1*I2 to C00 and +1*I2 to C11
    # Actually no, it contributes the full 2x2 result. So:
    # C00 += S1_result (full 2x2)
    # C11 += S1_result (full 2x2)
    
    # So for each C block that S_t maps to, the entire 2x2 result is added.
    
    # Top-level mappings (S_t -> [(C_block, scale_matrix)]):
    # The scale is always identity since we're adding the full result.
    
    top_to_c = [
        # S1
        [( (0,0), 1), ((1,1), 1)],
        # S2  
        [( (1,0), 1), ((1,1), -1)],
        # S3
        [( (0,1), 1), ((1,1), 1)],
        # S4
        [( (0,0), 1), ((1,0), 1)],
        # S5
        [( (0,0), -1), ((0,1), 1)],
        # S6
        [( (1,1), 1)],
        # S7
        [( (0,0), 1)],
    ]
    
    # Now rebuild:
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for t in range(7):
        a_coeffs, b_coeffs, _ = top_level_terms[t]
        c_maps = top_to_c[t]
        
        for i in range(7):
            u_inner, v_inner, _ = strassen_2x2_terms[i]
            col = t * 7 + i
            
            # U
            for r in range(4):
                for b in range(4):
                    ca = a_coeffs[b]
                    if ca != 0:
                        rb, cb = b // 2, b % 2
                        flat = block_A_indices[(rb, cb)][r]
                        U[flat][col] += ca * u_inner[r]
            
            # V
            for r in range(4):
                for b in range(4):
                    cb = b_coeffs[b]
                    if cb != 0:
                        rb, cbb = b // 2, b % 2
                        flat = block_B_indices[(rb, cbb)][r]
                        V[flat][col] += cb * v_inner[r]
            
            # W
            # inner_w[i] is a 2x2 matrix giving contribution to result[r,c]
            iw = inner_w[i]
            for (cb_row, cb_col), scale in c_maps:
                # C block (cb_row, cb_col) gets the full 2x2 result scaled by `scale`
                for rr in range(2):
                    for cc in range(2):
                        wcoeff = scale * iw[rr][cc]
                        if wcoeff != 0:
                            # C block (cb_row, cb_col) entry (rr, cc) -> global index
                            flat_c = block_C_indices[(cb_row, cb_col)][rr*2 + cc]
                            # Wait, block_C_indices maps to [c00, c01, c10, c11] positions
                            # Position 0=c[0,0], 1=c[0,1], 2=c[1,0], 3=c[1,1]
                            pos = rr * 2 + cc  # 0,1 for row 0; 2,3 for row 1
                            flat_c = block_C_indices[(cb_row, cb_col)][pos]
                            W[flat_c][col] += wcoeff
    
    return U, V, W


def _build_schoolbook_443():
    """Build schoolbook decomposition for 4x4 * 4x3.
    
    Returns U (16x48), V (12x48), W (12x48).
    n=4, m=4, p=3.
    U[i*4+j, k] = delta for A[i,j]
    V[j*3+k, k] = delta for B[j,k]  
    W[k*4+i, k] = delta for C[i,k]
    """
    n, m, p = 4, 4, 3
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"], dtype=object)
    V = np.array(result["V"], dtype=object)
    W = np.array(result["W"], dtype=object)
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    print(f"Rank: {U.shape[1]}")
    
    # Verify
    n, m, p = 4, 4, 7
    # C[i,k] = sum_c U[i*m+_, c] * V[_*p+k, c] ... 
    # Actually the convention is:
    # U: (n*m, rank) - indexes A[i,j] as i*m+j
    # V: (m*p, rank) - indexes B[j,k] as j*p+k
    # W: (p*n, rank) - indexes C[i,k] as k*n+i (trace convention W[k*n+i, r])
    
    # C[i,k] = sum_r U[A_idx(i,j), r] * V[B_idx(j,k), r] * W[C_idx(i,k), r] ... 
    # No, it's: the tensor T = sum_r U_r \otimes V_r \otimes W_r
    # T_{ijk} = sum_r U[i,j]_r * V[j,k]_r * W[i,k]_r
    
    # Verify: T_{ijk} should equal delta (i.e., 1 if this is a multiplication tensor entry)
    # The multiplication tensor has T_{ijk} = 1 for all valid (i,j,k), representing A[i,j]*B[j,k] contributing to C[i,k].
    
    # Let's verify numerically with random matrices.
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B normally
    C_ref = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    # Compute via decomposition
    C_decomp = [[0] * p for _ in range(n)]
    for r in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                u_val = float(U[i * m + j][r])
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = float(V[j * p + k][r])
                    if v_val == 0:
                        continue
                    w_val = float(W[k * n + i][r])
                    if w_val == 0:
                        continue
                    C_decomp[i][k] += u_val * v_val * w_val
    
    max_err = max(abs(C_ref[i][k] - C_decomp[i][k]) for i in range(n) for k in range(p))
    print(f"Max error: {max_err}")
    print("PASS" if max_err < 1e-9 else "FAIL")

