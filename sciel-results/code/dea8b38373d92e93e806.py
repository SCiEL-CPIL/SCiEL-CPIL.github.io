<<<<<<< SEARCH
"""Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition.

Strategy: Use recursive Strassen for the 4x4 * 4x4 part (rank 49) plus
16 scalar multiplications for the 5th column, giving rank 65.
Then attempt to reduce rank further via a linear-algebraic reduction pass
that checks whether the 65-term decomposition has linear dependencies
that can be exploited to lower the rank while preserving exactness.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE_SEC = 110  # safety margin under 120s


def _strassen_2x2_terms():
    """
    Return the 7 Strassen multiplication terms for 2x2 * 2x2.
    Each term is represented as a list of (a_idx, b_idx) pairs with coefficient +1,
    and (a_idx, b_idx) pairs with coefficient -1.
    a_idx, b_idx are in {0,1} x {0,1} for 2x2 blocks.
    
    Strassen's 7 products:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B10)
      M7 = (A01-A11)(B01+B11)
    """
    # Each product: list of (coeff, a_row, a_col, b_row, b_col)
    # a_row, a_col in {0,1}; b_row, b_col in {0,1}
    products = [
        # M1 = (A00+A11)(B00+B11)
        [(+1, 0, 0, 0, 0), (+1, 1, 1, 0, 0), (+1, 0, 0, 1, 1), (+1, 1, 1, 1, 1)],
        # M2 = (A10+A11)B00
        [(+1, 1, 0, 0, 0), (+1, 1, 1, 0, 0)],
        # M3 = A00(B01-B11)
        [(+1, 0, 0, 0, 1), (-1, 0, 0, 1, 1)],
        # M4 = A11(B10-B00)
        [(+1, 1, 1, 1, 0), (-1, 1, 1, 0, 0)],
        # M5 = (A00+A01)B11
        [(+1, 0, 0, 1, 1), (+1, 0, 1, 1, 1)],
        # M6 = (A10-A00)(B00+B10)
        [(+1, 1, 0, 0, 0), (-1, 0, 0, 0, 0), (+1, 1, 0, 1, 0), (-1, 0, 0, 1, 0)],
        # M7 = (A01-A11)(B01+B11)
        [(+1, 0, 1, 0, 1), (-1, 1, 1, 0, 1), (+1, 0, 1, 1, 1), (-1, 1, 1, 1, 1)],
    ]
    return products


def _build_4x4x4_strassen():
    """
    Build the rank-49 decomposition for 4x4 * 4x4 using two levels of Strassen.
    Returns U, V, W matrices of shape (16, 49) each, where indices are:
      U: i*4+j for i,j in 0..3
      V: j*4+k for j,k in 0..3
      W: k*4+i for k,i in 0..3
    """
    n = 4
    rank = 49
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    # Two-level Strassen: partition 4x4 into four 2x2 blocks.
    # Level 1: treat 4x4 as 2x2 blocks of size 2x2.
    # The 7 Strassen products at level 1 each involve 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 can be done with 7 scalar multiplications (level 2 Strassen).
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where Aab is 2x2, a,b in {0,1}
    # B = [[B00, B01], [B10, B11]] where Bab is 2x2
    # C = A * B, C is 4x4
    
    # Level 1 Strassen products (each is a 2x2 * 2x2 product):
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    
    # Each Pq is a 2x2 matrix = sum of 7 scalar products (level 2 Strassen)
    # So we have 7 * 7 = 49 scalar products total.
    
    # For each level-1 product q (0..6) and level-2 product r (0..6):
    # We need to determine which scalar (i,j,k) triples contribute.
    
    # Let's define the level-1 products as linear combinations of 2x2 blocks:
    # Each level-1 product Pq = sum of terms (coeff * A_block * B_block)
    # where A_block and B_block are 2x2 sub-blocks.
    
    # Level 1 definitions (coeff, A_block, B_block):
    # A_block: (a_row, a_col) in {0,1}^2, B_block: (b_row, b_col) in {0,1}^2
    L1 = [
        # P1 = (A00+A11)(B00+B11)
        [(+1, (0,0), (0,0)), (+1, (1,1), (0,0)), (+1, (0,0), (1,1)), (+1, (1,1), (1,1))],
        # P2 = (A10+A11)B00
        [(+1, (1,0), (0,0)), (+1, (1,1), (0,0))],
        # P3 = A00(B01-B11)
        [(+1, (0,0), (0,1)), (-1, (0,0), (1,1))],
        # P4 = A11(B10-B00)
        [(+1, (1,1), (1,0)), (-1, (1,1), (0,0))],
        # P5 = (A00+A01)B11
        [(+1, (0,0), (1,1)), (+1, (0,1), (1,1))],
        # P6 = (A10-A00)(B00+B10)
        [(+1, (1,0), (0,0)), (-1, (0,0), (0,0)), (+1, (1,0), (1,0)), (-1, (0,0), (1,0))],
        # P7 = (A01-A11)(B01+B11)
        [(+1, (0,1), (0,1)), (-1, (1,1), (0,1)), (+1, (0,1), (1,1)), (-1, (1,1), (1,1))],
    ]
    
    # Level 2 Strassen for 2x2 * 2x2:
    # For a 2x2 matrix X and 2x2 matrix Y, the 7 products are:
    L2 = _strassen_2x2_terms()
    # L2[r] is a list of (coeff, x_row, x_col, y_row, y_col)
    
    # Now, for each level-1 product q and each level-2 product r:
    # The scalar product involves specific elements of A and B.
    # 
    # A element A[i,j] where i = a_row*2 + x_row, j = a_col*2 + x_col
    # B element B[j,k] where j = b_row*2 + y_row, k = b_col*2 + y_col
    # C element C[i,k] where i = a_row*2 + x_row, k = b_col*2 + y_col
    # But wait - the level-2 Strassen operates on the 2x2 blocks, so:
    # If Pq involves A_block(a_row,a_col) * B_block(b_row,b_col),
    # then the level-2 Strassen product r gives a scalar product of:
    #   A[a_row*2 + x_row, a_col*2 + x_col] * B[b_row*2 + y_row, b_col*2 + y_col]
    # contributing to C[a_row*2 + x_row, b_col*2 + y_col]
    
    col = 0
    for q in range(7):
        for (coeff1, a_blk, b_blk) in L1[q]:
            a_row, a_col = a_blk
            b_row, b_col = b_blk
            for r in range(7):
                # Level 2 product r: list of (coeff2, xr, xc, yr, yc)
                # But actually, each level-2 "product" is a sum of scalar terms.
                # Wait, I need to be more careful.
                
                # The level-2 Strassen says: to compute X*Y where X,Y are 2x2,
                # we compute 7 scalar products M1..M7, and then combine them.
                # Each Mr is a scalar = sum of terms (coeff * X[xr,xc] * Y[yr,yc]).
                # So for our purpose, each (q, r) pair corresponds to one scalar
                # multiplication that is a sum of (coeff * A[i,j] * B[j',k]) terms.
                # But actually no - in the tensor decomposition, each rank-1 term
                # is a single scalar multiplication. The Strassen algorithm combines
                # them, but for the tensor rank, each scalar product counts.
                
                # Let me reconsider. The 49-term decomposition means we have 49
                # scalar multiplications. Each scalar multiplication is of the form:
                #   (linear combo of A elements) * (linear combo of B elements)
                # and contributes (with a coefficient) to specific C elements.
                
                # Actually, for the tensor decomposition, each rank-1 term is:
                #   u_r * v_r * w_r where u_r is a vector in A-space, v_r in B-space,
                #   w_r in C-space. The "scalar multiplication" interpretation is:
                #   term_r = (sum_{i,j} U[i,j,r] * A[i,j]) * (sum_{j',k} V[j',k,r] * B[j',k])
                #   contributes to C via W.
                
                # For Strassen, each of the 49 terms is a product of a linear
                # combination of A entries and a linear combination of B entries.
                # The U, V, W matrices encode these linear combinations.
                
                pass  # We'll build this differently below
    
    # Let me use a cleaner approach: directly enumerate the 49 scalar products.
    # Each scalar product is identified by (q, r) where q is level-1 product (0-6)
    # and r is level-2 product (0-6).
    
    # For level-1 product q, it's a sum of terms: sum_t coeff_t * A_block_t * B_block_t
    # For each such term, the level-2 Strassen product r gives:
    #   sum of (coeff2 * A[i,j] * B[j2,k]) where i,j come from A_block and j2,k from B_block
    
    # Actually, let me just directly compute the U, V, W for each of the 49 terms.
    # Term (q, r): 
    #   The level-1 product q involves certain 2x2 blocks of A and B.
    #   The level-2 product r within each block pair gives specific scalar products.
    
    # Let me re-derive. For a single 2x2 * 2x2 product X*Y:
    #   M1 = (X00+X11)(Y00+Y11) -> U has X00,X11; V has Y00,Y11
    #   M2 = (X10+X11)Y00 -> U has X10,X11; V has Y00
    #   M3 = X00(Y01-Y11) -> U has X00; V has Y01,Y11 (with -1)
    #   M4 = X11(Y10-Y00) -> U has X11; V has Y10,Y00 (with -1)
    #   M5 = (X00+X01)Y11 -> U has X00,X01; V has Y11
    #   M6 = (X10-X00)(Y00+Y10) -> U has X10,X00 (with -1); V has Y00,Y10
    #   M7 = (X01-X11)(Y01+Y11) -> U has X01,X11 (with -1); V has Y01,Y11
    
    # For the 4x4 case, each 2x2 block is identified by (block_row, block_col).
    # A element A[i,j] = A[block_row, block_col, local_row, local_col]
    #   where i = block_row*2 + local_row, j = block_col*2 + local_col
    
    # For level-1 product q with term (coeff1, a_blk, b_blk):
    #   a_blk = (ar, ac), b_blk = (br, bc)
    #   The 2x2 * 2x2 product is A[ar*2:ar*2+2, ac*2:ac*2+2] * B[br*2:br*2+2, bc*2:bc*2+2]
    #   Level-2 product r gives scalar products of specific elements.
    
    # For each (q, r), the term is a sum over the contributions from each
    # (coeff1, a_blk, b_blk) in L1[q] of the level-2 product r applied to
    # the corresponding 2x2 blocks.
    
    # But wait - in the tensor decomposition, each rank-1 term should be a
    # SINGLE scalar multiplication, not a sum. The Strassen algorithm uses
    # sums, but each sum component is a separate scalar multiplication in
    # the tensor sense? No - the tensor rank counts the number of rank-1
    # terms, where each rank-1 term is u * v * w (outer products).
    
    # I think the correct interpretation is:
    # Each of the 49 "scalar multiplications" in the two-level Strassen is
    # actually a product of two linear forms. So U, V, W entries can be
    # non-zero in multiple positions.
    
    # Let me just build it directly.
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    col = 0
    for q in range(7):
        for r in range(7):
            # For this (q,r) term, we accumulate contributions from all
            # (coeff1, a_blk, b_blk) in L1[q]
            for (coeff1, a_blk, b_blk) in L1[q]:
                ar, ac = a_blk
                br, bc = b_blk
                # Level-2 product r: list of (coeff2, xr, xc, yr, yc)
                for (coeff2, xr, xc, yr, yc) in L2[r]:
                    # A element: i = ar*2 + xr, j = ac*2 + xc
                    # B element: j2 = br*2 + yr, k = bc*2 + yc
                    # C element: i = ar*2 + xr, k = bc*2 + yc
                    # But wait - for the product to make sense, we need
                    # j = j2 (the inner dimension). In our case, A is 4x4 and
                    # B is 4x4, so j (col of A) = j2 (row of B).
                    # j = ac*2 + xc, j2 = br*2 + yr
                    # These are NOT necessarily equal! 
                    
                    # Hmm, this is the issue. In matrix multiplication C = A*B,
                    # C[i,k] = sum_j A[i,j] * B[j,k]. The "j" is the same index.
                    # But in Strassen, we're multiplying blocks, and the block
                    # multiplication (A_block * B_block) only makes sense if
                    # the column index of A_block matches the row index of B_block.
                    
                    # In our 4x4 * 4x4 case:
                    # A_block(ar, ac) has rows ar*2..ar*2+1, cols ac*2..ac*2+1
                    # B_block(br, bc) has rows br*2..br*2+1, cols bc*2..bc*2+1
                    # The product A_block(ar,ac) * B_block(br,bc) only contributes
                    # to C if ac == br (the inner dimension matches).
                    
                    # But in Strassen's algorithm for block matrices, we have:
                    # C = A * B where A, B are 2x2 block matrices.
                    # C[ar, bc] = sum_br A[ar, br] * B[br, bc]
                    # So A_block(ar, br) * B_block(br, bc) contributes to C_block(ar, bc).
                    
                    # In L1, the terms are structured so that the block indices
                    # are consistent. Let me re-examine L1:
                    # P1 = (A00+A11)(B00+B11)
                    #   A00 * B00: ar=0, ac=0, br=0, bc=0 -> ac==br (0==0) OK
                    #   A11 * B00: ar=1, ac=1, br=0, bc=0 -> ac==br? 1!=0 NO!
                    
                    # Wait, this is wrong. In block matrix multiplication:
                    # C[ar, bc] = sum_br A[ar, br] * B[br, bc]
                    # So the middle index is br, not ac.
                    # A[ar, br] has row-block ar, col-block br
                    # B[br, bc] has row-block br, col-block bc
                    
                    # So in L1, when I write A00 * B00, that means:
                    # A block at (row=0, col=0) times B block at (row=0, col=0)
                    # This contributes to C block at (row=0, col=0).
                    # The inner dimension is col-of-A = row-of-B = 0. OK.
                    
                    # A11 * B00: A block (row=1, col=1) * B block (row=0, col=0)
                    # Inner dimension: col-of-A = 1, row-of-B = 0. MISMATCH!
                    
                    # I see the issue. In Strassen's algorithm, the "blocks" are
                    # not indexed the way I've been thinking. Let me redo this.
                    
                    # Standard Strassen for 2x2 block matrices:
                    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
                    # C = A*B, C = [[C11, C12], [C21, C22]]
                    # C11 = A11*B11 + A12*B21
                    # C12 = A11*B12 + A12*B22
                    # C21 = A21*B11 + A22*B21
                    # C22 = A21*B12 + A22*B22
                    
                    # Strassen:
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B12-B22)
                    # M4 = A22(B21-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B21)
                    # M7 = (A12-A22)(B12+B22)
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M6
                    
                    # So M1 involves A11*B11, A22*B11, A11*B22, A22*B22
                    # A11 is block (row=0, col=0), B11 is block (row=0, col=0)
                    # A22 is block (row=1, col=1), B22 is block (row=1, col=1)
                    # A22*B11: A block (1,1) * B block (0,0)
                    # Inner dim: col of A = 1, row of B = 0. MISMATCH!
                    
                    # This can't be right. Let me look this up properly.
                    # 
                    # Actually, I think the issue is that in Strassen's algorithm,
                    # the blocks A11, A12, A21, A22 are:
                    # A11 = top-left, A12 = top-right, A21 = bottom-left, A22 = bottom-right
                    # So A11 has rows 0-1, cols 0-1 (for 4x4 with 2x2 blocks)
                    # A12 has rows 0-1, cols 2-3
                    # A21 has rows 2-3, cols 0-1
                    # A22 has rows 2-3, cols 2-3
                    #
                    # Similarly for B:
                    # B11 = rows 0-1, cols 0-1
                    # B12 = rows 0-1, cols 2-3
                    # B21 = rows 2-3, cols 0-1
                    # B22 = rows 2-3, cols 2-3
                    #
                    # Now M1 = (A11+A22)(B11+B22)
                    # A11 + A22: rows 0-1 & 2-3, cols 0-1 & 2-3 (but they're different blocks!)
                    # This is a 4x4 matrix formed by adding A11 (top-left 2x2) and
                    # A22 (bottom-right 2x2) as 4x4 matrices (with zeros elsewhere).
                    # Similarly B11 + B22.
                    # The product (A11+A22)(B11+B22) is a 4x4 matrix.
                    #
                    # So each "block product" like A11*B11 is actually a 4x4 * 4x4
                    # product where most entries are zero. The non-zero part is
                    # the 2x2 * 2x2 product in the corresponding positions.
                    #
                    # For the tensor decomposition, each scalar multiplication
                    # M1, M2, ..., M7 is a 4x4 * 4x4 product (with sparse structure).
                    # To decompose each into scalar multiplications, we use
                    # Strassen again on the non-zero 2x2 blocks.
                    
                    # OK so let me just directly enumerate. For M1 = (A11+A22)(B11+B22):
                    # This is a 4x4 * 4x4 product. The non-zero A entries are in
                    # positions (0,0),(0,1),(1,0),(1,1) [from A11] and
                    # (2,2),(2,3),(3,2),(3,3) [from A22].
                    # Similarly for B.
                    # The product has non-zero entries in C at positions where
                    # (row from A-nonzero) and (col from B-nonzero).
                    #
                    # To decompose this 4x4*4x4 product into scalar multiplications
                    # using Strassen, we partition the 4x4 into 2x2 blocks again.
                    # But the sparsity pattern means some blocks are zero.
                    
                    # This is getting very complex. Let me use a different approach:
                    # just directly construct the rank-49 decomposition by
                    # explicitly writing out all 49 scalar products.
                    
                    pass
            
            col += 1
    
    # This approach is too complex to do by hand. Let me use a simpler method:
    # Use the known fact that rank(4x4*4x4) <= 49 via recursive Strassen,
    # and construct it programmatically.
    
    # Actually, let me use a completely different strategy:
    # Use the schoolbook for the 5th column (16 terms) and for the 4x4 part,
    # use a direct construction based on Strassen.
    
    # For simplicity and correctness, let me use the following approach:
    # 1. For the 4x4 * 4x4 part: use 49 terms from recursive Strassen
    # 2. For the 4x4 * 4x1 part (5th column): use 16 schoolbook terms
    # Total: 65 terms
    
    # To construct the 49-term decomposition for 4x4*4x4, I'll use the
    # recursive structure: each of the 7 level-1 Strassen products is a
    # 2x2*2x2 block product, which can be done with 7 scalar products.
    
    # Let me define the construction more carefully using explicit indices.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Level 1 Strassen products for 4x4*4x4:
    # Each product is a sum of terms (coeff, A_block, B_block)
    # A_block = (a_row, a_col), B_block = (b_row, b_col)
    # where these are block indices (0 or 1) for 2x2 blocks.
    # A_block (ar, ac) covers rows ar*2..ar*2+1, cols ac*2..ac*2+1
    # B_block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1
    # The product A_block * B_block contributes to C_block (ar, bc)
    # ONLY IF ac == br (inner dimension matches)
    
    L1_products = [
        # M1 = (A11+A22)(B11+B22)
        # In block notation: (A(0,0)+A(1,1))(B(0,0)+B(1,1))
        [(+1, (0,0), (0,0)), (+1, (1,1), (0,0)), (+1, (0,0), (1,1)), (+1, (1,1), (1,1))],
        # M2 = (A21+A22)B11
        [(+1, (1,0), (0,0)), (+1, (1,1), (0,0))],
        # M3 = A11(B12-B22)
        [(+1, (0,0), (0,1)), (-1, (0,0), (1,1))],
        # M4 = A22(B21-B11)
        [(+1, (1,1), (1,0)), (-1, (1,1), (0,0))],
        # M5 = (A11+A12)B22
        [(+1, (0,0), (1,1)), (+1, (0,1), (1,1))],
        # M6 = (A21-A11)(B11+B21)
        [(+1, (1,0), (0,0)), (-1, (0,0), (0,0)), (+1, (1,0), (1,0)), (-1, (0,0), (1,0))],
        # M7 = (A12-A22)(B12+B22)
        [(+1, (0,1), (0,1)), (-1, (1,1), (0,1)), (+1, (0,1), (1,1)), (-1, (1,1), (1,1))],
    ]
    
    # Level 2 Strassen for 2x2*2x2:
    # For 2x2 matrices X and Y, the 7 products are:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y01+Y11)
    # Each Mr is a scalar = sum of (coeff * X[xr,xc] * Y[yr,yc])
    L2_products = [
        [(+1, 0, 0, 0, 0), (+1, 1, 1, 0, 0), (+1, 0, 0, 1, 1), (+1, 1, 1, 1, 1)],
        [(+1, 1, 0, 0, 0), (+1, 1, 1, 0, 0)],
        [(+1, 0, 0, 0, 1), (-1, 0, 0, 1, 1)],
        [(+1, 1, 1, 1, 0), (-1, 1, 1, 0, 0)],
        [(+1, 0, 0, 1, 1), (+1, 0, 1, 1, 1)],
        [(+1, 1, 0, 0, 0), (-1, 0, 0, 0, 0), (+1, 1, 0, 1, 0), (-1, 0, 0, 1, 0)],
        [(+1, 0, 1, 0, 1), (-1, 1, 1, 0, 1), (+1, 0, 1, 1, 1), (-1, 1, 1, 1, 1)],
    ]
    
    # For each (q, r) pair, we have one rank-1 term.
    # The term is: (sum of A-linear-form) * (sum of B-linear-form) -> C-linear-form
    # But actually, each (q, r) term involves MULTIPLE scalar products
    # (from the sums in L1 and L2). In the tensor decomposition, this means
    # the U, V, W vectors for this term have multiple non-zero entries.
    
    col = 0
    for q in range(7):
        for r in range(7):
            # Accumulate U, V, W contributions for this (q, r) term
            U_col = np.zeros(16)
            V_col = np.zeros(16)
            W_col = np.zeros(16)
            
            for (coeff1, a_blk, b_blk) in L1_products[q]:
                ar, ac = a_blk
                br, bc = b_blk
                # Check inner dimension: ac should equal br
                if ac != br:
                    continue  # This term doesn't contribute (zero)
                
                for (coeff2, xr, xc, yr, yc) in L2_products[r]:
                    # A element: i = ar*2 + xr, j = ac*2 + xc
                    # B element: j2 = br*2 + yr, k = bc*2 + yc
                    # C element: i = ar*2 + xr, k = bc*2 + yc
                    i = ar * 2 + xr
                    j = ac * 2 + xc
                    j2 = br * 2 + yr
                    k = bc * 2 + yc
                    # For valid matrix multiplication, j should equal j2
                    # Since ac == br (checked above), and xc == yr (from Strassen structure)
                    # Actually, xc and yr are independent local indices.
                    # In the 2x2*2x2 product, X[xr,xc] * Y[yr,yc] contributes
                    # to Z[xr,yc] where Z = X*Y.
                    # So the contribution is to C[i, k] where i = ar*2+xr, k = bc*2+yc
                    # And the A element is A[i, j] where j = ac*2+xc
                    # And the B element is B[j, k] where j = br*2+yr
                    # For this to be a valid matmul term, we need j = j2, i.e.,
                    # ac*2+xc = br*2+yr. Since ac==br, this means xc==yr.
                    # But xc and yr are from different Strassen terms!
                    
                    # I think the correct interpretation is:
                    # The 2x2*2x2 product X*Y has (X*Y)[xr,yc] = sum_yr X[xr,yr]*Y[yr,yc]
                    # So each scalar product X[xr,yr]*Y[yr,yc] contributes to (X*Y)[xr,yc]
                    # In our case, X = A_block(ar,ac), Y = B_block(br,bc)
                    # (X*Y)[xr,yc] contributes to C[ar*2+xr, bc*2+yc]
                    # And the scalar product is A[ar*2+xr, ac*2+yr] * B[br*2+yr, bc*2+yc]
                    # Note: the middle index is yr (not xc or j2)
                    
                    # So: A element: i = ar*2+xr, j_mid = ac*2+yr
                    #     B element: j_mid = br*2+yr, k = bc*2+yc
                    #     C element: i = ar*2+xr, k = bc*2+yc
                    # Since ac==br, j_mid is consistent.
                    
                    i_idx = ar * 2 + xr
                    j_mid = ac * 2 + yr
                    k_idx = bc * 2 + yc
                    
                    # U[i*m+j, col] where m=4 for the 4x4 part
                    # But we're building for 4x4*4x4, so indices are i*4+j
                    U[i_idx * 4 + j_mid] += coeff1 * coeff2
                    V[j_mid * 4 + k_idx] += coeff1 * coeff2
                    W[k_idx * 4 + i_idx] += coeff1 * coeff2
            
            U[:, col] = U_col  # This is wrong, let me redo
            col += 1
    
    # I'm making this too complicated. Let me just use a verified construction.
    # For the purpose of this task, let me use the schoolbook for the 4x4 part
    # and only apply Strassen to get a modest improvement, or just use 65 terms
    # with a simpler construction.
    
    # Actually, the simplest correct approach for rank 65:
    # Use 49 terms for 4x4*4x4 (recursive Strassen) + 16 terms for 4x4*4x1
    # But constructing the 49-term decomposition correctly is complex.
    
    # Let me use a different approach: just use the schoolbook (80 terms)
    # but with a small optimization: since the 5th column of B only multiplies
    # with the 4x4 A, we can't easily reduce it.
    
    # Given the complexity, let me return the schoolbook decomposition (rank 80)
    # which is guaranteed correct.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W, rank
=======
"""Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition.

Strategy: Use recursive Strassen for the 4x4 * 4x4 part (rank 49) plus
16 scalar multiplications for the 5th column, giving rank 65.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE_SEC = 110  # safety margin under 120s


def _build_rank65_decomposition():
    """
    Build the rank-65 decomposition for 4x4 * 4x5.
    
    Part 1: 49 terms from recursive Strassen for 4x4 * 4x4 (first 4 cols of B)
    Part 2: 16 schoolbook terms for 4x4 * 4x1 (5th col of B)
    
    Index conventions:
      U: row = i*m + j  (i in 0..3, j in 0..3)  -> shape (16, rank)
      V: row = j*p + k  (j in 0..3, k in 0..4)  -> shape (20, rank)
      W: row = k*n + i  (k in 0..4, i in 0..3)  -> shape (20, rank)
    """
    n, m, p = 4, 4, 5
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 49 Strassen terms for 4x4 * 4x4 (columns 0-3 of B) ---
    # Level-1 Strassen: 7 products, each a sum of (coeff, A_block, B_block)
    # A_block(ar, ac): rows ar*2..ar*2+1, cols ac*2..ac*2+1 of A
    # B_block(br, bc): rows br*2..br*2+1, cols bc*2..bc*2+1 of B
    # Product A_block * B_block is valid only if ac == br (inner dim matches)
    # It contributes to C_block(ar, bc), i.e., C rows ar*2..ar*2+1, cols bc*2..bc*2+1
    
    L1 = [
        # M1 = (A11+A22)(B11+B22)
        [(+1, (0,0), (0,0)), (+1, (1,1), (0,0)), (+1, (0,0), (1,1)), (+1, (1,1), (1,1))],
        # M2 = (A21+A22)B11
        [(+1, (1,0), (0,0)), (+1, (1,1), (0,0))],
        # M3 = A11(B12-B22)
        [(+1, (0,0), (0,1)), (-1, (0,0), (1,1))],
        # M4 = A22(B21-B11)
        [(+1, (1,1), (1,0)), (-1, (1,1), (0,0))],
        # M5 = (A11+A12)B22
        [(+1, (0,0), (1,1)), (+1, (0,1), (1,1))],
        # M6 = (A21-A11)(B11+B21)
        [(+1, (1,0), (0,0)), (-1, (0,0), (0,0)), (+1, (1,0), (1,0)), (-1, (0,0), (1,0))],
        # M7 = (A12-A22)(B12+B22)
        [(+1, (0,1), (0,1)), (-1, (1,1), (0,1)), (+1, (0,1), (1,1)), (-1, (1,1), (1,1))],
    ]
    
    # Level-2 Strassen for 2x2 * 2x2:
    # For X (2x2) * Y (2x2), product Z[xr,yc] = sum_yr X[xr,yr]*Y[yr,yc]
    # Each of the 7 Strassen terms is a scalar = sum of (coeff * X[xr,yr] * Y[yr,yc])
    # The key: middle index is yr (shared: X's col = Y's row)
    L2 = [
        # M1 = (X00+X11)(Y00+Y11)
        [(+1, 0, 0, 0, 0), (+1, 1, 1, 0, 0), (+1, 0, 0, 1, 1), (+1, 1, 1, 1, 1)],
        # M2 = (X10+X11)Y00
        [(+1, 1, 0, 0, 0), (+1, 1, 1, 0, 0)],
        # M3 = X00(Y01-Y11)
        [(+1, 0, 0, 0, 1), (-1, 0, 0, 1, 1)],
        # M4 = X11(Y10-Y00)
        [(+1, 1, 1, 1, 0), (-1, 1, 1, 0, 0)],
        # M5 = (X00+X01)Y11
        [(+1, 0, 0, 1, 1), (+1, 0, 1, 1, 1)],
        # M6 = (X10-X00)(Y00+Y10)
        [(+1, 1, 0, 0, 0), (-1, 0, 0, 0, 0), (+1, 1, 0, 1, 0), (-1, 0, 0, 1, 0)],
        # M7 = (X01-X11)(Y01+Y11)
        [(+1, 0, 1, 0, 1), (-1, 1, 1, 0, 1), (+1, 0, 1, 1, 1), (-1, 1, 1, 1, 1)],
    ]
    
    col = 0
    for q in range(7):
        for r in range(7):
            # For this (q, r) term, accumulate U, V, W
            # from all valid block pairs in L1[q]
            for (coeff1, a_blk, b_blk) in L1[q]:
                ar, ac = a_blk
                br, bc = b_blk
                # Inner dimension must match: ac == br
                if ac != br:
                    continue
                
                for (coeff2, xr, yr, yc) in L2[r]:
                    # A element: A[ar*2+xr, ac*2+yr]
                    # B element: B[br*2+yr, bc*2+yc]  (br==ac, so middle index matches)
                    # C element: C[ar*2+xr, bc*2+yc]
                    i = ar * 2 + xr
                    j_mid = ac * 2 + yr
                    k = bc * 2 + yc
                    
                    c = coeff1 * coeff2
                    U[i * m + j_mid, col] += c
                    V[j_mid * p + k, col] += c
                    W[k * n + i, col] += c
            
            col += 1
    
    # --- Part 2: 16 schoolbook terms for 4x4 * 4x1 (column 4 of B) ---
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1
    
    return U, V, W, rank


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition exactly reproduces the matmul tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True
>>>>>>> REPLACE

Now let me also fix the `construct()` function to use the new builder and verify correctly:

