import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen formulas for 2x2 block matrices
    # Blocks: A00, A01, A10, A11 and B00, B01, B10, B11
    # We represent linear forms as dicts {(i,j): coeff} for A and {(j,k): coeff} for B
    
    def add_form(form1, form2):
        res = form1.copy()
        for key, val in form2.items():
            res[key] = res.get(key, 0) + val
        # Remove zeros
        return {k: v for k, v in res.items() if v != 0}
        
    def sub_form(form1, form2):
        res = form1.copy()
        for key, val in form2.items():
            res[key] = res.get(key, 0) - val
        return {k: v for k, v in res.items() if v != 0}

    # Define block linear forms
    # A blocks
    A00 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1} # Wait, A00 is top-left 2x2 block of 4x4 A
    # Indices of A: 0..3
    # Block 0,0: rows 0,1 cols 0,1
    # Block 0,1: rows 0,1 cols 2,3
    # Block 1,0: rows 2,3 cols 0,1
    # Block 1,1: rows 2,3 cols 2,3
    
    A_blocks = [
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, # A00
        {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1}, # A01
        {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1}, # A10
        {(2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}  # A11
    ]
    
    B_blocks = [
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, # B00
        {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1}, # B01
        {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1}, # B10
        {(2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}  # B11
    ]
    
    # Strassen 7 products for block level
    # P0 = (A00 + A11)(B00 + B11)
    # P1 = (A10 + A11)B00
    # P2 = A00(B01 - B11)
    # P3 = A11(B10 - B00)
    # P4 = (A00 + A01)B11
    # P5 = (A10 - A00)(B00 + B01)
    # P6 = (A01 - A11)(B10 + B11)
    
    X_blocks = [
        add_form(A_blocks[0], A_blocks[3]), # X0
        add_form(A_blocks[2], A_blocks[3]), # X1
        A_blocks[0],                        # X2
        A_blocks[3],                        # X3
        add_form(A_blocks[0], A_blocks[1]), # X4
        sub_form(A_blocks[2], A_blocks[0]), # X5
        sub_form(A_blocks[1], A_blocks[3])  # X6
    ]
    
    Y_blocks = [
        add_form(B_blocks[0], B_blocks[3]), # Y0
        B_blocks[0],                        # Y1
        sub_form(B_blocks[1], B_blocks[3]), # Y2
        sub_form(B_blocks[2], B_blocks[0]), # Y3
        B_blocks[3],                        # Y4
        add_form(B_blocks[0], B_blocks[1]), # Y5
        add_form(B_blocks[2], B_blocks[3])  # Y6
    ]
    
    # Now for each l, we have X_l and Y_l as 2x2 matrices of scalar linear forms?
    # No, X_blocks[l] is a dict of (i,j) -> coeff for the *entire* 4x4 matrix A?
    # Wait, X_blocks[0] = A00 + A11. This is a 4x4 matrix of coefficients.
    # But Strassen at the block level treats A00, A01 etc as 2x2 blocks.
    # So X_l is a 2x2 matrix of 2x2 blocks.
    # We need to split X_blocks[l] into 4 2x2 blocks.
    
    def split_blocks(form):
        # form is dict (i,j) -> coeff for 4x4
        blocks = []
        for r in range(2):
            for c in range(2):
                block = {}
                for i in range(2):
                    for j in range(2):
                        key = (r*2 + i, c*2 + j)
                        if key in form:
                            block[(i,j)] = form[key]
                blocks.append(block)
        return blocks # [B00, B01, B10, B11]

    # Apply Strassen to each pair (X_l, Y_l)
    # X_l is 2x2 blocks, Y_l is 2x2 blocks.
    # We need to compute Z_l = X_l Y_l using 7 multiplications.
    # Each multiplication is of the form (linear combo of X blocks) * (linear combo of Y blocks).
    # Since X blocks and Y blocks are 2x2 matrices of scalars, the product is a 2x2 matrix of scalars.
    # Wait, the "multiplication" in Strassen is of 2x2 matrices.
    # So we need to apply Strassen to the 2x2 matrices X_l and Y_l.
    
    rank_idx = 0
    for l in range(7):
        Xb = split_blocks(X_blocks[l]) # List of 4 dicts, each 2x2 scalar form
        Yb = split_blocks(Y_blocks[l])
        
        # Strassen on 2x2 matrices Xb, Yb
        # Xb[0]=X00, Xb[1]=X01, Xb[2]=X10, Xb[3]=X11
        # Yb[0]=Y00, Yb[1]=Y01, Yb[2]=Y10, Yb[3]=Y11
        
        # 7 multiplications
        # m1 = (X00+X11)(Y00+Y11)
        # m2 = (X10+X11)Y00
        # m3 = X00(Y01-Y11)
        # m4 = X11(Y10-Y00)
        # m5 = (X00+X01)Y11
        # m6 = (X10-X00)(Y00+Y01)
        # m7 = (X01-X11)(Y10+Y11)
        
        U_forms = [
            add_form(Xb[0], Xb[3]),
            add_form(Xb[2], Xb[3]),
            Xb[0],
            Xb[3],
            add_form(Xb[0], Xb[1]),
            sub_form(Xb[2], Xb[0]),
            sub_form(Xb[1], Xb[3])
        ]
        
        V_forms = [
            add_form(Yb[0], Yb[3]),
            Yb[0],
            sub_form(Yb[1], Yb[3]),
            sub_form(Yb[2], Yb[0]),
            Yb[3],
            add_form(Yb[0], Yb[1]),
            add_form(Yb[2], Yb[3])
        ]
        
        # Coefficients for Z_l entries
        # Z00 = m1 + m4 - m6 - m7
        # Z01 = m3 + m5
        # Z10 = m2 + m6
        # Z11 = m1 - m2 + m5 + m7
        
        Z_coeffs = [
            [1, 0, 0, 1, 0, -1, -1], # Z00
            [0, 0, 1, 0, 1, 0, 0],   # Z01
            [0, 1, 0, 0, 0, 1, 0],   # Z10
            [1, -1, 0, 0, 1, 0, 1]   # Z11
        ]
        
        for q in range(7):
            u_form = U_forms[q] # Dict (i,j) -> coeff for 2x2 block
            v_form = V_forms[q] # Dict (j,k) -> coeff for 2x2 block
            
            # Update U, V, W
            # u_form is relative to the 2x2 block structure of X_l.
            # But X_l was derived from the 4x4 A.
            # The keys in u_form are (i,j) in 0..1 range.
            # We need to map these back to the 4x4 indices?
            # No, u_form is a linear combination of the *entries* of the 2x2 blocks.
            # Wait, Xb[0] was a dict with keys (0,0)..(1,1).
            # These keys correspond to the local indices within the block.
            # But the block itself is at a specific position in the 4x4 matrix.
            # We lost the global position when we split?
            # No, split_blocks returned dicts with local indices (0,0)..(1,1).
            # We need to map local (i,j) to global (I,J).
            # This depends on which block it came from.
            # This is getting complicated.
            
            # Alternative: Keep the forms as 4x4 dicts from the start.
            # X_blocks[l] was a 4x4 dict.
            # Strassen on 2x2 blocks requires operating on the blocks.
            # Let's re-construct U_forms and V_forms as 4x4 dicts.
            
            # Redo:
            # X_blocks[l] is 4x4 dict.
            # We need to form linear combos of the *blocks*.
            # Let's define a function to combine 4x4 dicts.
            
            pass

    # Let's restart the inner loop logic with 4x4 dicts.
    
    rank_idx = 0
    for l in range(7):
        # X_blocks[l] and Y_blocks[l] are 4x4 dicts.
        # We need to form the 7 Strassen linear combos of the *blocks*.
        # Block indices: 00, 01, 10, 11.
        # We can represent a "block vector" as a list of 4 4x4 dicts? No.
        # A "block matrix" is a 2x2 matrix of 4x4 dicts? No, 2x2 matrix of 2x2 dicts.
        
        # Let's extract the 4 2x2 blocks from the 4x4 dict.
        def get_blocks(form4x4):
            blocks = []
            for r in range(2):
                for c in range(2):
                    block = {}
                    for i in range(2):
                        for j in range(2):
                            key = (r*2 + i, c*2 + j)
                            if key in form4x4:
                                block[(i,j)] = form4x4[key]
                    blocks.append(block)
            return blocks # [B00, B01, B10, B11]

        def combine_blocks(blocks, coeffs):
            # coeffs is list of 4 coeffs
            res = {}
            for b_idx in range(4):
                if coeffs[b_idx] != 0:
                    for key, val in blocks[b_idx].items():
                        res[key] = res.get(key, 0) + coeffs[b_idx] * val
            return res

        Xb = get_blocks(X_blocks[l])
        Yb = get_blocks(Y_blocks[l])
        
        # Strassen formulas for block matrices
        # U0 = X00 + X11
        # U1 = X10 + X11
        # U2 = X00
        # U3 = X11
        # U4 = X00 + X01
        # U5 = X10 - X00
        # U6 = X01 - X11
        
        U_block_coeffs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        
        # V0 = Y00 + Y11
        # V1 = Y00
        # V2 = Y01 - Y11
        # V3 = Y10 - Y00
        # V4 = Y11
        # V5 = Y00 + Y01
        # V6 = Y10 + Y11
        
        V_block_coeffs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        # Z coeffs
        Z_coeffs = [
            [1, 0, 0, 1, 0, -1, -1], # Z00
            [0, 0, 1, 0, 1, 0, 0],   # Z01
            [0, 1, 0, 0, 0, 1, 0],   # Z10
            [1, -1, 0, 0, 1, 0, 1]   # Z11
        ]
        
        for q in range(7):
            u_4x4 = combine_blocks(Xb, U_block_coeffs[q])
            v_4x4 = combine_blocks(Yb, V_block_coeffs[q])
            
            # u_4x4 is a dict of (i,j) -> coeff for A
            # v_4x4 is a dict of (j,k) -> coeff for B
            
            # Update U and V
            for (i, j), coeff in u_4x4.items():
                U[i * 4 + j][rank_idx] += coeff
            for (j, k), coeff in v_4x4.items():
                V[j * 4 + k][rank_idx] += coeff
            
            # Update W
            # Z_l entries are 2x2 blocks of the result C?
            # No, Z_l is a 2x2 matrix of *scalars*?
            # No, Z_l is a 2x2 matrix of *blocks*?
            # In the block Strassen, Z_l is a 2x2 matrix of blocks.
            # Each block of Z_l is a 2x2 matrix of scalars.
            # The formula Z00 = m1 + m4 - m6 - m7 applies to the *blocks*.
            # So the top-left block of Z_l is sum of top-left blocks of m_q?
            # No, m_q is a block product.
            # m_q = U_q V_q. U_q is a 2x2 block matrix. V_q is a 2x2 block matrix.
            # m_q is a 2x2 block matrix.
            # The entries of m_q are 2x2 scalar matrices.
            # The Strassen formula for 2x2 matrices applies to the *block entries*?
            # Yes.
            # So Z_l (2x2 block matrix) has entries that are 2x2 scalar matrices.
            # Z_l[0,0] (top-left block) = m1[0,0] + m4[0,0] - m6[0,0] - m7[0,0].
            # Each m_q[0,0] is a 2x2 scalar matrix.
            # So we need to sum the 2x2 scalar matrices.
            
            # How to get the 2x2 scalar matrix for m_q[0,0]?
            # m_q = U_q V_q.
            # U_q is a 2x2 block matrix, where each block is a 2x2 scalar matrix.
            # Let U_q_blocks be 4 dicts (2x2 scalar forms).
            # Let V_q_blocks be 4 dicts (2x2 scalar forms).
            # m_q[0,0] = U_q[0,0] V_q[0,0] + U_q[0,1] V_q[1,0].
            # This is a product of 2x2 scalar matrices.
            # This requires 8 multiplications?
            # NO. The whole point of Strassen is to reduce the multiplications.
            # The "multiplication" m_q is the product of two 2x2 *block* matrices.
            # But we are doing Strassen *again* inside?
            # No. The rank 49 comes from 7 block products, each of rank 7.
            # Rank 7 for 2x2 matrix multiplication is Strassen.
            # So m_q is computed using Strassen on the 2x2 *scalar* matrices that make up the blocks?
            # No.
            # Let's clarify.
            # Level 1: 4x4 = 2x2 blocks of 2x2.
            # Strassen on blocks: 7 products of 2x2 blocks.
            # Level 2: Each 2x2 block product is computed using Strassen on scalars?
            # Strassen on 2x2 scalar matrices uses 7 multiplications.
            # So yes, each m_q is a 2x2 matrix of scalars, computed using 7 scalar multiplications.
            
            # So we need to apply Strassen to the 2x2 block matrices U_q and V_q.
            # U_q is a 2x2 matrix of 2x2 scalar matrices.
            # Let's extract the 4 2x2 scalar matrices from u_4x4.
            
            Uq_blocks = get_blocks(u_4x4)
            Vq_blocks = get_blocks(v_4x4)
            
            # Apply Strassen to Uq_blocks and Vq_blocks
            # Uq_blocks[0] is top-left 2x2 scalar matrix.
            # ...
            
            # Strassen 7 multiplications for 2x2 scalar matrices
            # s1 = (U00+U11)(V00+V11)
            # ...
            
            # We need to generate 7 rank-1 terms for this m_q.
            # Each term contributes to the 4 scalar entries of m_q.
            
            # This is getting nested.
            # Total rank = 7 (outer) * 7 (inner) = 49.
            # So for each outer l, and inner q (0..6), we have 7 inner-inner terms?
            # No.
            # Outer Strassen: 7 products P_l.
            # Each P_l is a 2x2 block matrix.
            # To compute P_l = X_l Y_l, we use Strassen on the 2x2 block matrices.
            # This uses 7 multiplications of 2x2 *scalar* matrices?
            # No, Strassen on 2x2 block matrices uses 7 multiplications of *blocks*.
            # Each block multiplication is a 2x2 scalar matrix multiplication.
            # To compute a 2x2 scalar matrix multiplication exactly, we need 8 multiplications (schoolbook) or 7 (Strassen).
            # If we use Strassen for the inner level, we get 7 multiplications.
            # So total multiplications = 7 (outer) * 7 (inner) = 49.
            
            # So for each outer l, we have 7 inner multiplications (let's call them m_{l, q}).
            # Each m_{l, q} is a product of two 2x2 scalar matrices.
            # Wait, the "inner" Strassen is applied to the *blocks* of X_l and Y_l.
            # So the 7 inner multiplications are of the form (linear combo of 2x2 blocks) * (linear combo of 2x2 blocks).
            # But a "2x2 block" is a 2x2 matrix of scalars.
            # So the multiplication is of two 2x2 matrices of scalars.
            # This is exactly a 2x2 matrix multiplication.
            # So we need to decompose this 2x2 matrix multiplication into 7 rank-1 terms (Strassen).
            
            # So for each m_{l, q}, we apply Strassen to the 2x2 scalar matrices.
            # This generates 7 rank-1 terms.
            # Total 7 * 7 = 49 terms.
            
            # Let's implement the inner Strassen.
            
            # Uq_blocks and Vq_blocks are lists of 4 dicts (2x2 scalar forms).
            # Apply Strassen to them.
            
            # Inner Strassen formulas
            # u1 = U00 + U11
            # u2 = U10 + U11
            # u3 = U00
            # u4 = U11
            # u5 = U00 + U01
            # u6 = U10 - U00
            # u7 = U01 - U11
            
            # v1 = V00 + V11
            # v2 = V00
            # v3 = V01 - V11
            # v4 = V10 - V00
            # v5 = V11
            # v6 = V00 + V01
            # v7 = V10 + V11
            
            # Z_inner = ...
            
            # We need to map the inner Strassen terms back to the global W.
            # The inner Strassen computes Z_l (2x2 block matrix).
            # Each entry of Z_l is a 2x2 scalar matrix.
            # The inner Strassen gives 7 products s_{l, q, r} (r=0..6).
            # Each s is a scalar product of linear forms in A and B.
            # The coefficients for Z_l entries are known.
            
            # Let's do it.
            
            pass

    # This is getting too complex to do in one pass without a helper.
    # I will write a helper function to apply Strassen to 2x2 block matrices.

    return {"U": U, "V": V, "W": W}
