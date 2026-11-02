import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Hybrid decomposition: Strassen (rank 7) for 2x2 block products, combined
    with schoolbook for the 3 extra columns.
    
    Strategy:
    - Split B into B_left (4x4) and B_right (4x3)
    - C_left = A * B_left: use Strassen with 4x4 blocks (4 blocks each 2x2)
    - C_right = A * B_right: schoolbook
    
    For Strassen on 4x4 = 2x2 blocks of 2x2 matrices:
    Each Strassen multiplier M_s involves a 2x2 matrix product.
    We decompose each 2x2 product using Strassen's 7-term algorithm.
    Total: 7 (outer Strassen) * 7 (inner Strassen) = 49 terms for C_left
    Plus 4*4*3 = 48 terms for C_right
    Total rank: 97
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    U_terms = []  # list of (row, val) for each term
    V_terms = []  # list of (row, val) for each term
    W_terms = []  # list of (row, val) for each term
    
    # Strassen decomposition for 2x2 matrix multiplication
    # For matrices X (2x2) and Y (2x2):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Result:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_s is itself a product of two 2x2 "combinations", but actually
    # M_s = X_comb * Y_comb where X_comb and Y_comb are specific linear combos.
    # But for tensor decomposition, each M_s is a single scalar product of
    # two vectors... no wait.
    
    # Actually, Strassen gives 7 multiplications. Each multiplication is
    # (linear combo of A entries) * (linear combo of B entries).
    # So each term is rank-1 in the tensor sense.
    
    # For 4x4 * 4x4 using Strassen on 2x2 blocks:
    # A is divided into 4 blocks: A00, A01, A10, A11 (each 2x2)
    # B is divided into 4 blocks: B00, B01, B10, B11 (each 2x2)
    
    # Outer Strassen multipliers (each is a 2x2 * 2x2 product):
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # Each M_s = X_s * Y_s where X_s is a 2x2 matrix (linear combo of A blocks)
    # and Y_s is a 2x2 matrix (linear combo of B blocks).
    # Then X_s * Y_s is a 2x2 matrix, which we decompose using inner Strassen.
    
    # Let's define the outer Strassen structure:
    # For each outer multiplier s, we have:
    # X_s = sum of coefficients * A_blocks
    # Y_s = sum of coefficients * B_blocks
    
    # A block Aij corresponds to rows 2i..2i+1, cols 2j..2j+1 of A
    # B block Bij corresponds to rows 2i..2i+1, cols 2j..2j+1 of B
    
    # Outer Strassen multipliers:
    # (X_coeffs, Y_coeffs) where X_coeffs maps block->coefficient, same for Y
    outer_strassen = [
        # (X_block_coeffs, Y_block_coeffs)
        # X and Y are 2x2 grids of coefficients for A-blocks and B-blocks respectively
        # A-blocks: 00, 01, 10, 11; B-blocks: 00, 01, 10, 11
        # For M1: X = A00+A11, Y = B00+B11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: X = A10+A11, Y = B00
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # M3: X = A00, Y = B01-B11
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        # M4: X = A11, Y = B10-B00
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        # M5: X = A00+A01, Y = B11
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        # M6: X = A10-A00, Y = B00+B01
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # M7: X = A01-A11, Y = B10+B11
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # Inner Strassen for 2x2 * 2x2:
    # For matrices X (2x2) and Y (2x2), the 7 multipliers are:
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    # C00 = m1+m4-m5+m7, C01 = m3+m5, C10 = m2+m4, C11 = m1-m2+m3+m6
    
    # Each inner multiplier is a product of (linear combo of X entries) * (linear combo of Y entries)
    # X entries: x00, x01, x10, x11 (positions in 2x2 block)
    # Y entries: y00, y01, y10, y11
    
    inner_strassen = [
        # (x_coeffs, y_coeffs) for each of 7 multipliers
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # m1
        ({(1,0):1, (1,1):1}, {(0,0):1}),            # m2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),           # m3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),           # m4
        ({(0,0):1, (0,1):1}, {(1,1):1}),            # m5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),  # m6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),  # m7
    ]
    
    # C_left accumulation: C00, C01, C10, C11 each get contributions from outer multipliers
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each output block (ci, cj), which outer multipliers contribute and with what sign:
    # ci, cj in {0,1}
    block_contributions = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],   # M1, M4, -M5, M7
        (0,1): [(2,1), (4,1)],                   # M3, M5
        (1,0): [(1,1), (3,1)],                   # M2, M4
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],   # M1, -M2, M3, M6
    }
    
    # For each contribution, we need to add the inner Strassen decomposition
    # The inner Strassen produces a 2x2 result, which maps to specific rows/cols in the 4x4 output
    
    # Let's build all terms for C_left:
    # For each outer multiplier s, for each inner multiplier t, we get a term.
    # But we only include terms that contribute to the final C via the block_contributions.
    
    # Actually, let me think differently. Each rank-1 term in the tensor is:
    # u * v * w where u is a vector in R^(n*m), v in R^(m*p), w in R^(p*n)
    # For a term contributing to C[ci*2+ri, cj*2+cs] (where ri,cs in {0,1} for inner block):
    # The term involves specific A entries and B entries.
    
    # Let me enumerate all terms by iterating:
    # For each (ci,cj) output block:
    #   For each (s, sign) in block_contributions[(ci,cj)]:
    #     outer_X = outer_strassen[s][0]  # dict mapping A-block (ai,aj) -> coeff
    #     outer_Y = outer_strassen[s][1]  # dict mapping B-block (bi,bj) -> coeff
    #     For each (t) in range(7):  # inner Strassen multipliers
    #       inner_x = inner_strassen[t][0]  # dict mapping (ri,ci_inner) -> coeff
    #       inner_y = inner_strassen[t][1]  # dict mapping (ri,cj_inner) -> coeff
    #       Now, the term's U vector has non-zero entries at:
    #         For each (ai,aj) in outer_X with coeff cx:
    #           For each (ri_inner, ci_inner) in inner_x with coeff ix:
    #             A row = ci*2 + ri_inner  (wait, no)
    
    # Hmm, let me reconsider the indexing.
    # A is n x m = 4x4. A[i,j] where i,j in 0..3.
    # A block Aij has rows 2i, 2i+1 and cols 2j, 2j+1.
    # So A[2*ai+ri, 2*aj+cj_inner] where ai,aj in {0,1} (block indices), ri,cj_inner in {0,1} (within block)
    
    # Wait, I'm confusing myself. Let me use:
    # A block (a_i, a_j): rows 2*a_i to 2*a_i+1, cols 2*a_j to 2*a_j+1
    # Within block, local indices (r, c) where r,c in {0,1}
    # Global A index: row = 2*a_i + r, col = 2*a_j + c
    
    # B block (b_i, b_j): rows 2*b_i to 2*b_i+1, cols 2*b_j to 2*b_j+1
    # Global B index: row = 2*b_i + r, col = 2*b_j + c
    
    # Output C block (c_i, c_j): rows 2*c_i to 2*c_i+1, cols 2*c_j to 2*c_j+1
    # Global C index: row = 2*c_i + r, col = 2*c_j + c
    
    # For inner Strassen on a 2x2 product X*Y:
    # X has entries X[r,c] for r,c in {0,1}
    # Y has entries Y[r,c] for r,c in {0,1}
    # Result Z[r,c] for r,c in {0,1}
    
    # The inner Strassen multipliers m_t = (sum of X entries with coeffs) * (sum of Y entries with coeffs)
    # Each m_t contributes to Z with coefficients.
    
    # For a given outer multiplier s with:
    # X_s = sum over (ai,aj) of outer_X[(ai,aj)] * A_block(ai,aj)
    # Y_s = sum over (bi,bj) of outer_Y[(bi,bj)] * B_block(bi,bj)
    
    # X_s[r,c] = sum over (ai,aj) of outer_X[(ai,aj)] * A[2*ai+r, 2*aj+c]
    # Y_s[r,c] = sum over (bi,bj) of outer_Y[(bi,bj)] * B[2*bi+r, 2*bj+c]
    
    # Inner multiplier m_t = (sum_{r,c} inner_x[(r,c)] * X_s[r,c]) * (sum_{r,c} inner_y[(r,c)] * Y_s[r,c])
    
    # = (sum_{r,c} inner_x[(r,c)] * sum_{ai,aj} outer_X[(ai,aj)] * A[2*ai+r, 2*aj+c])
    #   * (sum_{r,c} inner_y[(r,c)] * sum_{bi,bj} outer_Y[(bi,bj)] * B[2*bi+r, 2*bj+c])
    
    # = (sum over all A entries of coeff_A * A_entry) * (sum over all B entries of coeff_B * B_entry)
    
    # So for the U vector (index = i*m+j = i*4+j for A[i,j]):
    # U[i*4+j] = coeff_A for that entry
    # For the V vector (index = j*p+k = j*7+k for B[j,k]):
    # V[j*7+k] = coeff_B for that entry
    # For the W vector (index = k*n+i = k*4+i for C[i,k]):
    # W[k*4+i] = contribution coefficient
    
    # The contribution to C[2*ci+r, 2*cj+c] from inner multiplier t is:
    # inner_result[(r,c)] for the Z matrix, where Z is the 2x2 result of X_s*Y_s
    # And the block contribution sign from outer Strassen.
    
    # Inner Strassen result:
    # Z00 = m1+m4-m5+m7
    # Z01 = m3+m5
    # Z10 = m2+m4
    # Z11 = m1-m2+m3+m6
    
    inner_result_coeffs = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],
    }
    
    # Now let's build all terms:
    for ci in range(2):
        for cj in range(2):
            for (s, outer_sign) in block_contributions[(ci,cj)]:
                outer_X = outer_strassen[s][0]
                outer_Y = outer_strassen[s][1]
                for t in range(7):
                    inner_x = inner_strassen[t][0]
                    inner_y = inner_strassen[t][1]
                    # Build U, V, W coefficient dicts
                    U_coeff = {}  # (i,j) -> coeff
                    V_coeff = {}  # (j,k) -> coeff
                    W_coeff = {}  # (k,i) -> coeff for C output
                    
                    # Compute U coefficients
                    for (r,c) in inner_x:
                        ix = inner_x[(r,c)]
                        for (ai,aj) in outer_X:
                            ox = outer_X[(ai,aj)]
                            # A global index: row = 2*ai + r, col = 2*aj + c
                            a_row = 2*ai + r
                            a_col = 2*aj + c
                            key = (a_row, a_col)
                            U_coeff[key] = U_coeff.get(key, 0) + ix * ox
                    
                    # Compute V coefficients
                    for (r,c) in inner_y:
                        iy = inner_y[(r,c)]
                        for (bi,bj) in outer_Y:
                            oy = outer_Y[(bi,bj)]
                            # B global index: row = 2*bi + r, col = 2*bj + c
                            b_row = 2*bi + r
                            b_col = 2*bj + c
                            key = (b_row, b_col)
                            V_coeff[key] = V_coeff.get(key, 0) + iy * oy
                    
                    # Compute W coefficients: this term contributes to C[2*ci+rr, 2*cj+cc]
                    # for (rr,cc) where inner_result_coeffs[(rr,cc)] includes (t, coeff)
                    for (rr,cc) in inner_result_coeffs:
                        for (tt, inner_c) in inner_result_coeffs[(rr,cc)]:
                            if tt == t:
                                c_row = 2*ci + rr
                                c_col = 2*cj + cc
                                key = (c_col, c_row)  # W[k*n+i] so index = k*4+i
                                W_coeff[key] = W_coeff.get(key, 0) + outer_sign * inner_c
                    
                    # Add this term
                    U_term = [0]*16
                    for (i,j), val in U_coeff.items():
                        if val != 0:
                            U_term[i*4+j] = val
                    V_term = [0]*28
                    for (j,k), val in V_coeff.items():
                        if val != 0:
                            V_term[j*7+k] = val
                    W_term = [0]*28
                    for (k,i), val in W_coeff.items():
                        if val != 0:
                            W_term[k*4+i] = val
                    
                    if any(x != 0 for x in U_term) and any(x != 0 for x in V_term) and any(x != 0 for x in W_term):
                        U_terms.append(U_term)
                        V_terms.append(V_term)
                        W_terms.append(W_term)
    
    # C_right: schoolbook for columns 4,5,6 of B
    # C[i, 4+k] = sum_j A[i,j] * B[j, 4+k] for k=0,1,2
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_term = [0]*16
                U_term[i*4+j] = 1
                V_term = [0]*28
                V_term[j*7+(4+k)] = 1
                W_term = [0]*28
                W_term[(4+k)*4+i] = 1
                U_terms.append(U_term)
                V_terms.append(V_term)
                W_terms.append(W_term)
    
    rank = len(U_terms)
    U = U_terms
    V = V_terms
    W = W_terms
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
