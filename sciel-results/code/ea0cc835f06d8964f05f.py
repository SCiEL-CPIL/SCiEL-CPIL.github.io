"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
# For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B:
# C00 = a*e + c*h
# C01 = b*h
# C10 = a*f + d*g
# C11 = b*g + d*h
#
# Strassen products:
# P1 = (a+d)(e+h)
# P2 = (c+d)e
# P3 = a(f-h)
# P4 = d(g-e)
# P5 = (a+b)h
# P6 = (c-a)(e+f)
# P7 = (b-d)(g+h)
#
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

# Each Strassen product is a linear combination of entries of A and B.
# We represent each product as:
#   (list of (ai, aj, coeff) for A-part, list of (bk, bl, coeff) for B-part)
# where ai,aj index into the 2x2 A block and bk,bl index into the 2x2 B block.

# Strassen products for a 2x2 block:
# A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
# B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h

_STRASSEN_2X2 = [
    # P1 = (a+d)(e+h)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # P2 = (c+d)e
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    # P3 = a(f-h)
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    # P4 = d(g-e)
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    # P5 = (a+b)h
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    # P6 = (c-a)(e+f)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    # P7 = (b-d)(g+h)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

# How each C entry is formed from the 7 Strassen products:
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

_STRASSEN_C_COMBINATION = [
    # (c_row, c_col, list of (product_idx, coeff))
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen: apply Strassen at the 4x4 level (7 block products),
    then apply Strassen within each 2x2 sub-block product (7 scalar mults each).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]  # shape (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 49)

    # At the 4x4 level, we partition A into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # Similarly for B and C.
    #
    # The 7 Strassen block products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two 2x2 matrices (linear combinations of blocks).
    # For each Mi, we then apply Strassen's 7 products to compute the 2x2 product.
    # Each of those 7 products corresponds to one rank-1 term in the full decomposition.
    #
    # The block-level Strassen structure:
    # For each block product Mi = Xi * Yi where Xi and Yi are 2x2 blocks:
    #   Xi is a linear combination of A-blocks
    #   Yi is a linear combination of B-blocks
    #
    # We need to express Xi as a linear combination of individual A entries,
    # and Yi as a linear combination of individual B entries.

    # Define the 7 block-level products. Each is (A_block_combo, B_block_combo).
    # A_block_combo: list of (block_row, block_col, coeff) for 2x2 blocks of A
    # B_block_combo: list of (block_row, block_col, coeff) for 2x2 blocks of B
    # Block (br, bc) of A covers rows 2*br..2*br+1, cols 2*bc..2*bc+1.

    block_products = [
        # M1 = (A00+A11)(B00+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2 = (A10+A11)B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        # M3 = A00(B01-B11)
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        # M4 = A11(B10-B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        # M5 = (A00+A01)B11
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]

    # C block combination (same as Strassen 2x2 C combination)
    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # For each block product Mi (i=0..6), we apply Strassen's 7 products.
    # The resulting 2x2 product gives entries for the C-block.
    #
    # For block product i with A-block-combo a_combo and B-block-combo b_combo:
    #   The "effective" 2x2 A matrix X has entries:
    #     X[r][c] = sum of coeff * A[2*br+r][2*bc+c] for (br,bc,coeff) in a_combo
    #   Similarly for B.
    #
    # Then Strassen's 7 products on X*Y give 7 rank-1 terms.
    # Each term contributes to C-block entries via the Strassen C combination.
    #
    # The C-block (cbr, cbc) gets contribution from block product i with
    # coefficient w_i, and within that, Strassen product s contributes to
    # C-local entry (lr, lc) with coefficient v_s.
    # So C[2*cbr+lr][2*cbc+lc] gets contribution w_i * v_s * (Strassen term).

    for i, (a_combo, b_combo) in enumerate(block_products):
        for s, (a_local, b_local) in enumerate(_STRASSEN_2X2):
            r = i * 7 + s  # global rank index (0..48)

            # Build U contributions: for each (a_br, a_bc, a_coeff) in a_combo
            # and each (lr, lc, l_coeff) in a_local:
            #   U[2*a_br + lr][2*a_bc + lc][r] += a_coeff * l_coeff
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            # Build V contributions: for each (b_br, b_bc, b_coeff) in b_combo
            # and each (lr, lc, l_coeff) in b_local:
            #   V[2*b_br + lr][2*b_bc + lc][r] += b_coeff * l_coeff
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

            # Build W contributions: for each C-block (cbr, cbc) that uses block
            # product i with weight w_i, and for each C-local entry (lr, lc)
            # that uses Strassen product s with weight v_s:
            #   W[2*cbc + lc][2*cbr + lr][r] += w_i * v_s
            # Note: W is indexed as W[k*n+i] where k is column of C, i is row of C.
            # C[row][col] -> W[col*n + row]
            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                # For each C-local entry that uses Strassen product s:
                for (cbr2, cbc2, c_list2) in c_block_combination:
                    pass  # This is handled below

            # Better approach: iterate over all C-block entries and all Strassen
            # C-combination entries to find contributions.
            # Actually, let's redo the W construction more carefully.

    # Let me redo the W construction properly.
    # Reset W
    W = [[0] * rank for _ in range(p * n)]

    for i, (a_combo, b_combo) in enumerate(block_products):
        for s, (a_local, b_local) in enumerate(_STRASSEN_2X2):
            r = i * 7 + s
            # For each C-block (cbr, cbc) with weight w_i for block product i:
            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                # For each C-local entry (lr, lc) with weight v_s for Strassen product s:
                for (cbr2, cbc2, c_list2) in c_block_combination:
                    pass

    # I need to be more careful. Let me restructure.
    # The C-block (cbr, cbc) is computed as:
    #   C[cbr,cbc] = sum_i w_i * Mi  where Mi = Xi * Yi (2x2 matrices)
    # And Mi is computed via Strassen:
    #   Mi[lr,lc] = sum_s v_s * Ps  where Ps is the s-th Strassen product
    # So C[2*cbr+lr, 2*cbc+lc] = sum_i sum_s w_i(cbr,cbc,i) * v_s(lr,lc,s) * Ps(i,s)
    # And Ps(i,s) is the s-th Strassen product of the i-th block product.
    # Each Ps(i,s) is a rank-1 term with U, V, W coefficients.
    # The W coefficient for C[2*cbr+lr, 2*cbc+lc] from term (i,s) is w_i * v_s.

    # Let's build W properly:
    W = [[0] * rank for _ in range(p * n)]

    for i in range(7):
        for s in range(7):
            r = i * 7 + s
            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                # Now find v_s for each C-local entry
                for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
                    v_s = 0
                    for (si, vs) in v_list:
                        if si == s:
                            v_s = vs
                            break
                    if v_s == 0:
                        continue
                    c_row = 2 * cbr + lr
                    c_col = 2 * cbc + lc
                    # W is indexed as W[col*n + row]
                    W[c_col * n + c_row][r] += w_i * v_s

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using NumPy."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(result, 1.0)
    except Exception:
        return False


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
