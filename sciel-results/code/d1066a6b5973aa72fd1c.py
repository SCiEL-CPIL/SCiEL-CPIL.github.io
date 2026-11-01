"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Block Strassen decomposition: rank 36 for 2x4 * 4x5
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition adds 7 terms
    # A block at rows [ar, ar+1], cols [ac, ac+1]
    # B block at rows [br, br+1], cols [bc, bc+1]
    # C block at rows [cr, cr+1], cols [cc, cc+1]
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i (trace convention)
    
    def add_strassen(ar, ac, br, bc, cr, cc):
        # Strassen for 2x2 * 2x2 -> 2x2
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        col = 0
        for coeff_A, coeff_B, coeff_C in [
            # (i_offset, j_offset, k_offset, c_i, c_j, sign)
            # M1: (A00+A11)(B00+B11)
            ((0,0), (0,0), (0,0), 1),
            ((1,1), (0,0), (0,0), 1),
            ((0,0), (1,1), (0,0), 1),
            ((1,1), (1,1), (0,0), 1),
            # M2: (A10+A11)B00
            ((1,0), (0,0), (1,0), 1),
            ((1,1), (0,0), (1,0), 1),
            # M3: A00(B01-B11)
            ((0,0), (0,1), (0,1), 1),
            ((0,0), (1,1), (0,1), -1),
            # M4: A11(B10-B00)
            ((1,1), (1,0), (1,0), 1),
            ((1,1), (0,0), (1,0), -1),
            # M5: (A00+A01)B11
            ((0,0), (1,1), (0,1), 1),
            ((0,1), (1,1), (0,1), 1),
            # M6: (A10-A00)(B00+B01)
            ((1,0), (0,0), (1,0), 1),
            ((0,0), (0,0), (1,0), -1),
            ((1,0), (0,1), (1,0), 1),
            ((0,0), (0,1), (1,0), -1),
            # M7: (A01-A11)(B10+B11)
            ((0,1), (1,0), (0,1), 1),
            ((1,1), (1,0), (0,1), -1),
            ((0,1), (1,1), (0,1), 1),
            ((1,1), (1,1), (0,1), -1),
        ]:
            pass  # This approach is getting complex; let me use a direct construction
        
        return col
    
    # Direct construction: 36 terms
    # Split B into 5 columns: B1 (cols 0,1), B2 (cols 2,3), B3 (col 4)
    # Split A into 2 rows: A1 (row 0), A2 (row 1)
    # C = [A1*B1 | A1*B2 | A1*B3 | A2*B1 | A2*B2 | A2*B3]
    # Wait, C is 2x5, so C[i,k] = sum_j A[i,j] B[j,k]
    # A is 2x4, B is 4x5
    # Let me think of it as:
    # Split the 4 intermediate dimension into two halves: j=0,1 and j=2,3
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    # B1 = B[0:2, :] (2x5), B2 = B[2:4, :] (2x5)
    # C = A1*B1 + A2*B2
    # Each A1*B1 is 2x2 * 2x5. Split B1 into B1_1 (2x2, cols 0,1), B1_2 (2x2, cols 2,3), B1_3 (2x1, col 4)
    # A1*B1 = A1*B1_1 + A1*B1_2 + A1*B1_3
    # A1*B1_1: 2x2 * 2x2 -> use Strassen (7 terms)
    # A1*B1_2: 2x2 * 2x2 -> use Strassen (7 terms)
    # A1*B1_3: 2x2 * 2x1 -> schoolbook (4 terms)
    # Similarly for A2*B2:
    # A2*B2_1: 2x2 * 2x2 -> Strassen (7 terms)
    # A2*B2_2: 2x2 * 2x2 -> Strassen (7 terms)
    # A2*B2_3: 2x2 * 2x1 -> schoolbook (4 terms)
    # Total: 7+7+4+7+7+4 = 36 terms
    
    col = 0
    
    def add_2x2x2_strassen(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_i0, c_k0):
        nonlocal col
        # a_i0, a_i1: row indices in A (0,1)
        # a_j0, a_j1: col indices in A (j offset in the 4-dim)
        # b_j0, b_j1: row indices in B (j offset in the 4-dim)
        # b_k0, b_k1: col indices in B (k offset in the 5-dim)
        # c_i0, c_k0: top-left of C block (2x2)
        # U[i*m+j], V[j*p+k], W[k*n+i]
        
        # Strassen 2x2*2x2:
        # M1 = (A00+A11)(B00+B11)
        for (ai, aj, bj, bk, ci, ck, sign) in [
            (a_i0, a_j0, b_j0, b_k0, c_i0, c_k0, 1),
            (a_i1, a_j1, b_j0, b_k0, c_i0, c_k0, 1),
            (a_i0, a_j0, b_j1, b_k1, c_i0, c_k0, 1),
            (a_i1, a_j1, b_j1, b_k1, c_i0, c_k0, 1),
        ]:
            U[ai*m+aj][col] += sign
            V[bj*p+bk][col] += sign
            W[ck*n+ci][col] += sign
        # M2 = (A10+A11)B00
        for (ai, aj, bj, bk, ci, ck, sign) in [
            (a_i1, a_j0, b_j0, b_k0, c_i1, c_k0, 1),
            (a_i1, a_j1, b_j0, b_k0, c_i1, c_k0, 1),
        ]:
            U[ai*m+aj][col] += sign
            V[bj*p+bk][col] += sign
            W[ck*n+ci][col] += sign
        # M3 = A00(B01-B11)
        for (ai, aj, bj, bk, ci, ck, sign) in [
            (a_i0, a_j0, b_j0, b_k1, c_i0, c_k1, 1),
            (a_i0, a_j0, b_j1, b_k1, c_i0, c_k1, -1),
        ]:
            U[ai*m+aj][col] += sign
            V[bj*p+bk][col] += sign
            W[ck*n+ci][col] += sign
        # M4 = A11(B10-B00)
        for (ai, aj, bj, bk, ci, ck, sign) in [
            (a_i1, a_j1, b_j1, b_k0, c_i1, c_k0, 1),
            (a_i1, a_j1, b_j0, b_k0, c_i1, c_k0, -1),
        ]:
            U[ai*m+aj][col] += sign
            V[bj*p+bk][col] += sign
            W[ck*n+ci][col] += sign
        # M5 = (A00+A01)B11
        for (ai, aj, bj, bk, ci, ck, sign) in [
            (a_i0, a_j0, b_j1, b_k1, c_i0, c_k1, 1),
            (a_i0, a_j1, b_j1, b_k1, c_i0, c_k1, 1),
        ]:
            U[ai*m+aj][col] += sign
            V[bj*p+bk][col] += sign
            W[ck*n+ci][col] += sign
        # M6 = (A10-A00)(B00+B01)
        for (ai, aj, bj, bk, ci, ck, sign) in [
            (a_i1, a_j0, b_j0, b_k0, c_i1, c_k0, 1),
            (a_i0, a_j0, b_j0, b_k0, c_i1, c_k0, -1),
            (a_i1, a_j0, b_j0, b_k1, c_i1, c_k0, 1),
            (a_i0, a_j0, b_j0, b_k1, c_i1, c_k0, -1),
        ]:
            U[ai*m+aj][col] += sign
            V[bj*p+bk][col] += sign
            W[ck*n+ci][col] += sign
        # M7 = (A01-A11)(B10+B11)
        for (ai, aj, bj, bk, ci, ck, sign) in [
            (a_i0, a_j1, b_j1, b_k0, c_i0, c_k1, 1),
            (a_i1, a_j1, b_j1, b_k0, c_i0, c_k1, -1),
            (a_i0, a_j1, b_j1, b_k1, c_i0, c_k1, 1),
            (a_i1, a_j1, b_j1, b_k1, c_i0, c_k1, -1),
        ]:
            U[ai*m+aj][col] += sign
            V[bj*p+bk][col] += sign
            W[ck*n+ci][col] += sign
        col += 1
    
    def add_2x2x1_schoolbook(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k, c_i0, c_k):
        nonlocal col
        for (ai, aj, bj, ci) in [
            (a_i0, a_j0, b_j0, c_i0),
            (a_i0, a_j1, b_j1, c_i0),
            (a_i1, a_j0, b_j0, c_i0+1),
            (a_i1, a_j1, b_j1, c_i0+1),
        ]:
            U[ai*m+aj][col] = 1
            V[bj*p+b_k][col] = 1
            W[c_k*n+ci][col] = 1
            col += 1
    
    # A1 = A[:, 0:2], B1 = B[0:2, :]
    # B1_1 = B[0:2, 0:2], B1_2 = B[0:2, 2:4], B1_3 = B[0:2, 4]
    add_2x2x2_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 0)  # A1*B1_1 -> C[:, 0:2]
    add_2x2x2_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 2)  # A1*B1_2 -> C[:, 2:4]
    add_2x2x1_schoolbook(0, 1, 0, 1, 0, 1, 4, 0, 4)   # A1*B1_3 -> C[:, 4]
    
    # A2 = A[:, 2:4], B2 = B[2:4, :]
    # B2_1 = B[2:4, 0:2], B2_2 = B[2:4, 2:4], B2_3 = B[2:4, 4]
    add_2x2x2_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 0)  # A2*B2_1 -> C[:, 0:2]
    add_2x2x2_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 2)  # A2*B2_2 -> C[:, 2:4]
    add_2x2x1_schoolbook(0, 1, 2, 3, 2, 3, 4, 0, 4)   # A2*B2_3 -> C[:, 4]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
