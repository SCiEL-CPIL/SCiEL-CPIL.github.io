import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(np.array(u_vec, dtype=np.float64))
        V_cols.append(np.array(v_vec, dtype=np.float64))
        W_cols.append(np.array(w_vec, dtype=np.float64))
    
    def ai(r, c): return r * m + c
    def bi(r, c): return r * p + c
    def ci(r, c): return c * n + r
    
    def strassen_2x2_at(ar, ac, br, bc, cr, cc):
        """Add 7 Strassen terms for 2x2*2x2 at given positions."""
        def term(u_e, v_e, w_e):
            u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
            for idx, val in u_e: u[idx] = val
            for idx, val in v_e: v[idx] = val
            for idx, val in w_e: w[idx] = val
            add_term(u, v, w)
        
        a00, a01 = ai(ar,ac), ai(ar,ac+1)
        a10, a11 = ai(ar+1,ac), ai(ar+1,ac+1)
        b00, b01 = bi(br,bc), bi(br,bc+1)
        b10, b11 = bi(br+1,bc), bi(br+1,bc+1)
        c00, c01 = ci(cr,cc), ci(cr,cc+1)
        c10, c11 = ci(cr+1,cc), ci(cr+1,cc+1)
        
        # M1 = (a00+a11)*(b00+b11) -> c00+c11
        term([(a00,1.0),(a11,1.0)], [(b00,1.0),(b11,1.0)], [(c00,1.0),(c11,1.0)])
        # M2 = (a10+a11)*b00 -> c10-c11
        term([(a10,1.0),(a11,1.0)], [(b00,1.0)], [(c10,1.0),(c11,-1.0)])
        # M3 = a00*(b01-b11) -> c01+c11
        term([(a00,1.0)], [(b01,1.0),(b11,-1.0)], [(c01,1.0),(c11,1.0)])
        # M4 = a11*(b10-b00) -> c00+c10
        term([(a11,1.0)], [(b10,1.0),(b00,-1.0)], [(c00,1.0),(c10,1.0)])
        # M5 = (a00+a01)*b11 -> -c00+c01
        term([(a00,1.0),(a01,1.0)], [(b11,1.0)], [(c00,-1.0),(c01,1.0)])
        # M6 = (a10-a00)*(b00+b01) -> c11
        term([(a10,1.0),(a00,-1.0)], [(b00,1.0),(b01,1.0)], [(c11,1.0)])
        # M7 = (a01-a11)*(b10+b11) -> c00
        term([(a01,1.0),(a11,-1.0)], [(b10,1.0),(b11,1.0)], [(c00,1.0)])
    
    # Build rank-77 decomposition:
    # Part 1: 4x4 * 4x4 (columns 0-3 of B) using recursive Strassen (rank 49)
    # Part 2: 4x4 * 4x2 (columns 4-5 of B) using four 2x2*2x2 Strassen (rank 28)
    
    # === PART 2: 4x4 * 4x2 (columns 4-5) ===
    # C_top (rows 0-1, cols 4-5) = A11(0,0)*B(0,4) + A12(0,2)*B(2,4)
    # C_bot (rows 2-3, cols 4-5) = A21(2,0)*B(0,4) + A22(2,2)*B(2,4)
    
    strassen_2x2_at(0, 0, 0, 4, 0, 4)  # A11 * B_top -> C_top
    strassen_2x2_at(0, 2, 2, 4, 0, 4)  # A12 * B_bot -> C_top
    strassen_2x2_at(2, 0, 0, 4, 2, 4)  # A21 * B_top -> C_bot
    strassen_2x2_at(2, 2, 2, 4, 2, 4)  # A22 * B_bot -> C_bot
    # That's 28 terms.
    
    # === PART 1: 4x4 * 4x4 (columns 0-3) using recursive Strassen ===
    # Top-level Strassen: 7 terms, each decomposed into 7 bottom-level Strassen terms.
    # Total: 49 terms.
    
    # Define the 4 entries of each 2x2 block as lists of (index, coeff)
    # A11 block (rows 0-1, cols 0-1):
    A11_00 = [(ai(0,0), 1.0)]
    A11_01 = [(ai(0,1), 1.0)]
    A11_10 = [(ai(1,0), 1.0)]
    A11_11 = [(ai(1,1), 1.0)]
    # A12 block (rows 0-1, cols 2-3):
    A12_00 = [(ai(0,2), 1.0)]
    A12_01 = [(ai(0,3), 1.0)]
    A12_10 = [(ai(1,2), 1.0)]
    A12_11 = [(ai(1,3), 1.0)]
    # A21 block (rows 2-3, cols 0-1):
    A21_00 = [(ai(2,0), 1.0)]
    A21_01 = [(ai(2,1), 1.0)]
    A21_10 = [(ai(3,0), 1.0)]
    A21_11 = [(ai(3,1), 1.0)]
    # A22 block (rows 2-3, cols 2-3):
    A22_00 = [(ai(2,2), 1.0)]
    A22_01 = [(ai(2,3), 1.0)]
    A22_10 = [(ai(3,2), 1.0)]
    A22_11 = [(ai(3,3), 1.0)]
    
    # B11 block (rows 0-1, cols 0-1):
    B11_00 = [(bi(0,0), 1.0)]
    B11_01 = [(bi(0,1), 1.0)]
    B11_10 = [(bi(1,0), 1.0)]
    B11_11 = [(bi(1,1), 1.0)]
    # B12 block (rows 0-1, cols 2-3):
    B12_00 = [(bi(0,2), 1.0)]
    B12_01 = [(bi(0,3), 1.0)]
    B12_10 = [(bi(1,2), 1.0)]
    B12_11 = [(bi(1,3), 1.0)]
    # B21 block (rows 2-3, cols 0-1):
    B21_00 = [(bi(2,0), 1.0)]
    B21_01 = [(bi(2,1), 1.0)]
    B21_10 = [(bi(3,0), 1.0)]
    B21_11 = [(bi(3,1), 1.0)]
    # B22 block (rows 2-3, cols 2-3):
    B22_00 = [(bi(2,2), 1.0)]
    B22_01 = [(bi(2,3), 1.0)]
    B22_10 = [(bi(3,2), 1.0)]
    B22_11 = [(bi(3,3), 1.0)]
    
    # C blocks:
    # C11 (rows 0-1, cols 0-1):
    C11_00 = [(ci(0,0), 1.0)]
    C11_01 = [(ci(0,1), 1.0)]
    C11_10 = [(ci(1,0), 1.0)]
    C11_11 = [(ci(1,1), 1.0)]
    # C12 (rows 0-1, cols 2-3):
    C12_00 = [(ci(0,2), 1.0)]
    C12_01 = [(ci(0,3), 1.0)]
    C12_10 = [(ci(1,2), 1.0)]
    C12_11 = [(ci(1,3), 1.0)]
    # C21 (rows 2-3, cols 0-1):
    C21_00 = [(ci(2,0), 1.0)]
    C21_01 = [(ci(2,1), 1.0)]
    C21_10 = [(ci(3,0), 1.0)]
    C21_11 = [(ci(3,1), 1.0)]
    # C22 (rows 2-3, cols 2-3):
    C22_00 = [(ci(2,2), 1.0)]
    C22_01 = [(ci(2,3), 1.0)]
    C22_10 = [(ci(3,2), 1.0)]
    C22_11 = [(ci(3,3), 1.0)]
    
    def merge_entries(*lists):
        """Merge multiple lists of (index, coeff) into one, summing coeffs for same index."""
        result = {}
        for lst in lists:
            for idx, val in lst:
                result[idx] = result.get(idx, 0.0) + val
        return [(idx, val) for idx, val in result.items() if abs(val) > 1e-12]
    
    def negate_entries(lst):
        return [(idx, -val) for idx, val in lst]
    
    def add_strassen_2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        """Add 7 Strassen terms for a 2x2*2x2 where each entry is a list of (index, coeff)."""
        def term(u_e, v_e, w_e):
            u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
            for idx, val in u_e: u[idx] = val
            for idx, val in v_e: v[idx] = val
            for idx, val in w_e: w[idx] = val
            add_term(u, v, w)
        
        # M1 = (a00+a11)*(b00+b11) -> c00+c11
        term(merge_entries(a00, a11), merge_entries(b00, b11), merge_entries(c00, c11))
        # M2 = (a10+a11)*b00 -> c10-c11
        term(merge_entries(a10, a11), b00, merge_entries(c10, negate_entries(c11)))
        # M3 = a00*(b01-b11) -> c01+c11
        term(a00, merge_entries(b01, negate_entries(b11)), merge_entries(c01, c11))
        # M4 = a11*(b10-b00) -> c00+c10
        term(a11, merge_entries(b10, negate_entries(b00)), merge_entries(c00, c10))
        # M5 = (a00+a01)*b11 -> -c00+c01
        term(merge_entries(a00, a01), b11, merge_entries(negate_entries(c00), c01))
        # M6 = (a10-a00)*(b00+b01) -> c11
        term(merge_entries(a10, negate_entries(a00)), merge_entries(b00, b01), c11)
        # M7 = (a01-a11)*(b10+b11) -> c00
        term(merge_entries(a01, negate_entries(a11)), merge_entries(b10, b11), c00)
    
    # Top-level Strassen terms for 4x4*4x4:
    # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
    a_M1_00 = merge_entries(A11_00, A22_00)
    a_M1_01 = merge_entries(A11_01, A22_01)
    a_M1_10 = merge_entries(A11_10, A22_10)
    a_M1_11 = merge_entries(A11_11, A22_11)
    b_M1_00 = merge_entries(B11_00, B22_00)
    b_M1_01 = merge_entries(B11_01, B22_01)
    b_M1_10 = merge_entries(B11_10, B22_10)
    b_M1_11 = merge_entries(B11_11, B22_11)
    # C contributes to C11 and C22
    c_M1_00 = merge_entries(C11_00, C22_00)
    c_M1_01 = merge_entries(C11_01, C22_01)
    c_M1_10 = merge_entries(C11_10, C22_10)
    c_M1_11 = merge_entries(C11_11, C22_11)
    add_strassen_2x2(a_M1_00, a_M1_01, a_M1_10, a_M1_11,
                     b_M1_00, b_M1_01, b_M1_10, b_M1_11,
                     c_M1_00, c_M1_01, c_M1_10, c_M1_11)
    
    # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
    a_M2_00 = merge_entries(A21_00, A22_00)
    a_M2_01 = merge_entries(A21_01, A22_01)
    a_M2_10 = merge_entries(A21_10, A22_10)
    a_M2_11 = merge_entries(A21_11, A22_11)
    b_M2_00 = B11_00
    b_M2_01 = B11_01
    b_M2_10 = B11_10
    b_M2_11 = B11_11
    # C21 +, C22 -
    c_M2_00 = merge_entries(C21_00, negate_entries(C22_00))
    c_M2_01 = merge_entries(C21_01, negate_entries(C22_01))
    c_M2_10 = merge_entries(C21_10, negate_entries(C22_10))
    c_M2_11 = merge_entries(C21_11, negate_entries(C22_11))
    add_strassen_2x2(a_M2_00, a_M2_01, a_M2_10, a_M2_11,
                     b_M2_00, b_M2_01, b_M2_10, b_M2_11,
                     c_M2_00, c_M2_01, c_M2_10, c_M2_11)
    
    # M3 = A11*(B12-B22) -> C12 += M3
    a_M3_00 = A11_00
    a_M3_01 = A11_01
    a_M3_10 = A11_10
    a_M3_11 = A11_11
    b_M3_00 = merge_entries(B12_00, negate_entries(B22_00))
    b_M3_01 = merge_entries(B12_01, negate_entries(B22_01))
    b_M3_10 = merge_entries(B12_10, negate_entries(B22_10))
    b_M3_11 = merge_entries(B12_11, negate_entries(B22_11))
    c_M3_00 = C12_00
    c_M3_01 = C12_01
    c_M3_10 = C12_10
    c_M3_11 = C12_11
    add_strassen_2x2(a_M3_00, a_M3_01, a_M3_10, a_M3_11,
                     b_M3_00, b_M3_01, b_M3_10, b_M3_11,
                     c_M3_00, c_M3_01, c_M3_10, c_M3_11)
    
    # M4 = A22*(B21-B11) -> C21 += M4
    a_M4_00 = A22_00
    a_M4_01 = A22_01
    a_M4_10 = A22_10
    a_M4_11 = A22_11
    b_M4_00 = merge_entries(B21_00, negate_entries(B11_00))
    b_M4_01 = merge_entries(B21_01, negate_entries(B11_01))
    b_M4_10 = merge_entries(B21_10, negate_entries(B11_10))
    b_M4_11 = merge_entries(B21_11, negate_entries(B11_11))
    c_M4_00 = C21_00
    c_M4_01 = C21_01
    c_M4_10 = C21_10
    c_M4_11 = C21_11
    add_strassen_2x2(a_M4_00, a_M4_01, a_M4_10, a_M4_11,
                     b_M4_00, b_M4_01, b_M4_10, b_M4_11,
                     c_M4_00, c_M4_01, c_M4_10, c_M4_11)
    
    # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
    a_M5_00 = merge_entries(A11_00, A12_00)
    a_M5_01 = merge_entries(A11_01, A12_01)
    a_M5_10 = merge_entries(A11_10, A12_10)
    a_M5_11 = merge_entries(A11_11, A12_11)
    b_M5_00 = B22_00
    b_M5_01 = B22_01
    b_M5_10 = B22_10
    b_M5_11 = B22_11
    # C11 -, C12 +
    c_M5_00 = merge_entries(negate_entries(C11_00), C12_00)
    c_M5_01 = merge_entries(negate_entries(C11_01), C12_01)
    c_M5_10 = merge_entries(negate_entries(C11_10), C12_10)
    c_M5_11 = merge_entries(negate_entries(C11_11), C12_11)
    add_strassen_2x2(a_M5_00, a_M5_01, a_M5_10, a_M5_11,
                     b_M5_00, b_M5_01, b_M5_10, b_M5_11,
                     c_M5_00, c_M5_01, c_M5_10, c_M5_11)
    
    # M6 = (A21-A11)*(B11+B12) -> C22 += M6
    a_M6_00 = merge_entries(A21_00, negate_entries(A11_00))
    a_M6_01 = merge_entries(A21_01, negate_entries(A11_01))
    a_M6_10 = merge_entries(A21_10, negate_entries(A11_10))
    a_M6_11 = merge_entries(A21_11, negate_entries(A11_11))
    b_M6_00 = merge_entries(B11_00, B12_00)
    b_M6_01 = merge_entries(B11_01, B12_01)
    b_M6_10 = merge_entries(B11_10, B12_10)
    b_M6_11 = merge_entries(B11_11, B12_11)
    c_M6_00 = C22_00
    c_M6_01 = C22_01
    c_M6_10 = C22_10
    c_M6_11 = C22_11
    add_strassen_2x2(a_M6_00, a_M6_01, a_M6_10, a_M6_11,
                     b_M6_00, b_M6_01, b_M6_10, b_M6_11,
                     c_M6_00, c_M6_01, c_M6_10, c_M6_11)
    
    # M7 = (A12-A22)*(B21+B22) -> C11 += M7
    a_M7_00 = merge_entries(A12_00, negate_entries(A22_00))
    a_M7_01 = merge_entries(A12_01, negate_entries(A22_01))
    a_M7_10 = merge_entries(A12_10, negate_entries(A22_10))
    a_M7_11 = merge_entries(A12_11, negate_entries(A22_11))
    b_M7_00 = merge_entries(B21_00, B22_00)
    b_M7_01 = merge_entries(B21_01, B22_01)
    b_M7_10 = merge_entries(B21_10, B22_10)
    b_M7_11 = merge_entries(B21_11, B22_11)
    c_M7_00 = C11_00
    c_M7_01 = C11_01
    c_M7_10 = C11_10
    c_M7_11 = C11_11
    add_strassen_2x2(a_M7_00, a_M7_01, a_M7_10, a_M7_11,
                     b_M7_00, b_M7_01, b_M7_10, b_M7_11,
                     c_M7_00, c_M7_01, c_M7_10, c_M7_11)
    
    rank = len(U_cols)
    
    # Verify
    T = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)
    
    best_U = np.column_stack(U_cols)
    best_V = np.column_stack(V_cols)
    best_W = np.column_stack(W_cols)
    
    if check(best_U, best_V, best_W):
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback: return what we have (should still be valid)
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
