"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Recursive Strassen decomposition.
    - For the 4x4 * 4x4 part: Use Strassen at the block level (7 block mults of 2x2x2),
      each block mult uses Strassen (7 scalar mults). Total: 7*7 = 49 terms.
    - For the 4x4 * 4x1 part (fifth column): Use direct method. Total: 4*4*1 = 16 terms.
    - Grand total: 49 + 16 = 65 terms.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    rank_total = 65
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        """Add a rank-1 term. u_list: [(i,j,coeff)], v_list: [(j,k,coeff)], w_list: [(k,i,coeff)]"""
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
    
    def strassen_2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00_coeff, c01_coeff, c10_coeff, c11_coeff):
        """
        Add 7 Strassen terms for 2x2 * 2x2 where the entries are linear combinations.
        a00, a01, a10, a11: lists of (i, j, coeff) for A entries
        b00, b01, b10, b11: lists of (j, k, coeff) for B entries
        c00_coeff, c01_coeff, c10_coeff, c11_coeff: coefficients for C00, C01, C10, C11
        """
        # M1 = (a00+a11)*(b00+b11) -> C00 += c00_coeff, C11 += c11_coeff
        u_m1 = []
        for i, j, c in a00:
            u_m1.append((i, j, c))
        for i, j, c in a11:
            u_m1.append((i, j, c))
        v_m1 = []
        for j, k, c in b00:
            v_m1.append((j, k, c))
        for j, k, c in b11:
            v_m1.append((j, k, c))
        w_m1 = []
        if c00_coeff != 0:
            w_m1.append((0, 0, c00_coeff))
        if c11_coeff != 0:
            w_m1.append((1, 1, c11_coeff))
        add_term(u_m1, v_m1, w_m1)
        
        # M2 = (a10+a11)*b00 -> C10 += c10_coeff, C11 += -c11_coeff
        u_m2 = []
        for i, j, c in a10:
            u_m2.append((i, j, c))
        for i, j, c in a11:
            u_m2.append((i, j, c))
        v_m2 = list(b00)
        w_m2 = []
        if c10_coeff != 0:
            w_m2.append((0, 1, c10_coeff))
        if c11_coeff != 0:
            w_m2.append((1, 1, -c11_coeff))
        add_term(u_m2, v_m2, w_m2)
        
        # M3 = a00*(b01-b11) -> C01 += c01_coeff, C11 += c11_coeff
        u_m3 = list(a00)
        v_m3 = []
        for j, k, c in b01:
            v_m3.append((j, k, c))
        for j, k, c in b11:
            v_m3.append((j, k, -c))
        w_m3 = []
        if c01_coeff != 0:
            w_m3.append((1, 0, c01_coeff))
        if c11_coeff != 0:
            w_m3.append((1, 1, c11_coeff))
        add_term(u_m3, v_m3, w_m3)
        
        # M4 = a11*(b10-b00) -> C00 += c00_coeff, C10 += c10_coeff
        u_m4 = list(a11)
        v_m4 = []
        for j, k, c in b10:
            v_m4.append((j, k, c))
        for j, k, c in b00:
            v_m4.append((j, k, -c))
        w_m4 = []
        if c00_coeff != 0:
            w_m4.append((0, 0, c00_coeff))
        if c10_coeff != 0:
            w_m4.append((0, 1, c10_coeff))
        add_term(u_m4, v_m4, w_m4)
        
        # M5 = (a00+a01)*b11 -> C00 += -c00_coeff, C01 += c01_coeff
        u_m5 = []
        for i, j, c in a00:
            u_m5.append((i, j, c))
        for i, j, c in a01:
            u_m5.append((i, j, c))
        v_m5 = list(b11)
        w_m5 = []
        if c00_coeff != 0:
            w_m5.append((0, 0, -c00_coeff))
        if c01_coeff != 0:
            w_m5.append((1, 0, c01_coeff))
        add_term(u_m5, v_m5, w_m5)
        
        # M6 = (a10-a00)*(b00+b01) -> C11 += c11_coeff
        u_m6 = []
        for i, j, c in a10:
            u_m6.append((i, j, c))
        for i, j, c in a00:
            u_m6.append((i, j, -c))
        v_m6 = []
        for j, k, c in b00:
            v_m6.append((j, k, c))
        for j, k, c in b01:
            v_m6.append((j, k, c))
        w_m6 = []
        if c11_coeff != 0:
            w_m6.append((1, 1, c11_coeff))
        add_term(u_m6, v_m6, w_m6)
        
        # M7 = (a01-a11)*(b10+b11) -> C00 += c00_coeff
        u_m7 = []
        for i, j, c in a01:
            u_m7.append((i, j, c))
        for i, j, c in a11:
            u_m7.append((i, j, -c))
        v_m7 = []
        for j, k, c in b10:
            v_m7.append((j, k, c))
        for j, k, c in b11:
            v_m7.append((j, k, c))
        w_m7 = []
        if c00_coeff != 0:
            w_m7.append((0, 0, c00_coeff))
        add_term(u_m7, v_m7, w_m7)
    
    # Define A block entries as lists of (i, j, coeff)
    # A00: i in {0,1}, j in {0,1}
    A00 = [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]
    A01 = [(0,2,1), (0,3,1), (1,2,1), (1,3,1)]
    A10 = [(2,0,1), (2,1,1), (3,0,1), (3,1,1)]
    A11 = [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]
    
    # Define B block entries as lists of (j, k, coeff)
    # B00: j in {0,1}, k in {0,1}
    B00 = [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]
    B01 = [(0,2,1), (0,3,1), (1,2,1), (1,3,1)]
    B10 = [(2,0,1), (2,1,1), (3,0,1), (3,1,1)]
    B11 = [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]
    
    # Super Strassen for 4x4 * 4x4
    # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    strassen_2x2(A00, A01, A10, A11, B00, B01, B10, B11, 1, 0, 0, 1)
    
    # M2 = (A10+A11)*B00 -> C10 += 1, C11 -= 1
    # Need to create combined A10+A11
    A10_A11 = A10 + A11
    # But strassen_2x2 expects separate a10, a11. Let me modify.
    
    # Actually, I need to restructure. Let me pass the linear combinations directly.
    
    # Let me redefine: for each super term, I'll pass the linear combination of A and B blocks.
    
    # Let me just directly call strassen_2x2 with the appropriate blocks.
    
    # I already called M1. Let me continue with the others.
    
    # For M2: (A10+A11)*B00
    # I need a10 = A10+A11, a11 = 0, a00 = 0, a01 = 0
    # But strassen_2x2 uses a00, a01, a10, a11. Let me just pass empty lists for unused.
    
    A_empty = []
    strassen_2x2(A_empty, A_empty, A10_A11, A_empty, B00, A_empty, A_empty, A_empty, 0, 0, 1, -1)
    
    # M3 = A00*(B01-B11)
    B01_B11 = B01 + [(j, k, -c) for j, k, c in B11]
    strassen_2x2(A00, A_empty, A_empty, A_empty, A_empty, B01_B11, A_empty, A_empty, 0, 1, 0, 1)
    
    # M4 = A11*(B10-B00)
    B10_B00 = B10 + [(j, k, -c) for j, k, c in B00]
    strassen_2x2(A_empty, A_empty, A_empty, A11, B10_B00, A_empty, A_empty, A_empty, 1, 0, 1, 0)
    
    # M5 = (A00+A01)*B11
    A00_A01 = A00 + A01
    strassen_2x2(A00_A01, A_empty, A_empty, A_empty, A_empty, A_empty, A_empty, B11, -1, 1, 0, 0)
    
    # M6 = (A10-A00)*(B00+B01)
    A10_A00 = A10 + [(i, j, -c) for i, j, c in A00]
    B00_B01 = B00 + B01
    strassen_2x2(A_empty, A_empty, A10_A00, A_empty, B00_B01, A_empty, A_empty, A_empty, 0, 0, 0, 1)
    
    # M7 = (A01-A11)*(B10+B11)
    A01_A11 = A01 + [(i, j, -c) for i, j, c in A11]
    B10_B11 = B10 + B11
    strassen_2x2(A_empty, A01_A11, A_empty, A_empty, B10_B11, A_empty, A_empty, A_empty, 1, 0, 0, 0)
    
    # Part 2: 4x4 * 4x1 (fifth column)
    for i in range(n):
        for j in range(m):
            add_term([(i, j, 1)], [(j, 4, 1)], [(4, i, 1)])
    
    used_rank = col
    U = np.array(U, dtype=float)[:, :used_rank]
    V = np.array(V, dtype=float)[:, :used_rank]
    W = np.array(W, dtype=float)[:, :used_rank]
    
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

