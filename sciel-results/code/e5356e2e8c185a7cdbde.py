"""Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 6 matrix."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Verify a decomposition
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
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
    
    # Strassen 2x2x2: returns list of (u_dict, v_dict, w_dict)
    # Each dict maps (local_index) -> coefficient
    def strassen_2x2x2():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
        terms.append((u, v, w))
        
        # M2 = (A00+A10)(B00+B01)
        u = {(0,0): 1, (1,0): 1}
        v = {(0,0): 1, (0,1): 1}
        w = {(1,0): 1, (1,1): 1}
        terms.append((u, v, w))
        
        # M3 = A00(B01-B11)
        u = {(0,0): 1}
        v = {(0,1): 1, (1,1): -1}
        w = {(0,1): 1}
        terms.append((u, v, w))
        
        # M4 = A11(B10-B00)
        u = {(1,1): 1}
        v = {(1,0): 1, (0,0): -1}
        w = {(0,0): 1, (1,0): 1}
        terms.append((u, v, w))
        
        # M5 = (A01+A11)B11
        u = {(0,1): 1, (1,1): 1}
        v = {(1,1): 1}
        w = {(0,1): 1, (1,1): -1}
        terms.append((u, v, w))
        
        # M6 = (A00-A10)(B10+B11)
        u = {(0,0): 1, (1,0): -1}
        v = {(1,0): 1, (1,1): 1}
        w = {(1,1): 1}
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B00+B10)
        u = {(0,1): 1, (1,1): -1}
        v = {(0,0): 1, (1,0): 1}
        w = {(0,0): 1}
        terms.append((u, v, w))
        
        return terms
    
    # Schoolbook for n×m × m×p
    def schoolbook_nm_p(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        return terms
    
    # 2x2 x 2x3: Use Strassen on first 2 cols + schoolbook on last col
    # A(2x2) * B(2x3) -> C(2x3)
    # Split B into B_left(2x2) and B_right(2x1)
    def decomp_2x2_2x3():
        terms = []
        # A(2x2) * B_left(2x2) using Strassen
        strassen_terms = strassen_2x2x2()
        for u, v, w in strassen_terms:
            # v keys are (j,k) with k in {0,1} for B_left
            # w keys are (k,i) with k in {0,1} for B_left
            terms.append((u, v, w))
        
        # A(2x2) * B_right(2x1) using schoolbook
        # B_right is column 2, so k=2
        for i in range(2):
            for j in range(2):
                u = {(i, j): 1}
                v = {(j, 2): 1}
                w = {(2, i): 1}
                terms.append((u, v, w))
        
        return terms
    
    # 2x3 x 3x3: Split A into A_left(2x2) + A_right(2x1)
    # Split B into B_top(2x3) + B_bottom(1x3)
    # Products: A_left*B_top(2x2x3), A_left*B_bottom(2x2x1), A_right*B_top(2x1x3), A_right*B_bottom(2x1x1)
    def decomp_2x3_3x3():
        terms = []
        
        # A_left(2x2) * B_top(2x3): use decomp_2x2_2x3
        terms_2x2_2x3 = decomp_2x2_2x3()
        for u, v, w in terms_2x2_2x3:
            # u: (i,j) with i in {0,1}, j in {0,1} -> A_left
            # v: (j,k) with j in {0,1}, k in {0,1,2} -> B_top
            # w: (k,i) with k in {0,1,2}, i in {0,1} -> C
            terms.append((u, v, w))
        
        # A_left(2x2) * B_bottom(1x3): schoolbook
        # B_bottom is row 2 of B, so j=2
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = {(i, j): 1}
                    v = {(2, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        
        # A_right(2x1) * B_top(2x3): schoolbook
        # A_right is column 2 of A, so j=2
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = {(i, 2): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        
        # A_right(2x1) * B_bottom(1x3): schoolbook
        for i in range(2):
            for k in range(3):
                u = {(i, 2): 1}
                v = {(2, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
        
        return terms
    
    # 2x2 x 3x3: Split B into B_top(2x3) + B_bottom(1x3)
    # A(2x2) * B_top(2x3): use decomp_2x2_2x3
    # A(2x2) * B_bottom(1x3): schoolbook
    def decomp_2x2_3x3():
        terms = []
        
        # A(2x2) * B_top(2x3)
        terms_2x2_2x3 = decomp_2x2_2x3()
        for u, v, w in terms_2x2_2x3:
            terms.append((u, v, w))
        
        # A(2x2) * B_bottom(1x3): schoolbook
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = {(i, j): 1}
                    v = {(2, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        
        return terms
    
    # Assemble full 4x5 x 5x6 decomposition
    # A(4x5) split into:
    #   A11: rows 0-1, cols 0-1 (2x2)
    #   A12: rows 0-1, cols 2-4 (2x3)
    #   A21: rows 2-3, cols 0-1 (2x2)
    #   A22: rows 2-3, cols 2-4 (2x3)
    
    # B(5x6) split into:
    #   B11: rows 0-1, cols 0-2 (2x3)
    #   B12: rows 0-1, cols 3-5 (2x3)
    #   B21: rows 2-4, cols 0-2 (3x3)
    #   B22: rows 2-4, cols 3-5 (3x3)
    
    # C(4x6) split into:
    #   C11: rows 0-1, cols 0-2 (2x3)
    #   C12: rows 0-1, cols 3-5 (2x3)
    #   C21: rows 2-3, cols 0-2 (2x3)
    #   C22: rows 2-3, cols 3-5 (2x3)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_dict, v_dict, w_dict, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        u_global = [0.0] * (n * m)
        v_global = [0.0] * (m * p)
        w_global = [0.0] * (p * n)
        
        for (i, j), val in u_dict.items():
            gi = a_row_off + i
            gj = a_col_off + j
            u_global[gi * m + gj] += val
        
        for (j, k), val in v_dict.items():
            gj = b_row_off + j
            gk = b_col_off + k
            v_global[gj * p + gk] += val
        
        for (k, i), val in w_dict.items():
            gi = c_row_off + i
            gk = c_col_off + k
            w_global[gk * n + gi] += val
        
        U_list.append(u_global)
        V_list.append(v_global)
        W_list.append(w_global)
    
    # C11 = A11*B11 + A12*B21
    terms = decomp_2x2_2x3()
    for u, v, w in terms:
        add_term(u, v, w, 0, 0, 0, 0, 0, 0)
    
    terms = decomp_2x3_3x3()
    for u, v, w in terms:
        add_term(u, v, w, 0, 2, 2, 0, 0, 0)
    
    # C12 = A11*B12 + A12*B22
    terms = decomp_2x2_2x3()
    for u, v, w in terms:
        add_term(u, v, w, 0, 0, 0, 3, 0, 3)
    
    terms = decomp_2x3_3x3()
    for u, v, w in terms:
        add_term(u, v, w, 0, 2, 2, 3, 0, 3)
    
    # C21 = A21*B11 + A22*B21
    terms = decomp_2x2_2x3()
    for u, v, w in terms:
        add_term(u, v, w, 2, 0, 0, 0, 2, 0)
    
    terms = decomp_2x3_3x3()
    for u, v, w in terms:
        add_term(u, v, w, 2, 2, 2, 0, 2, 0)
    
    # C22 = A21*B12 + A22*B22
    terms = decomp_2x2_2x3()
    for u, v, w in terms:
        add_term(u, v, w, 2, 0, 0, 3, 2, 3)
    
    terms = decomp_2x3_3x3()
    for u, v, w in terms:
        add_term(u, v, w, 2, 2, 2, 3, 2, 3)
    
    # Convert to final format
    rank = len(U_list)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_list[c][i]
        for j in range(m * p):
            V[j][c] = V_list[c][j]
        for k in range(p * n):
            W[k][c] = W_list[c][k]
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if verify(U, V, W):
        return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
