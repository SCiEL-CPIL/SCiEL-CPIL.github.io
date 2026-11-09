"""Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 6 matrix."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np
from itertools import product

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
    
    # Strassen 2x2x2 decomposition
    # Returns list of (u_dict, v_dict, w_dict) where keys are (i,j), (j,k), (k,i)
    def strassen_2x2x2():
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        # M2 = (A00+A10)(B00+B01) -> contributes to C10, C11
        # M3 = A00(B01-B11) -> contributes to C01
        # M4 = A11(B10-B00) -> contributes to C00, C10
        # M5 = (A01+A11)B11 -> contributes to C01, C11
        # M6 = (A00-A10)(B10+B11) -> contributes to C11
        # M7 = (A01-A11)(B00+B10) -> contributes to C00
        
        # Standard Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        terms = []
        
        # Term 1: M1
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
        terms.append((u, v, w))
        
        # Term 2: M2
        u = {(0,0): 1, (1,0): 1}
        v = {(0,0): 1, (0,1): 1}
        w = {(0,1): 1, (1,1): 1}
        terms.append((u, v, w))
        
        # Term 3: M3
        u = {(0,0): 1}
        v = {(0,1): 1, (1,1): -1}
        w = {(1,0): 1}
        terms.append((u, v, w))
        
        # Term 4: M4
        u = {(1,1): 1}
        v = {(1,0): 1, (0,0): -1}
        w = {(0,0): 1, (0,1): 1}
        terms.append((u, v, w))
        
        # Term 5: M5
        u = {(0,1): 1, (1,1): 1}
        v = {(1,1): 1}
        w = {(1,0): 1, (1,1): -1}
        terms.append((u, v, w))
        
        # Term 6: M6
        u = {(0,0): 1, (1,0): -1}
        v = {(1,0): 1, (1,1): 1}
        w = {(1,1): 1}
        terms.append((u, v, w))
        
        # Term 7: M7
        u = {(0,1): 1, (1,1): -1}
        v = {(0,0): 1, (1,0): 1}
        w = {(0,0): 1}
        terms.append((u, v, w))
        
        return terms
    
    # Schoolbook for nxm x pxq -> (n*p) x (m*q)
    # Returns list of (u_dict, v_dict, w_dict)
    def schoolbook_block(n, m, p, q):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        return terms
    
    # 2x2 x 2x3: Use Strassen on left 2x2x2 part + schoolbook on 2x2x1 part
    # Split B (2x3) into B_left (2x2) and B_right (2x1)
    def decomp_2x2_x_2x3():
        terms = []
        
        # Part 1: A (2x2) * B_left (2x2) using Strassen
        strassen_terms = strassen_2x2x2()
        for u, v, w in strassen_terms:
            # Map B indices: (j,k) where j in 0,1 and k in 0,1 (left part)
            # This is already correct for the 2x2x2 case
            terms.append((u, v, w))
        
        # Part 2: A (2x2) * B_right (2x1) using schoolbook
        # B_right has columns k=2 only
        for i in range(2):
            for j in range(2):
                u = {(i, j): 1}
                v = {(j, 2): 1}
                w = {(2, i): 1}
                terms.append((u, v, w))
        
        return terms
    
    # 2x3 x 3x3: Need a decomposition
    # Split A (2x3) into A_left (2x2) and A_right (2x1)
    # Split B (3x3) into B_top (2x3) and B_bottom (1x3)
    # Products: A_left*B_top (2x2x3), A_left*B_bottom (2x2x1), A_right*B_top (2x1x3), A_right*B_bottom (2x1x1)
    def decomp_2x3_x_3x3():
        terms = []
        
        # A_left (2x2) * B_top (2x3): use 2x2x3 decomposition
        terms_2x2x3 = decomp_2x2_x_2x3()
        for u, v, w in terms_2x2x3:
            # u: keys (i,j) with i in 0,1, j in 0,1 -> already correct
            # v: keys (j,k) with j in 0,1, k in 0,1,2 -> j in 0,1, k in 0,1,2 (B_top is rows 0,1 of B)
            # w: keys (k,i) with k in 0,1,2, i in 0,1 -> already correct
            terms.append((u, v, w))
        
        # A_left (2x2) * B_bottom (1x3): schoolbook
        # B_bottom is row 2 of B, so j=2
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = {(i, j): 1}
                    v = {(2, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        
        # A_right (2x1) * B_top (2x3): schoolbook
        # A_right is column 2 of A, so j=2
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = {(i, 2): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        
        # A_right (2x1) * B_bottom (1x3): schoolbook
        for i in range(2):
            for k in range(3):
                u = {(i, 2): 1}
                v = {(2, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
        
        return terms
    
    # 2x2 x 3x3: Split B (3x3) into B_top (2x3) and B_bottom (1x3)
    # A (2x2) * B_top (2x3): use 2x2x3
    # A (2x2) * B_bottom (1x3): schoolbook
    def decomp_2x2_x_3x3():
        terms = []
        
        # A (2x2) * B_top (2x3)
        terms_2x2x3 = decomp_2x2_x_2x3()
        for u, v, w in terms_2x2x3:
            # v: keys (j,k) with j in 0,1, k in 0,1,2 -> B_top is rows 0,1
            terms.append((u, v, w))
        
        # A (2x2) * B_bottom (1x3): schoolbook
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = {(i, j): 1}
                    v = {(2, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        
        return terms
    
    # 2x3 x 3x3: already defined above
    
    # Now assemble the full 4x5 x 5x6 decomposition
    # A (4x5) -> A11 (2x2), A12 (2x3), A21 (2x2), A22 (2x3)
    # B (5x6) -> B11 (2x3), B12 (2x3), B21 (3x3), B22 (3x3)
    # C (4x6) -> C11 (2x3), C12 (2x3), C21 (2x3), C22 (2x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We need to map local indices to global indices
    
    # A global: rows 0-3, cols 0-4
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-4
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-4
    
    # B global: rows 0-4, cols 0-5
    # B11: rows 0-1, cols 0-2
    # B12: rows 0-1, cols 3-5
    # B21: rows 2-4, cols 0-2
    # B22: rows 2-4, cols 3-5
    
    # C global: rows 0-3, cols 0-5
    # C11: rows 0-1, cols 0-2
    # C12: rows 0-1, cols 3-5
    # C21: rows 2-3, cols 0-2
    # C22: rows 2-3, cols 3-5
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_dict, v_dict, w_dict, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Add a term with local indices mapped to global indices."""
        u_global = [0.0] * (n * m)
        v_global = [0.0] * (m * p)
        w_global = [0.0] * (p * n)
        
        for (i, j), val in u_dict.items():
            gi = a_row_offset + i
            gj = a_col_offset + j
            u_global[gi * m + gj] += val
        
        for (j, k), val in v_dict.items():
            gj = b_row_offset + j
            gk = b_col_offset + k
            v_global[gj * p + gk] += val
        
        for (k, i), val in w_dict.items():
            gi = c_row_offset + i
            gk = c_col_offset + k
            w_global[gk * n + gi] += val
        
        U_list.append(u_global)
        V_list.append(v_global)
        W_list.append(w_global)
    
    # Product 1: A11 (2x2) * B11 (2x3) -> contributes to C11 (2x3)
    terms = decomp_2x2_x_2x3()
    for u, v, w in terms:
        add_term(u, v, w, 0, 0, 0, 0, 0, 0)
    
    # Product 2: A12 (2x3) * B21 (3x3) -> contributes to C11 (2x3)
    terms = decomp_2x3_x_3x3()
    for u, v, w in terms:
        add_term(u, v, w, 0, 2, 2, 0, 0, 0)
    
    # Product 3: A11 (2x2) * B12 (2x3) -> contributes to C12 (2x3)
    terms = decomp_2x2_x_2x3()
    for u, v, w in terms:
        add_term(u, v, w, 0, 0, 0, 3, 0, 3)
    
    # Product 4: A12 (2x3) * B22 (3x3) -> contributes to C12 (2x3)
    terms = decomp_2x3_x_3x3()
    for u, v, w in terms:
        add_term(u, v, w, 0, 2, 2, 3, 0, 3)
    
    # Product 5: A21 (2x2) * B11 (2x3) -> contributes to C21 (2x3)
    terms = decomp_2x2_x_2x3()
    for u, v, w in terms:
        add_term(u, v, w, 2, 0, 0, 0, 2, 0)
    
    # Product 6: A22 (2x3) * B21 (3x3) -> contributes to C21 (2x3)
    terms = decomp_2x3_x_3x3()
    for u, v, w in terms:
        add_term(u, v, w, 2, 2, 2, 0, 2, 0)
    
    # Product 7: A21 (2x2) * B12 (2x3) -> contributes to C22 (2x3)
    terms = decomp_2x2_x_2x3()
    for u, v, w in terms:
        add_term(u, v, w, 2, 0, 0, 3, 2, 3)
    
    # Product 8: A22 (2x3) * B22 (3x3) -> contributes to C22 (2x3)
    terms = decomp_2x3_x_3x3()
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
