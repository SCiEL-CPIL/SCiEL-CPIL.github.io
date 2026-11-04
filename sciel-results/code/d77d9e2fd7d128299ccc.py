import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # U: (n*m, rank) = (12, R)
    # V: (m*p, rank) = (32, R)
    # W: (p*n, rank) = (24, R)
    
    # We will build the decomposition in two parts.
    # Part 1: i=0 (1x4x8 schoolbook)
    # Part 2: i=1,2 (2x4x8 using Strassen blocks)
    
    U_list = []
    V_list = []
    W_list = []
    
    # --- Part 1: i=0 ---
    # C[0, k] = sum_j A[0, j] B[j, k]
    # U index: i*m + j = 0*4 + j = j
    # V index: j*p + k = j*8 + k
    # W index: k*n + i = k*3 + 0 = 3k
    
    for j in range(m):
        for k in range(p):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            u[j] = 1
            v[j * p + k] = 1
            w[k * n + 0] = 1
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
    # --- Part 2: i=1,2 ---
    # A2 is 2x4. Rows 1,2 of A.
    # B is 4x8.
    # We split B into B_L (cols 0-3) and B_R (cols 4-7).
    # We need to compute C2 = A2 * B.
    # C2_L = A2 * B_L (2x4)
    # C2_R = A2 * B_R (2x4)
    
    # For each 2x4x4 multiplication (e.g., A2 * B_L):
    # A2 = [A_L | A_R] where A_L, A_R are 2x2.
    # B_L = [[B11, B12], [B21, B22]] where blocks are 2x2.
    # C_L = [A_L B11 + A_R B21 | A_L B12 + A_R B22]
    
    # We use Strassen for each 2x2x2 multiplication.
    # There are 4 such multiplications per 2x4x4 block.
    # So 8 such multiplications total for Part 2.
    # Each Strassen uses 7 terms.
    # Total terms for Part 2 = 8 * 7 = 56.
    
    # Helper to add Strassen terms for a 2x2x2 multiplication
    # X (2x2) from A part, Y (2x2) from B part.
    # X is part of A2, Y is part of B.
    # The result contributes to C.
    
    def add_strassen_term(X_indices, Y_indices, C_indices, coeff_X, coeff_Y, coeff_C):
        # This function is a bit abstract. Let's just hardcode the 7 Strassen patterns.
        pass

    # Let's define the Strassen patterns explicitly.
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = XY
    
    # M1 = (x00+x11)(y00+y11) -> Z00 += M1, Z11 += M1
    # M2 = (x10+x11)y00 -> Z10 += M2
    # M3 = x00(y01-y11) -> Z01 += M3, Z11 += M3
    # M4 = x11(y10-y00) -> Z00 += M4, Z10 += M4
    # M5 = (x00+x01)y11 -> Z00 -= M5, Z01 += M5
    # M6 = (x10-x00)(y00+y01) -> Z11 += M6
    # M7 = (x01-x11)(y10+y11) -> Z00 += M7
    
    # We need to map these to the global U, V, W indices.
    # For a specific 2x2x2 multiplication:
    # X comes from A2. A2 rows are 1,2 of A.
    # X indices in A:
    # x00 -> A[1, col_x00]
    # x01 -> A[1, col_x01]
    # x10 -> A[2, col_x10]
    # x11 -> A[2, col_x11]
    
    # Y comes from B.
    # y00 -> B[row_y00, col_y00]
    # etc.
    
    # Z comes from C.
    # z00 -> C[row_z00, col_z00]
    # etc.
    
    # We will iterate over the 8 blocks.
    # Block 0: A_L B11 (for C_L left)
    # Block 1: A_R B21 (for C_L left)
    # Block 2: A_L B12 (for C_L right)
    # Block 3: A_R B22 (for C_L right)
    # Block 4-7: Same for B_R
    
    # A_L is A[1:3, 0:2]
    # A_R is A[1:3, 2:4]
    
    # B_L is B[0:4, 0:4]
    # B11 is B[0:2, 0:2]
    # B12 is B[0:2, 2:4]
    # B21 is B[2:4, 0:2]
    # B22 is B[2:4, 2:4]
    
    # C_L is C[1:3, 0:4]
    # C_L left is C[1:3, 0:2]
    # C_L right is C[1:3, 2:4]
    
    # We define a helper to add a single Strassen multiplication.
    
    def add_strassen_mult(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # a_row_offset: 1 or 2 (row in A2, so 1 or 2 in A)
        # a_col_offset: 0,1,2,3 (col in A2, so 0..3 in A)
        # b_row_offset: 0..3 (row in B)
        # b_col_offset: 0..7 (col in B)
        # c_row_offset: 1 or 2 (row in C2, so 1 or 2 in C)
        # c_col_offset: 0..7 (col in C)
        
        # X is 2x2, Y is 2x2.
        # X[0,0] is A[a_row_offset, a_col_offset]
        # X[0,1] is A[a_row_offset, a_col_offset+1]
        # X[1,0] is A[a_row_offset+1, a_col_offset]
        # X[1,1] is A[a_row_offset+1, a_col_offset+1]
        
        # Y[0,0] is B[b_row_offset, b_col_offset]
        # Y[0,1] is B[b_row_offset, b_col_offset+1]
        # Y[1,0] is B[b_row_offset+2, b_col_offset]
        # Y[1,1] is B[b_row_offset+2, b_col_offset+1]
        
        # Z[0,0] is C[c_row_offset, c_col_offset]
        # Z[0,1] is C[c_row_offset, c_col_offset+1]
        # Z[1,0] is C[c_row_offset+1, c_col_offset]
        # Z[1,1] is C[c_row_offset+1, c_col_offset+1]
        
        # 7 Strassen terms
        
        # M1: (x00+x11)(y00+y11)
        # u: A[a_row_offset, a_col_offset] + A[a_row_offset+1, a_col_offset+1]
        # v: B[b_row_offset, b_col_offset] + B[b_row_offset+2, b_col_offset+1]
        # w: C[c_row_offset, c_col_offset] + C[c_row_offset+1, c_col_offset+1]
        
        u1 = [0] * (n * m)
        v1 = [0] * (m * p)
        w1 = [0] * (p * n)
        
        u1[a_row_offset * m + a_col_offset] = 1
        u1[(a_row_offset + 1) * m + (a_col_offset + 1)] = 1
        
        v1[b_row_offset * p + b_col_offset] = 1
        v1[(b_row_offset + 2) * p + (b_col_offset + 1)] = 1
        
        w1[c_col_offset * n + c_row_offset] = 1
        w1[(c_col_offset + 1) * n + (c_row_offset + 1)] = 1
        
        U_list.append(u1)
        V_list.append(v1)
        W_list.append(w1)
        
        # M2: (x10+x11)y00
        # u: A[a_row_offset+1, a_col_offset] + A[a_row_offset+1, a_col_offset+1]
        # v: B[b_row_offset, b_col_offset]
        # w: C[c_row_offset+1, c_col_offset]
        
        u2 = [0] * (n * m)
        v2 = [0] * (m * p)
        w2 = [0] * (p * n)
        
        u2[(a_row_offset + 1) * m + a_col_offset] = 1
        u2[(a_row_offset + 1) * m + (a_col_offset + 1)] = 1
        
        v2[b_row_offset * p + b_col_offset] = 1
        
        w2[c_col_offset * n + (c_row_offset + 1)] = 1
        
        U_list.append(u2)
        V_list.append(v2)
        W_list.append(w2)
        
        # M3: x00(y01-y11)
        # u: A[a_row_offset, a_col_offset]
        # v: B[b_row_offset, b_col_offset+1] - B[b_row_offset+2, b_col_offset+1]
        # w: C[c_row_offset, c_col_offset+1] + C[c_row_offset+1, c_col_offset+1]
        
        u3 = [0] * (n * m)
        v3 = [0] * (m * p)
        w3 = [0] * (p * n)
        
        u3[a_row_offset * m + a_col_offset] = 1
        
        v3[b_row_offset * p + (b_col_offset + 1)] = 1
        v3[(b_row_offset + 2) * p + (b_col_offset + 1)] = -1
        
        w3[(c_col_offset + 1) * n + c_row_offset] = 1
        w3[(c_col_offset + 1) * n + (c_row_offset + 1)] = 1
        
        U_list.append(u3)
        V_list.append(v3)
        W_list.append(w3)
        
        # M4: x11(y10-y00)
        # u: A[a_row_offset+1, a_col_offset+1]
        # v: B[b_row_offset+2, b_col_offset] - B[b_row_offset, b_col_offset]
        # w: C[c_row_offset, c_col_offset] + C[c_row_offset+1, c_col_offset]
        
        u4 = [0] * (n * m)
        v4 = [0] * (m * p)
        w4 = [0] * (p * n)
        
        u4[(a_row_offset + 1) * m + (a_col_offset + 1)] = 1
        
        v4[(b_row_offset + 2) * p + b_col_offset] = 1
        v4[b_row_offset * p + b_col_offset] = -1
        
        w4[c_col_offset * n + c_row_offset] = 1
        w4[c_col_offset * n + (c_row_offset + 1)] = 1
        
        U_list.append(u4)
        V_list.append(v4)
        W_list.append(w4)
        
        # M5: (x00+x01)y11
        # u: A[a_row_offset, a_col_offset] + A[a_row_offset, a_col_offset+1]
        # v: B[b_row_offset+2, b_col_offset+1]
        # w: -C[c_row_offset, c_col_offset] + C[c_row_offset, c_col_offset+1]
        
        u5 = [0] * (n * m)
        v5 = [0] * (m * p)
        w5 = [0] * (p * n)
        
        u5[a_row_offset * m + a_col_offset] = 1
        u5[a_row_offset * m + (a_col_offset + 1)] = 1
        
        v5[(b_row_offset + 2) * p + (b_col_offset + 1)] = 1
        
        w5[c_col_offset * n + c_row_offset] = -1
        w5[(c_col_offset + 1) * n + c_row_offset] = 1
        
        U_list.append(u5)
        V_list.append(v5)
        W_list.append(w5)
        
        # M6: (x10-x00)(y00+y01)
        # u: A[a_row_offset+1, a_col_offset] - A[a_row_offset, a_col_offset]
        # v: B[b_row_offset, b_col_offset] + B[b_row_offset, b_col_offset+1]
        # w: C[c_row_offset+1, c_col_offset+1]
        
        u6 = [0] * (n * m)
        v6 = [0] * (m * p)
        w6 = [0] * (p * n)
        
        u6[(a_row_offset + 1) * m + a_col_offset] = 1
        u6[a_row_offset * m + a_col_offset] = -1
        
        v6[b_row_offset * p + b_col_offset] = 1
        v6[b_row_offset * p + (b_col_offset + 1)] = 1
        
        w6[(c_col_offset + 1) * n + (c_row_offset + 1)] = 1
        
        U_list.append(u6)
        V_list.append(v6)
        W_list.append(w6)
        
        # M7: (x01-x11)(y10+y11)
        # u: A[a_row_offset, a_col_offset+1] - A[a_row_offset+1, a_col_offset+1]
        # v: B[b_row_offset+2, b_col_offset] + B[b_row_offset+2, b_col_offset+1]
        # w: C[c_row_offset, c_col_offset]
        
        u7 = [0] * (n * m)
        v7 = [0] * (m * p)
        w7 = [0] * (p * n)
        
        u7[a_row_offset * m + (a_col_offset + 1)] = 1
        u7[(a_row_offset + 1) * m + (a_col_offset + 1)] = -1
        
        v7[(b_row_offset + 2) * p + b_col_offset] = 1
        v7[(b_row_offset + 2) * p + (b_col_offset + 1)] = 1
        
        w7[c_col_offset * n + c_row_offset] = 1
        
        U_list.append(u7)
        V_list.append(v7)
        W_list.append(w7)
        
    # Call add_strassen_mult for the 8 blocks
    
    # Block 0: A_L B11 -> C_L Left
    # A_L: row 1, col 0. (a_row=1, a_col=0)
    # B11: row 0, col 0. (b_row=0, b_col=0)
    # C_L Left: row 1, col 0. (c_row=1, c_col=0)
    add_strassen_mult(1, 0, 0, 0, 1, 0)
    
    # Block 1: A_R B21 -> C_L Left
    # A_R: row 1, col 2. (a_row=1, a_col=2)
    # B21: row 2, col 0. (b_row=2, b_col=0)
    # C_L Left: row 1, col 0. (c_row=1, c_col=0)
    add_strassen_mult(1, 2, 2, 0, 1, 0)
    
    # Block 2: A_L B12 -> C_L Right
    # A_L: row 1, col 0.
    # B12: row 0, col 2.
    # C_L Right: row 1, col 2.
    add_strassen_mult(1, 0, 0, 2, 1, 2)
    
    # Block 3: A_R B22 -> C_L Right
    # A_R: row 1, col 2.
    # B22: row 2, col 2.
    # C_L Right: row 1, col 2.
    add_strassen_mult(1, 2, 2, 2, 1, 2)
    
    # Block 4: A_L B11 -> C_L Left (for row 2 of C)
    # Wait, Strassen computes the whole 2x2 Z.
    # Z[0,0] is C[1, c_col], Z[1,0] is C[2, c_col].
    # So one Strassen call handles both rows of C for a given 2x2 block of B and 2x2 block of A.
    # So Block 0 handles C[1,0], C[1,1], C[2,0], C[2,1].
    # This covers C_L Left completely.
    
    # So we only need 4 Strassen calls for B_L (Left half of B).
    # And 4 Strassen calls for B_R (Right half of B).
    
    # Block 4: A_L B11_R -> C_R Left
    # A_L: row 1, col 0.
    # B11_R: row 0, col 4. (b_row=0, b_col=4)
    # C_R Left: row 1, col 4.
    add_strassen_mult(1, 0, 0, 4, 1, 4)
    
    # Block 5: A_R B21_R -> C_R Left
    # A_R: row 1, col 2.
    # B21_R: row 2, col 4.
    # C_R Left: row 1, col 4.
    add_strassen_mult(1, 2, 2, 4, 1, 4)
    
    # Block 6: A_L B12_R -> C_R Right
    # A_L: row 1, col 0.
    # B12_R: row 0, col 6.
    # C_R Right: row 1, col 6.
    add_strassen_mult(1, 0, 0, 6, 1, 6)
    
    # Block 7: A_R B22_R -> C_R Right
    # A_R: row 1, col 2.
    # B22_R: row 2, col 6.
    # C_R Right: row 1, col 6.
    add_strassen_mult(1, 2, 2, 6, 1, 6)
    
    # Total rank = 32 + 8*7 = 32 + 56 = 88.
    
    rank = len(U_list)
    U = [list(map(int, u)) for u in U_list]
    V = [list(map(int, v)) for v in V_list]
    W = [list(map(int, w)) for w in W_list]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
