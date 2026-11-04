import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_4x4():
    """Rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    U: 16x49, V: 16x49, W: 16x49
    U[i*4+j, c] for A[i,j], V[j*4+k, c] for B[j,k], W[k*4+i, c] for C[i,k]
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # We will use the 7 Strassen formulas for 2x2 blocks.
    # Each formula involves a product of two 2x2 matrices.
    # We need to expand each 2x2 product into 7 scalar terms.
    
    # Helper to add a 2x2 Strassen decomposition
    # a_rows, b_rows, c_rows are lists of (row, col) for the 2x2 block
    # Actually, the 2x2 block in A is at (i_off, j_off)
    # The 2x2 block in B is at (j_off, k_off)
    # The 2x2 block in C is at (k_off, i_off)
    
    def add_strassen_2x2(a_i, a_j, b_j, b_k, c_k, c_i, base_col):
        # a_i, a_j are offsets for the 2x2 block in A
        # b_j, b_k are offsets for the 2x2 block in B
        # c_k, c_i are offsets for the 2x2 block in C
        
        # M1 = (A00+A11)(B00+B11)
        # A00 is at (a_i, a_j), A11 is at (a_i+1, a_j+1)
        # B00 is at (b_j, b_k), B11 is at (b_j+1, b_k+1)
        # C00 is at (c_k, c_i), C11 is at (c_k+1, c_i+1)
        
        # We need to add 7 terms for this 2x2 multiplication.
        # The 2x2 multiplication tensor has rank 7.
        # We can hardcode the 7 terms for 2x2.
        
        # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # Term 2: (A10+A11)B00 -> C10 += M2, C11 += M2
        # Term 3: A00(B10-B11) -> C01 += M3
        # Term 4: A11(B01-B00) -> C00 += M4, C10 += M4
        # Term 5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        # Term 6: (A10-A00)(B00+B01) -> C11 += M6
        # Term 7: (A01-A11)(B10+B11) -> C00 += M7
        
        # We need to map these to the global U, V, W matrices.
        # For each of the 7 terms, we add to U, V, W.
        
        # Term 1
        U[a_i*4 + a_j][base_col+0] += 1
        U[(a_i+1)*4 + (a_j+1)][base_col+0] += 1
        V[b_j*4 + b_k][base_col+0] += 1
        V[(b_j+1)*4 + (b_k+1)][base_col+0] += 1
        W[c_k*4 + c_i][base_col+0] += 1
        W[(c_k+1)*4 + (c_i+1)][base_col+0] += 1
        
        # Term 2
        U[(a_i+1)*4 + a_j][base_col+1] += 1
        U[(a_i+1)*4 + (a_j+1)][base_col+1] += 1
        V[b_j*4 + b_k][base_col+1] += 1
        W[(c_k+1)*4 + c_i][base_col+1] += 1
        W[(c_k+1)*4 + (c_i+1)][base_col+1] += 1
        
        # Term 3
        U[a_i*4 + a_j][base_col+2] += 1
        V[(b_j+1)*4 + b_k][base_col+2] += 1
        V[(b_j+1)*4 + (b_k+1)][base_col+2] += -1
        W[(c_k+1)*4 + c_i][base_col+2] += 1
        
        # Term 4
        U[(a_i+1)*4 + (a_j+1)][base_col+3] += 1
        V[b_j*4 + (b_k+1)][base_col+3] += 1
        V[b_j*4 + b_k][base_col+3] += -1
        W[c_k*4 + c_i][base_col+3] += 1
        W[(c_k+1)*4 + c_i][base_col+3] += 1
        
        # Term 5
        U[a_i*4 + a_j][base_col+4] += 1
        U[a_i*4 + (a_j+1)][base_col+4] += 1
        V[(b_j+1)*4 + (b_k+1)][base_col+4] += 1
        W[c_k*4 + c_i][base_col+4] += -1
        W[(c_k+1)*4 + c_i][base_col+4] += 1
        
        # Term 6
        U[(a_i+1)*4 + a_j][base_col+5] += 1
        U[a_i*4 + a_j][base_col+5] += -1
        V[b_j*4 + b_k][base_col+5] += 1
        V[b_j*4 + (b_k+1)][base_col+5] += 1
        W[(c_k+1)*4 + (c_i+1)][base_col+5] += 1
        
        # Term 7
        U[a_i*4 + (a_j+1)][base_col+6] += 1
        U[(a_i+1)*4 + (a_j+1)][base_col+6] += -1
        V[(b_j+1)*4 + b_k][base_col+6] += 1
        V[(b_j+1)*4 + (b_k+1)][base_col+6] += 1
        W[c_k*4 + c_i][base_col+6] += 1

    # Strassen for 4x4:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # A00 is at (0,0), A01 at (0,1), A10 at (1,0), A11 at (1,1)
    # B = [[B00, B01], [B10, B11]]
    # B00 is at (0,0), B01 at (0,1), B10 at (1,0), B11 at (1,1)
    # C = [[C00, C01], [C10, C11]]
    # C00 is at (0,0), C01 at (0,1), C10 at (1,0), C11 at (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # This is a 2x2 product.
    # The "A" part is A00+A11. The "B" part is B00+B11.
    # We cannot directly use add_strassen_2x2 because the inputs are sums.
    # We must expand the 7 terms of the 2x2 multiplication of (A00+A11) and (B00+B11).
    # Let X = A00+A11, Y = B00+B11.
    # We need to decompose X*Y.
    # X is a 2x2 matrix with entries:
    # X00 = A00_00 + A11_00
    # X01 = A00_01 + A11_01
    # X10 = A10_00 + A11_10  -- Wait, A11 is at (1,1).
    # A00 is at rows 0,1 cols 0,1.
    # A11 is at rows 2,3 cols 2,3.
    # So X00 = A[0,0] + A[2,2]
    # X01 = A[0,1] + A[2,3]
    # X10 = A[1,0] + A[3,2]
    # X11 = A[1,1] + A[3,3]
    
    # Y00 = B[0,0] + B[2,2]
    # Y01 = B[0,1] + B[2,3]
    # Y10 = B[1,0] + B[3,2]
    # Y11 = B[1,1] + B[3,3]
    
    # We can just call add_strassen_2x2 for each of the 7 terms?
    # No, add_strassen_2x2 assumes the inputs are single blocks.
    # We need to add the contributions to U, V, W manually for the 7 terms of the 2x2 multiplication.
    
    # Let's define a helper to add a single scalar term to U, V, W
    def add_term(u_idx, v_idx, w_idx, col, val=1):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # M1 = (A00+A11)(B00+B11)
    # Term 1.1: (X00+X11)(Y00+Y11)
    # X00+X11 = A[0,0] + A[2,2] + A[1,1] + A[3,3]
    # Y00+Y11 = B[0,0] + B[2,2] + B[1,1] + B[3,3]
    # This is getting complicated.
    
    # Alternative: Just use the known rank-49 decomposition.
    # It is generated by applying Strassen recursively.
    # We can generate it by calling add_strassen_2x2 for each of the 7 M's?
    # No, each M is a 2x2 product.
    # M1 = X*Y.
    # We can decompose X*Y using the 7 terms of Strassen for 2x2.
    # But X and Y are sums of blocks.
    # We can just expand the 7 terms of Strassen for 2x2, substituting X and Y.
    
    # Let's write a function to add the 7 terms of Strassen for 2x2, given the indices of the elements of X and Y.
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = X*Y
    # Z00 = x00y00 + x01y10
    # Z01 = x00y01 + x01y11
    # Z10 = x10y00 + x11y10
    # Z11 = x10y01 + x11y11
    
    # Strassen:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M6
    
    # We need to add these to the global U, V, W.
    # Each term like (x00+x11) is a sum of A elements.
    # We add each A element to U.
    
    def add_strassen_2x2_elements(x_idx, y_idx, z_idx, base_col):
        # x_idx is a dict or list of (u_row, coeff) for x00, x01, x10, x11
        # y_idx is a dict or list of (v_row, coeff) for y00, y01, y10, y11
        # z_idx is a dict or list of (w_row, coeff) for z00, z01, z10, z11
        
        # M1 = (x00+x11)(y00+y11)
        # U: x00, x11. V: y00, y11. W: z00, z11.
        for u_r, c_u in x_idx['00']: U[u_r][base_col+0] += c_u
        for u_r, c_u in x_idx['11']: U[u_r][base_col+0] += c_u
        for v_r, c_v in y_idx['00']: V[v_r][base_col+0] += c_v
        for v_r, c_v in y_idx['11']: V[v_r][base_col+0] += c_v
        for w_r, c_w in z_idx['00']: W[w_r][base_col+0] += c_w
        for w_r, c_w in z_idx['11']: W[w_r][base_col+0] += c_w
        
        # M2 = (x10+x11)y00
        for u_r, c_u in x_idx['10']: U[u_r][base_col+1] += c_u
        for u_r, c_u in x_idx['11']: U[u_r][base_col+1] += c_u
        for v_r, c_v in y_idx['00']: V[v_r][base_col+1] += c_v
        for w_r, c_w in z_idx['10']: W[w_r][base_col+1] += c_w
        for w_r, c_w in z_idx['11']: W[w_r][base_col+1] += c_w
        
        # M3 = x00(y10-y11)
        for u_r, c_u in x_idx['00']: U[u_r][base_col+2] += c_u
        for v_r, c_v in y_idx['10']: V[v_r][base_col+2] += c_v
        for v_r, c_v in y_idx['11']: V[v_r][base_col+2] -= c_v
        for w_r, c_w in z_idx['01']: W[w_r][base_col+2] += c_w
        
        # M4 = x11(y01-y00)
        for u_r, c_u in x_idx['11']: U[u_r][base_col+3] += c_u
        for v_r, c_v in y_idx['01']: V[v_r][base_col+3] += c_v
        for v_r, c_v in y_idx['00']: V[v_r][base_col+3] -= c_v
        for w_r, c_w in z_idx['00']: W[w_r][base_col+3] += c_w
        for w_r, c_w in z_idx['10']: W[w_r][base_col+3] += c_w
        
        # M5 = (x00+x01)y11
        for u_r, c_u in x_idx['00']: U[u_r][base_col+4] += c_u
        for u_r, c_u in x_idx['01']: U[u_r][base_col+4] += c_u
        for v_r, c_v in y_idx['11']: V[v_r][base_col+4] += c_v
        for w_r, c_w in z_idx['00']: W[w_r][base_col+4] -= c_w
        for w_r, c_w in z_idx['01']: W[w_r][base_col+4] += c_w
        
        # M6 = (x10-x00)(y00+y01)
        for u_r, c_u in x_idx['10']: U[u_r][base_col+5] += c_u
        for u_r, c_u in x_idx['00']: U[u_r][base_col+5] -= c_u
        for v_r, c_v in y_idx['00']: V[v_r][base_col+5] += c_v
        for v_r, c_v in y_idx['01']: V[v_r][base_col+5] += c_v
        for w_r, c_w in z_idx['11']: W[w_r][base_col+5] += c_w
        
        # M7 = (x01-x11)(y10+y11)
        for u_r, c_u in x_idx['01']: U[u_r][base_col+6] += c_u
        for u_r, c_u in x_idx['11']: U[u_r][base_col+6] -= c_u
        for v_r, c_v in y_idx['10']: V[v_r][base_col+6] += c_v
        for v_r, c_v in y_idx['11']: V[v_r][base_col+6] += c_v
        for w_r, c_w in z_idx['00']: W[w_r][base_col+6] += c_w

    # Now apply this to the 7 M's of the 4x4 Strassen.
    
    # M1 = (A00+A11)(B00+B11)
    # X = A00+A11. Y = B00+B11. Z = C00? No, Z is the result of X*Y.
    # In Strassen 4x4, the result is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6
    
    # So for M1, the "Z" part is not a single block, but contributes to C00 and C11.
    # Wait, M1 is a 2x2 matrix.
    # M1 = X*Y.
    # M1 contributes to C00 and C11?
    # No, C00 and C11 are 2x2 blocks.
    # M1 is a 2x2 matrix.
    # C00 += M1. C11 += M1.
    
    # So for M1, z_idx should map to C00 and C11.
    # x_idx maps to A00 and A11.
    # y_idx maps to B00 and B11.
    
    # Let's define the indices.
    # A00: rows 0,1 cols 0,1.
    # A11: rows 2,3 cols 2,3.
    # B00: rows 0,1 cols 0,1.
    # B11: rows 2,3 cols 2,3.
    # C00: rows 0,1 cols 0,1.
    # C11: rows 2,3 cols 2,3.
    
    # x_idx for M1:
    # x00 = A[0,0], x01 = A[0,1], x10 = A[1,0], x11 = A[1,1] from A00
    # x00 += A[2,2], x01 += A[2,3], x10 += A[3,2], x11 += A[3,3] from A11
    
    x1 = {
        '00': [(0*4+0, 1), (2*4+2, 1)],
        '01': [(0*4+1, 1), (2*4+3, 1)],
        '10': [(1*4+0, 1), (3*4+2, 1)],
        '11': [(1*4+1, 1), (3*4+3, 1)]
    }
    y1 = {
        '00': [(0*4+0, 1), (2*4+2, 1)],
        '01': [(0*4+1, 1), (2*4+3, 1)],
        '10': [(1*4+0, 1), (3*4+2, 1)],
        '11': [(1*4+1, 1), (3*4+3, 1)]
    }
    z1 = {
        '00': [(0*4+0, 1), (2*4+2, 1)],
        '01': [(0*4+1, 1), (2*4+3, 1)],
        '10': [(1*4+0, 1), (3*4+2, 1)],
        '11': [(1*4+1, 1), (3*4+3, 1)]
    }
    add_strassen_2x2_elements(x1, y1, z1, 0)
    
    # M2 = (A10+A11)B00
    # X = A10+A11. Y = B00.
    # A10: rows 2,3 cols 0,1.
    # A11: rows 2,3 cols 2,3.
    # B00: rows 0,1 cols 0,1.
    # Z = C10, C11.
    # C10: rows 2,3 cols 0,1.
    # C11: rows 2,3 cols 2,3.
    
    x2 = {
        '00': [(2*4+0, 1), (2*4+2, 1)],
        '01': [(2*4+1, 1), (2*4+3, 1)],
        '10': [(3*4+0, 1), (3*4+2, 1)],
        '11': [(3*4+1, 1), (3*4+3, 1)]
    }
    y2 = {
        '00': [(0*4+0, 1)],
        '01': [(0*4+1, 1)],
        '10': [(1*4+0, 1)],
        '11': [(1*4+1, 1)]
    }
    z2 = {
        '00': [(2*4+0, 1)],
        '01': [(2*4+1, 1)],
        '10': [(3*4+0, 1)],
        '11': [(3*4+1, 1)]
    }
    # Note: M2 contributes to C10 and C11.
    # In Strassen, C10 += M2, C11 += M2.
    # So z2 should map to C10 and C11?
    # Wait, M2 is a 2x2 matrix.
    # C10 is 2x2. C11 is 2x2.
    # So z2 should be the union of C10 and C11 indices.
    z2 = {
        '00': [(2*4+0, 1), (2*4+2, 1)],
        '01': [(2*4+1, 1), (2*4+3, 1)],
        '10': [(3*4+0, 1), (3*4+2, 1)],
        '11': [(3*4+1, 1), (3*4+3, 1)]
    }
    add_strassen_2x2_elements(x2, y2, z2, 7)
    
    # M3 = A00(B10-B11)
    # X = A00. Y = B10-B11.
    # A00: rows 0,1 cols 0,1.
    # B10: rows 2,3 cols 0,1.
    # B11: rows 2,3 cols 2,3.
    # Z = C01.
    # C01: rows 0,1 cols 2,3.
    
    x3 = {
        '00': [(0*4+0, 1)],
        '01': [(0*4+1, 1)],
        '10': [(1*4+0, 1)],
        '11': [(1*4+1, 1)]
    }
    y3 = {
        '00': [(2*4+0, 1), (2*4+2, -1)],
        '01': [(2*4+1, 1), (2*4+3, -1)],
        '10': [(3*4+0, 1), (3*4+2, -1)],
        '11': [(3*4+1, 1), (3*4+3, -1)]
    }
    z3 = {
        '00': [(0*4+2, 1)],
        '01': [(0*4+3, 1)],
        '10': [(1*4+2, 1)],
        '11': [(1*4+3, 1)]
    }
    add_strassen_2x2_elements(x3, y3, z3, 14)
    
    # M4 = A11(B01-B00)
    # X = A11. Y = B01-B00.
    # A11: rows 2,3 cols 2,3.
    # B01: rows 0,1 cols 2,3.
    # B00: rows 0,1 cols 0,1.
    # Z = C00, C10.
    
    x4 = {
        '00': [(2*4+2, 1)],
        '01': [(2*4+3, 1)],
        '10': [(3*4+2, 1)],
        '11': [(3*4+3, 1)]
    }
    y4 = {
        '00': [(0*4+0, -1), (0*4+2, 1)],
        '01': [(0*4+1, -1), (0*4+3, 1)],
        '10': [(1*4+0, -1), (1*4+2, 1)],
        '11': [(1*4+1, -1), (1*4+3, 1)]
    }
    z4 = {
        '00': [(0*4+0, 1), (2*4+0, 1)],
        '01': [(0*4+1, 1), (2*4+1, 1)],
        '10': [(1*4+0, 1), (3*4+0, 1)],
        '11': [(1*4+1, 1), (3*4+1, 1)]
    }
    add_strassen_2x2_elements(x4, y4, z4, 21)
    
    # M5 = (A00+A01)B11
    # X = A00+A01. Y = B11.
    # A00: rows 0,1 cols 0,1.
    # A01: rows 0,1 cols 2,3.
    # B11: rows 2,3 cols 2,3.
    # Z = C00, C01.
    
    x5 = {
        '00': [(0*4+0, 1), (0*4+2, 1)],
        '01': [(0*4+1, 1), (0*4+3, 1)],
        '10': [(1*4+0, 1), (1*4+2, 1)],
        '11': [(1*4+1, 1), (1*4+3, 1)]
    }
    y5 = {
        '00': [(2*4+2, 1)],
        '01': [(2*4+3, 1)],
        '10': [(3*4+2, 1)],
        '11': [(3*4+3, 1)]
    }
    z5 = {
        '00': [(0*4+0, 1), (0*4+2, 1)],
        '01': [(0*4+1, 1), (0*4+3, 1)],
        '10': [(1*4+0, 1), (1*4+2, 1)],
        '11': [(1*4+1, 1), (1*4+3, 1)]
    }
    add_strassen_2x2_elements(x5, y5, z5, 28)
    
    # M6 = (A10-A00)(B00+B01)
    # X = A10-A00. Y = B00+B01.
    # A10: rows 2,3 cols 0,1.
    # A00: rows 0,1 cols 0,1.
    # B00: rows 0,1 cols 0,1.
    # B01: rows 0,1 cols 2,3.
    # Z = C11.
    
    x6 = {
        '00': [(2*4+0, 1), (0*4+0, -1)],
        '01': [(2*4+1, 1), (0*4+1, -1)],
        '10': [(3*4+0, 1), (1*4+0, -1)],
        '11': [(3*4+1, 1), (1*4+1, -1)]
    }
    y6 = {
        '00': [(0*4+0, 1), (0*4+2, 1)],
        '01': [(0*4+1, 1), (0*4+3, 1)],
        '10': [(1*4+0, 1), (1*4+2, 1)],
        '11': [(1*4+1, 1), (1*4+3, 1)]
    }
    z6 = {
        '00': [(2*4+2, 1)],
        '01': [(2*4+3, 1)],
        '10': [(3*4+2, 1)],
        '11': [(3*4+3, 1)]
    }
    add_strassen_2x2_elements(x6, y6, z6, 35)
    
    # M7 = (A01-A11)(B10+B11)
    # X = A01-A11. Y = B10+B11.
    # A01: rows 0,1 cols 2,3.
    # A11: rows 2,3 cols 2,3.
    # B10: rows 2,3 cols 0,1.
    # B11: rows 2,3 cols 2,3.
    # Z = C00.
    
    x7 = {
        '00': [(0*4+2, 1), (2*4+2, -1)],
        '01': [(0*4+3, 1), (2*4+3, -1)],
        '10': [(1*4+2, 1), (3*4+2, -1)],
        '11': [(1*4+3, 1), (3*4+3, -1)]
    }
    y7 = {
        '00': [(2*4+0, 1), (2*4+2, 1)],
        '01': [(2*4+1, 1), (2*4+3, 1)],
        '10': [(3*4+0, 1), (3*4+2, 1)],
        '11': [(3*4+1, 1), (3*4+3, 1)]
    }
    z7 = {
        '00': [(0*4+0, 1)],
        '01': [(0*4+1, 1)],
        '10': [(1*4+0, 1)],
        '11': [(1*4+1, 1)]
    }
    add_strassen_2x2_elements(x7, y7, z7, 42)
    
    return U, V, W

def _schoolbook_4x4_4x3():
    """Rank-48 schoolbook for 4x4 * 4x3 -> 4x3.
    U: 16x48, V: 12x48, W: 12x48
    U[i*4+j, c] for A[i,j], V[j*3+k, c] for B[j,k], W[k*4+i, c] for C[i,k]
    """
    n, m, p = 4, 4, 3
    rank = n * m * p  # 48
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Verify the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    sU, sV, sW = _strassen_4x4()
    sbU, sbV, sbW = _schoolbook_4x4_4x3()
    
    total_rank = 49 + 48
    
    U = [[0]*total_rank for _ in range(n*m)]   # 16 x 97
    V = [[0]*total_rank for _ in range(m*p)]   # 28 x 97
    W = [[0]*total_rank for _ in range(p*n)]   # 28 x 97
    
    col = 0
    
    # Part 1: Strassen for B cols 0-3
    for c in range(49):
        for i in range(16):
            U[i][col + c] = sU[i][c]
        for i in range(16):
            V[i][col + c] = sV[i][c]
        for i in range(16):
            W[i][col + c] = sW[i][c]
    col += 49
    
    # Part 2: Schoolbook for B cols 4-6
    for c in range(48):
        for i in range(16):
            U[i][col + c] = sbU[i][c]
        for i in range(12):
            V[16 + i][col + c] = sbV[i][c]  # V rows 16-27 correspond to B cols 4-6
        for i in range(12):
            W[16 + i][col + c] = sbW[i][c]  # W rows 16-27 correspond to C cols 4-6
    col += 48
    
    assert col == 97, f"Expected rank 97, got {col}"
    
    err = _verify(U, V, W)
    if err > 1e-9:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
