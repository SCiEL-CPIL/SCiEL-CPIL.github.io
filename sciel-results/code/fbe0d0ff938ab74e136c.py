"""Matrix multiplication 2x5x6 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # We will build U, V, W lists
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        # u_idx, v_idx, w_idx are integers
        # We store them as tuples or just append to lists
        # The final output is a list of lists (matrices)
        # We need to accumulate columns.
        pass

    # Better approach:
    # Keep a list of (u_idx, coeff_u, v_idx, coeff_v, w_idx, coeff_w)
    # Then convert to matrices.
    terms = []
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        # a_r, a_c: start row, col of A block (2x2)
        # b_r, b_c: start row, col of B block (2x2)
        # c_r, c_c: start row, col of C block (2x2)
        
        # A indices:
        # A00 = (a_r, a_c)
        # A01 = (a_r, a_c+1)
        # A10 = (a_r+1, a_c)
        # A11 = (a_r+1, a_c+1)
        
        # B indices:
        # B00 = (b_r, b_c)
        # B01 = (b_r, b_c+1)
        # B10 = (b_r+1, b_c)
        # B11 = (b_r+1, b_c+1)
        
        # C indices:
        # C00 = (c_r, c_c)
        # C01 = (c_r, c_c+1)
        # C10 = (c_r+1, c_c)
        # C11 = (c_r+1, c_c+1)
        
        # M1 = (A00 + A11)(B00 + B11)
        terms.append((a_r*m + a_c, 1, b_r*p + b_c, 1, c_c*n + c_r, 1)) # C00 part
        terms.append((a_r*m + a_c, 1, b_r*p + b_c, 1, c_c*n + c_r+1, 1)) # C10 part? No.
        
        # Wait, the W matrix index is k*n + i.
        # C[i, k] corresponds to W[k*n + i].
        # So for C00 (i=0, k=0 relative to block), index is (c_c)*n + (c_r).
        # For C01 (i=0, k=1), index is (c_c+1)*n + (c_r).
        # For C10 (i=1, k=0), index is (c_c)*n + (c_r+1).
        # For C11 (i=1, k=1), index is (c_c+1)*n + (c_r+1).
        
        # Let's define helper to add a product term
        def add_prod(a_idx, a_coeff, b_idx, b_coeff, c_idx, c_coeff):
             terms.append((a_idx, a_coeff, b_idx, b_coeff, c_idx, c_coeff))
             
        # M1
        add_prod(a_r*m + a_c, 1, b_r*p + b_c, 1, c_c*n + c_r, 1)
        add_prod(a_r*m + a_c, 1, b_r*p + b_c+1, 1, c_c*n + c_r, 1) # Wait, M1 contributes to C00 and C11?
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00 + A11)(B00 + B11)
        # This expands to 4 terms: A00B00, A00B11, A11B00, A11B11
        # We can't just add "M1" as a single term in U, V, W if U, V, W are linear.
        # U, V, W are linear combinations of inputs.
        # The term in the tensor is $u_{a} v_{b} w_{c}$.
        # If $u_{a} = A_{00} + A_{11}$, then $U$ row has 1s at A00 and A11.
        # So yes, we can define the rows of U, V, W to have multiple non-zeros.
        
        # So for M1:
        # U row: 1 at A00, 1 at A11
        # V row: 1 at B00, 1 at B11
        # W row: 1 at C00, 1 at C11
        # This represents $(A_{00}+A_{11})(B_{00}+B_{11})$ contributing to C00 and C11.
        
        # So I need to construct the U, V, W rows for each of the 7 M terms.
        
        pass

    # This logic is getting complex to manage in a single `terms` list of scalars.
    # It is better to directly build the U, V, W matrices.
    
    U = []
    V = []
    W = []
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        # Initialize rows
        # M1
        u1 = [0]*(n*m); u1[a_r*m+a_c] = 1; u1[(a_r+1)*m+(a_c+1)] = 1
        v1 = [0]*(m*p); v1[b_r*p+b_c] = 1; v1[(b_r+1)*p+(b_c+1)] = 1
        w1 = [0]*(p*n); w1[c_c*n+c_r] = 1; w1[(c_c+1)*n+(c_r+1)] = 1
        U.append(u1); V.append(v1); W.append(w1)
        
        # M2 = (A10 + A11) B00
        u2 = [0]*(n*m); u2[(a_r+1)*m+a_c] = 1; u2[(a_r+1)*m+(a_c+1)] = 1
        v2 = [0]*(m*p); v2[b_r*p+b_c] = 1
        w2 = [0]*(p*n); w2[c_c*n+(c_r+1)] = 1
        U.append(u2); V.append(v2); W.append(w2)
        
        # M3 = A00 (B10 - B11)
        u3 = [0]*(n*m); u3[a_r*m+a_c] = 1
        v3 = [0]*(m*p); v3[(b_r+1)*p+b_c] = 1; v3[(b_r+1)*p+(b_c+1)] = -1
        w3 = [0]*(p*n); w3[(c_c+1)*n+c_r] = 1
        U.append(u3); V.append(v3); W.append(w3)
        
        # M4 = A11 (B01 - B00)
        u4 = [0]*(n*m); u4[(a_r+1)*m+(a_c+1)] = 1
        v4 = [0]*(m*p); v4[b_r*p+(b_c+1)] = 1; v4[b_r*p+b_c] = -1
        w4 = [0]*(p*n); w4[c_c*n+c_r] = 1; w4[c_c*n+(c_r+1)] = 1
        U.append(u4); V.append(v4); W.append(w4)
        
        # M5 = (A00 + A01) B11
        u5 = [0]*(n*m); u5[a_r*m+a_c] = 1; u5[a_r*m+(a_c+1)] = 1
        v5 = [0]*(m*p); v5[(b_r+1)*p+(b_c+1)] = 1
        w5 = [0]*(p*n); w5[c_c*n+c_r] = -1; w5[(c_c+1)*n+c_r] = 1
        U.append(u5); V.append(v5); W.append(w5)
        
        # M6 = (A10 - A00) (B00 + B01)
        u6 = [0]*(n*m); u6[(a_r+1)*m+a_c] = 1; u6[a_r*m+a_c] = -1
        v6 = [0]*(m*p); v6[b_r*p+b_c] = 1; v6[b_r*p+(b_c+1)] = 1
        w6 = [0]*(p*n); w6[(c_c+1)*n+(c_r+1)] = 1
        U.append(u6); V.append(v6); W.append(w6)
        
        # M7 = (A01 - A11) (B10 + B11)
        u7 = [0]*(n*m); u7[a_r*m+(a_c+1)] = 1; u7[(a_r+1)*m+(a_c+1)] = -1
        v7 = [0]*(m*p); v7[(b_r+1)*p+b_c] = 1; v7[(b_r+1)*p+(b_c+1)] = 1
        w7 = [0]*(p*n); w7[c_c*n+c_r] = 1; w7[(c_c+1)*n+(c_r+1)] = 1
        U.append(u7); V.append(v7); W.append(w7)

    def add_trivial_2x1x6(a_r, a_c, b_r, b_c, c_r, c_c):
        # A is 2x1 at (a_r, a_c)
        # B is 1x6 at (b_r, b_c)
        # C is 2x6 at (c_r, c_c)
        # C[i, k] = A[i, 0] B[0, k]
        for i in range(2):
            for k in range(6):
                u = [0]*(n*m); u[(a_r+i)*m + a_c] = 1
                v = [0]*(m*p); v[(b_r)*p + (b_c+k)] = 1
                w = [0]*(p*n); w[(c_c+k)*n + (c_r+i)] = 1
                U.append(u); V.append(v); W.append(w)

    # Split A (2x5) into A1 (2x2) and A2 (2x3)
    # Split B (5x6) into B1 (2x6) and B2 (3x6)
    
    # A1 B1: A1 is 2x2 (rows 0-1, cols 0-1). B1 is 2x6 (rows 0-1, cols 0-5).
    # Split B1 into B11 (2x2, cols 0-1), B12 (2x4, cols 2-5).
    # B12 split into B121 (2x2, cols 2-3), B122 (2x2, cols 4-5).
    
    add_strassen(0, 0, 0, 0, 0, 0) # A1 * B11 -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 2, 0, 2) # A1 * B121 -> C[0:2, 2:4]
    add_strassen(0, 0, 0, 4, 0, 4) # A1 * B122 -> C[0:2, 4:6]
    
    # A2 B2: A2 is 2x3 (rows 0-1, cols 2-4). B2 is 3x6 (rows 2-4, cols 0-5).
    # Split A2 into A21 (2x1, col 2) and A22 (2x2, cols 3-4).
    # Split B2 into B21 (1x6, row 2) and B22 (2x6, rows 3-4).
    
    # A21 * B21:
    add_trivial_2x1x6(0, 2, 2, 0, 0, 0)
    
    # A22 * B22:
    # A22 is 2x2 at (0, 3). B22 is 2x6 at (3, 0).
    # Split B22 into B221 (2x2, cols 0-1), B222 (2x4, cols 2-5).
    # B222 split into B2221 (2x2, cols 2-3), B2222 (2x2, cols 4-5).
    
    add_strassen(0, 3, 3, 0, 0, 0) # A22 * B221 -> C[0:2, 0:2]
    add_strassen(0, 3, 3, 2, 0, 2) # A22 * B2221 -> C[0:2, 2:4]
    add_strassen(0, 3, 3, 4, 0, 4) # A22 * B2222 -> C[0:2, 4:6]
    
    return {"U": U, "V": V, "W": W}
