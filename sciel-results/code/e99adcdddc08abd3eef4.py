import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    Uses Strassen for 2x2 sub-blocks and schoolbook for the rest.
    Rank expected: 117.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # We will build the matrices dynamically
    U = []
    V = []
    W = []
    
    def add_column(u_vals, v_vals, w_vals):
        U.append(u_vals)
        V.append(v_vals)
        W.append(w_vals)
        
    def add_schoolbook_block(i_s, i_e, j_s, j_e, k_s, k_e):
        """
        Adds schoolbook terms for A[i_s:i_e, j_s:j_e] * B[j_s:j_e, k_s:k_e]
        Note: In our convention, A is n x m, B is m x p.
        The product is C[i, k].
        The sum is over j.
        So we iterate i in [i_s, i_e), j in [j_s, j_e), k in [k_s, k_e).
        """
        for i in range(i_s, i_e):
            for j in range(j_s, j_e):
                for k in range(k_s, k_e):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    add_column(u, v, w)
                    
    def add_strassen_block(i_s, j_s, k_s):
        """
        Adds Strassen terms for A[i_s:i_s+2, j_s:j_s+2] * B[j_s:j_s+2, k_s:k_s+2]
        i_s, j_s, k_s are start indices.
        """
        # Local indices 0, 1
        # Global indices:
        # A: (i_s, j_s), (i_s, j_s+1), (i_s+1, j_s), (i_s+1, j_s+1)
        # B: (j_s, k_s), (j_s, k_s+1), (j_s+1, k_s), (j_s+1, k_s+1)
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        # A00
        u1[i_s*m + j_s] += 1
        # A11
        u1[(i_s+1)*m + (j_s+1)] += 1
        # B00
        v1[j_s*p + k_s] += 1
        # B11
        v1[(j_s+1)*p + (k_s+1)] += 1
        # W contributions for M1:
        # C00 += M1 -> W[k_s*n + i_s] += 1
        # C11 += M1 -> W[(k_s+1)*n + (i_s+1)] += 1
        w1[k_s*n + i_s] += 1
        w1[(k_s+1)*n + (i_s+1)] += 1
        add_column(u1, v1, w1)
        
        # M2 = (A10 + A11)B00
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[(i_s+1)*m + j_s] += 1
        u2[(i_s+1)*m + (j_s+1)] += 1
        v2[j_s*p + k_s] += 1
        # C10 += M2 -> W[k_s*n + (i_s+1)] += 1
        # C11 -= M2 -> W[(k_s+1)*n + (i_s+1)] -= 1
        w2[k_s*n + (i_s+1)] += 1
        w2[(k_s+1)*n + (i_s+1)] -= 1
        add_column(u2, v2, w2)
        
        # M3 = A00(B10 - B11)
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[i_s*m + j_s] += 1
        v3[(j_s+1)*p + k_s] += 1
        v3[(j_s+1)*p + (k_s+1)] -= 1
        # C01 += M3 -> W[(k_s+1)*n + i_s] += 1
        # C11 += M3 -> W[(k_s+1)*n + (i_s+1)] += 1
        w3[(k_s+1)*n + i_s] += 1
        w3[(k_s+1)*n + (i_s+1)] += 1
        add_column(u3, v3, w3)
        
        # M4 = A11(B01 - B00)
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[(i_s+1)*m + (j_s+1)] += 1
        v4[j_s*p + (k_s+1)] += 1
        v4[j_s*p + k_s] -= 1
        # C00 += M4 -> W[k_s*n + i_s] += 1
        # C10 += M4 -> W[k_s*n + (i_s+1)] += 1
        w4[k_s*n + i_s] += 1
        w4[k_s*n + (i_s+1)] += 1
        add_column(u4, v4, w4)
        
        # M5 = (A00 + A01)B11
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[i_s*m + j_s] += 1
        u5[i_s*m + (j_s+1)] += 1
        v5[(j_s+1)*p + (k_s+1)] += 1
        # C00 -= M5 -> W[k_s*n + i_s] -= 1
        # C01 += M5 -> W[(k_s+1)*n + i_s] += 1
        w5[k_s*n + i_s] -= 1
        w5[(k_s+1)*n + i_s] += 1
        add_column(u5, v5, w5)
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[(i_s+1)*m + j_s] += 1
        u6[i_s*m + j_s] -= 1
        v6[j_s*p + k_s] += 1
        v6[j_s*p + (k_s+1)] += 1
        # C11 += M6 -> W[(k_s+1)*n + (i_s+1)] += 1
        w6[(k_s+1)*n + (i_s+1)] += 1
        add_column(u6, v6, w6)
        
        # M7 = (A01 - A11)(B10 + B11)
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[i_s*m + (j_s+1)] += 1
        u7[(i_s+1)*m + (j_s+1)] -= 1
        v7[(j_s+1)*p + k_s] += 1
        v7[(j_s+1)*p + (k_s+1)] += 1
        # C00 += M7 -> W[k_s*n + i_s] += 1
        w7[k_s*n + i_s] += 1
        add_column(u7, v7, w7)
        
    # Partition 5 = 2 + 2 + 1
    # Blocks:
    # 1: rows 0-1, cols 0-1 (Strassen)
    # 2: rows 0-1, cols 2-3 (Strassen)
    # 3: rows 0-1, cols 4 (Schoolbook)
    # 4: rows 2-3, cols 0-1 (Strassen)
    # 5: rows 2-3, cols 2-3 (Strassen)
    # 6: rows 2-3, cols 4 (Schoolbook)
    # 7: rows 4, cols 0-1 (Schoolbook)
    # 8: rows 4, cols 2-3 (Schoolbook)
    # 9: rows 4, cols 4 (Schoolbook)
    
    # C11: A11*B11 + A12*B21 + A13*B31
    add_strassen_block(0, 0, 0) # A11 B11
    add_strassen_block(0, 2, 2) # A12 B21
    add_schoolbook_block(0, 2, 4, 5, 0, 2) # A13 B31
    
    # C12: A11*B12 + A12*B22 + A13*B32
    add_strassen_block(0, 0, 2) # A11 B12
    add_strassen_block(0, 2, 2) # A12 B22
    add_schoolbook_block(0, 2, 4, 5, 2, 4) # A13 B32
    
    # C13: A11*B13 + A12*B23 + A13*B33
    add_schoolbook_block(0, 2, 0, 2, 4, 5) # A11 B13
    add_schoolbook_block(0, 2, 2, 4, 4, 5) # A12 B23
    add_schoolbook_block(0, 2, 4, 5, 4, 5) # A13 B33
    
    # C21: A21*B11 + A22*B21 + A23*B31
    add_strassen_block(2, 0, 0) # A21 B11
    add_strassen_block(2, 2, 2) # A22 B21
    add_schoolbook_block(2, 4, 4, 5, 0, 2) # A23 B31
    
    # C22: A21*B12 + A22*B22 + A23*B32
    add_strassen_block(2, 0, 2) # A21 B12
    add_strassen_block(2, 2, 2) # A22 B22
    add_schoolbook_block(2, 4, 4, 5, 2, 4) # A23 B32
    
    # C23: A21*B13 + A22*B23 + A23*B33
    add_schoolbook_block(2, 4, 0, 2, 4, 5) # A21 B13
    add_schoolbook_block(2, 4, 2, 4, 4, 5) # A22 B23
    add_schoolbook_block(2, 4, 4, 5, 4, 5) # A23 B33
    
    # C31: A31*B11 + A32*B21 + A33*B31
    add_schoolbook_block(4, 5, 0, 2, 0, 2) # A31 B11
    add_schoolbook_block(4, 5, 2, 4, 0, 2) # A32 B21
    add_schoolbook_block(4, 5, 4, 5, 0, 2) # A33 B31
    
    # C32: A31*B12 + A32*B22 + A33*B32
    add_schoolbook_block(4, 5, 0, 2, 2, 4) # A31 B12
    add_schoolbook_block(4, 5, 2, 4, 2, 4) # A32 B22
    add_schoolbook_block(4, 5, 4, 5, 2, 4) # A33 B32
    
    # C33: A31*B13 + A32*B23 + A33*B33
    add_schoolbook_block(4, 5, 0, 2, 4, 5) # A31 B13
    add_schoolbook_block(4, 5, 2, 4, 4, 5) # A32 B23
    add_schoolbook_block(4, 5, 4, 5, 4, 5) # A33 B33
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
