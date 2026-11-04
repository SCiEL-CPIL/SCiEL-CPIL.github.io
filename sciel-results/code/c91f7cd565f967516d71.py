import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 96
    # We aim for rank 88
    
    U = [[0] * 88 for _ in range(n * m)]
    V = [[0] * 88 for _ in range(m * p)]
    W = [[0] * 88 for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Bottom row (i=2)
    # A[2, j] * B[j, k] for j in 0..3, k in 0..7
    for j in range(m):
        for k in range(p):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: Top two rows (i=0, 1)
    # A_top is 2x4. B is 4x8.
    # Split A_top into A11 (cols 0,1) and A12 (cols 2,3)
    # Split B into B1 (rows 0,1) and B2 (rows 2,3)
    # C_top = A11 B1 + A12 B2
    
    def add_strassen_block(a_indices, b_indices, c_indices, start_col):
        # a_indices: list of 4 U indices [a, b, c, d]
        # b_indices: list of 4 V indices [e, f, g, h]
        # c_indices: list of 4 W indices [c11, c12, c21, c22]
        # Returns new col index
        c = start_col
        
        # m1 = (a+d)(e+h)
        U[a_indices[0]][c] += 1
        U[a_indices[3]][c] += 1
        V[b_indices[0]][c] += 1
        V[b_indices[3]][c] += 1
        W[c_indices[0]][c] += 1
        W[c_indices[2]][c] += 1
        W[c_indices[3]][c] += 1
        W[c_indices[1]][c] += 1 # Wait, c12 gets m1? No.
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # Let's map coefficients correctly.
        # m1: C11(+1), C22(+1)
        # m2: C21(+1), C22(-1)
        # m3: C12(+1), C22(+1)
        # m4: C11(+1), C21(+1)
        # m5: C11(-1), C12(+1)
        # m6: C22(+1)
        # m7: C11(+1)
        
        # Reset and do it properly
        # m1
        U[a_indices[0]][c] = 1
        U[a_indices[3]][c] = 1
        V[b_indices[0]][c] = 1
        V[b_indices[3]][c] = 1
        W[c_indices[0]][c] = 1 # C11
        W[c_indices[3]][c] = 1 # C22
        c += 1
        
        # m2 = (c+d)e
        U[a_indices[2]][c] = 1
        U[a_indices[3]][c] = 1
        V[b_indices[0]][c] = 1
        W[c_indices[2]][c] = 1 # C21
        W[c_indices[3]][c] = -1 # C22
        c += 1
        
        # m3 = a(f-h)
        U[a_indices[0]][c] = 1
        V[b_indices[1]][c] = 1
        V[b_indices[3]][c] = -1
        W[c_indices[1]][c] = 1 # C12
        W[c_indices[3]][c] = 1 # C22
        c += 1
        
        # m4 = d(g-e)
        U[a_indices[3]][c] = 1
        V[b_indices[2]][c] = 1
        V[b_indices[0]][c] = -1
        W[c_indices[0]][c] = 1 # C11
        W[c_indices[2]][c] = 1 # C21
        c += 1
        
        # m5 = (a+b)h
        U[a_indices[0]][c] = 1
        U[a_indices[1]][c] = 1
        V[b_indices[3]][c] = 1
        W[c_indices[0]][c] = -1 # C11
        W[c_indices[1]][c] = 1 # C12
        c += 1
        
        # m6 = (c-a)(e+f)
        U[a_indices[2]][c] = 1
        U[a_indices[0]][c] = -1
        V[b_indices[0]][c] = 1
        V[b_indices[1]][c] = 1
        W[c_indices[3]][c] = 1 # C22
        c += 1
        
        # m7 = (b-d)(g+h)
        U[a_indices[1]][c] = 1
        U[a_indices[3]][c] = -1
        V[b_indices[2]][c] = 1
        V[b_indices[3]][c] = 1
        W[c_indices[0]][c] = 1 # C11
        c += 1
        
        return c

    # A11: U indices 0, 1, 4, 5
    a11 = [0, 1, 4, 5]
    # B1 blocks (rows 0,1)
    # B11: V indices 0, 1, 8, 9
    b11 = [0, 1, 8, 9]
    # B12: V indices 2, 3, 10, 11
    b12 = [2, 3, 10, 11]
    # B13: V indices 4, 5, 12, 13
    b13 = [4, 5, 12, 13]
    # B14: V indices 6, 7, 14, 15
    b14 = [6, 7, 14, 15]
    
    # C_top indices: W 0, 3, 1, 4
    c_top = [0, 3, 1, 4]
    
    col = add_strassen_block(a11, b11, c_top, col)
    col = add_strassen_block(a11, b12, c_top, col)
    col = add_strassen_block(a11, b13, c_top, col)
    col = add_strassen_block(a11, b14, c_top, col)
    
    # A12: U indices 2, 3, 6, 7
    a12 = [2, 3, 6, 7]
    # B2 blocks (rows 2,3)
    # B21: V indices 16, 17, 24, 25
    b21 = [16, 17, 24, 25]
    # B22: V indices 18, 19, 26, 27
    b22 = [18, 19, 26, 27]
    # B23: V indices 20, 21, 28, 29
    b23 = [20, 21, 28, 29]
    # B24: V indices 22, 23, 30, 31
    b24 = [22, 23, 30, 31]
    
    col = add_strassen_block(a12, b21, c_top, col)
    col = add_strassen_block(a12, b22, c_top, col)
    col = add_strassen_block(a12, b23, c_top, col)
    col = add_strassen_block(a12, b24, c_top, col)
    
    return {"U": U, "V": V, "W": W}
