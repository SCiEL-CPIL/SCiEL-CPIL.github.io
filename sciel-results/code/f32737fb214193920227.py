import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Part 1: 2x2x6 using 3 Strassen blocks
    # A is 2x2 (cols 0,1 of original A)
    # B is 2x6 (rows 0,1 of original B)
    # C is 2x6
    
    # Strassen formulas for 2x2x2
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-f)
    # M5: (a+b)(h+g)
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3 in U (row-major i*m+j)
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 in V (row-major j*p+k) for 2x2 block
    # But original B is 3x6.
    # Block 1 of B is rows 0,1, cols 0,1.
    # V index j*p+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 6
    # j=1, k=1 -> 7
    
    # C indices: 00, 01, 10, 11 -> W index k*n+i
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    def add_strassen_block(b_col_offset):
        nonlocal col
        # b_col_offset is 0, 2, 4
        # B block cols are b_col_offset, b_col_offset+1
        
        # M1: (a+d)(e+h)
        U[0, col] = 1 # a
        U[3, col] = 1 # d
        V[0, col] = 1 # e (j=0, k=0)
        V[1+b_col_offset, col] = 1 # h (j=0, k=1) -- Wait, V index is j*p+k.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 6
        # j=1, k=1 -> 7
        # So e is V[0], h is V[1], g is V[6], f is V[7].
        # For block offset:
        # e: j=0, k=offset -> offset
        # h: j=0, k=offset+1 -> offset+1
        # f: j=1, k=offset -> 6+offset
        # g: j=1, k=offset+1 -> 7+offset
        
        V[0, col] = 1 # e
        V[1+b_col_offset, col] = 1 # h? No, V index is j*p+k.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 6
        # j=1, k=1 -> 7
        # So for offset 0: e=0, h=1, f=6, g=7.
        # For offset 2: e=2, h=3, f=8, g=9.
        
        V[0+b_col_offset, col] = 1 # e
        V[1+b_col_offset, col] = 1 # h
        W[0, col] = 1 # C11 part
        W[1, col] = -1 # C22 part? No, C22 = M1 - M2 + M3 + M6.
        # Let's map W carefully.
        # C11 = M1 + M4 - M5 + M7 -> W[0]
        # C12 = M3 + M5 -> W[2]
        # C21 = M2 + M4 -> W[1]
        # C22 = M1 - M2 + M3 + M6 -> W[3]
        
        W[0, col] = 1
        W[1, col] = 0
        W[2, col] = 0
        W[3, col] = 1
        col += 1
        
        # M2: (c+d)e
        U[2, col] = 1 # c
        U[3, col] = 1 # d
        V[0+b_col_offset, col] = 1 # e
        W[1, col] = 1 # C21
        W[3, col] = -1 # C22
        col += 1
        
        # M3: a(f-h)
        U[0, col] = 1 # a
        V[6+b_col_offset, col] = 1 # f
        V[1+b_col_offset, col] = -1 # h
        W[2, col] = 1 # C12
        W[3, col] = 1 # C22
        col += 1
        
        # M4: d(g-f)
        U[3, col] = 1 # d
        V[7+b_col_offset, col] = 1 # g
        V[6+b_col_offset, col] = -1 # f
        W[0, col] = 1 # C11
        W[1, col] = 1 # C21
        col += 1
        
        # M5: (a+b)(h+g)
        U[0, col] = 1 # a
        U[1, col] = 1 # b
        V[1+b_col_offset, col] = 1 # h
        V[7+b_col_offset, col] = 1 # g
        W[0, col] = -1 # C11
        W[2, col] = 1 # C12
        col += 1
        
        # M6: (c-a)(e+f)
        U[2, col] = 1 # c
        U[0, col] = -1 # a
        V[0+b_col_offset, col] = 1 # e
        V[6+b_col_offset, col] = 1 # f
        W[3, col] = 1 # C22
        col += 1
        
        # M7: (b-d)(g+h)
        U[1, col] = 1 # b
        U[3, col] = -1 # d
        V[7+b_col_offset, col] = 1 # g
        V[1+b_col_offset, col] = 1 # h
        W[0, col] = 1 # C11
        col += 1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Part 2: 2x1x6 (Schoolbook)
    # A is 2x1 (col 2 of original A) -> U indices 4, 5
    # B is 1x6 (row 2 of original B) -> V indices 12..17
    # C is 2x6
    
    for k in range(6):
        # C_{0,k} = a_{0,2} b_{2,k}
        U[4, col] = 1
        V[12+k, col] = 1
        W[k*2 + 0, col] = 1
        col += 1
        
        # C_{1,k} = a_{1,2} b_{2,k}
        U[5, col] = 1
        V[12+k, col] = 1
        W[k*2 + 1, col] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
