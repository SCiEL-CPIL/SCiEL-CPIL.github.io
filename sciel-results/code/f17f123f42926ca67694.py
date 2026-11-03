import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # U: 16 x rank, V: 36 x rank, W: 36 x rank
    # U index: i*m+j = i*4+j
    # V index: j*p+k = j*9+k
    # W index: k*n+i = k*4+i
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        u_dict: dict from (i,j) to coeff
        v_dict: dict from (j,k) to coeff
        w_dict: dict from (k,i) to coeff
        """
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        for (i,j), c in u_dict.items():
            u[i*m+j] = c
        for (j,k), c in v_dict.items():
            v[j*p+k] = c
        for (k,i), c in w_dict.items():
            w[k*n+i] = c
        terms.append((u, v, w))
    
    # 4x4 x 4x4 Strassen decomposition
    # Define the 7 products
    
    # For each product, define X (2x2) as linear combo of A elements, Y (2x2) as linear combo of B elements
    # Then apply 2x2 Strassen to get 7 scalar products
    
    # X elements: X00, X01, X10, X11 (each is a dict from (i,j) to coeff)
    # Y elements: Y00, Y01, Y10, Y11 (each is a dict from (j,k) to coeff)
    
    def strassen_2x2(X, Y, result_positions):
        """
        X: list of 4 dicts [X00, X01, X10, X11]
        Y: list of 4 dicts [Y00, Y01, Y10, Y11]
        result_positions: list of 4 lists, each list contains (position_in_C, coeff)
        where position_in_C is (i,k) and coeff is the coefficient
        """
        X00, X01, X10, X11 = X
        Y00, Y01, Y10, Y11 = Y
        
        # s1 = (X00+X11)(Y00+Y11)
        u1 = {}
        for (i,j), c in X00.items():
            u1[(i,j)] = u1.get((i,j), 0) + c
        for (i,j), c in X11.items():
            u1[(i,j)] = u1.get((i,j), 0) + c
        v1 = {}
        for (j,k), c in Y00.items():
            v1[(j,k)] = v1.get((j,k), 0) + c
        for (j,k), c in Y11.items():
            v1[(j,k)] = v1.get((j,k), 0) + c
        w1 = {}
        for (pos, coeff) in result_positions[0]:
            w1[pos] = w1.get(pos, 0) + coeff
        for (pos, coeff) in result_positions[3]:
            w1[pos] = w1.get(pos, 0) + coeff
        add_term(u1, v1, w1)
        
        # s2 = (X10+X11)Y00
        u2 = {}
        for (i,j), c in X10.items():
            u2[(i,j)] = u2.get((i,j), 0) + c
        for (i,j), c in X11.items():
            u2[(i,j)] = u2.get((i,j), 0) + c
        v2 = dict(Y00)
        w2 = {}
        for (pos, coeff) in result_positions[2]:
            w2[pos] = w2.get(pos, 0) + coeff
        for (pos, coeff) in result_positions[3]:
            w2[pos] = w2.get(pos, 0) + coeff * (-1)
        add_term(u2, v2, w2)
        
        # s3 = X00(Y01-Y11)
        u3 = dict(X00)
        v3 = {}
        for (j,k), c in Y01.items():
            v3[(j,k)] = v3.get((j,k), 0) + c
        for (j,k), c in Y11.items():
            v3[(j,k)] = v3.get((j,k), 0) - c
        w3 = {}
        for (pos, coeff) in result_positions[1]:
            w3[pos] = w3.get(pos, 0) + coeff
        for (pos, coeff) in result_positions[3]:
            w3[pos] = w3.get(pos, 0) + coeff
        add_term(u3, v3, w3)
        
        # s4 = X11(Y10-Y00)
        u4 = dict(X11)
        v4 = {}
        for (j,k), c in Y10.items():
            v4[(j,k)] = v4.get((j,k), 0) + c
        for (j,k), c in Y00.items():
            v4[(j,k)] = v4.get((j,k), 0) - c
        w4 = {}
        for (pos, coeff) in result_positions[0]:
            w4[pos] = w4.get(pos, 0) + coeff
        for (pos, coeff) in result_positions[2]:
            w4[pos] = w4.get(pos, 0) + coeff
        add_term(u4, v4, w4)
        
        # s5 = (X00+X01)Y11
        u5 = {}
        for (i,j), c in X00.items():
            u5[(i,j)] = u5.get((i,j), 0) + c
        for (i,j), c in X01.items():
            u5[(i,j)] = u5.get((i,j), 0) + c
        v5 = dict(Y11)
        w5 = {}
        for (pos, coeff) in result_positions[0]:
            w5[pos] = w5.get(pos, 0) + coeff * (-1)
        for (pos, coeff) in result_positions[1]:
            w5[pos] = w5.get(pos, 0) + coeff
        add_term(u5, v5, w5)
        
        # s6 = (X10-X00)(Y00+Y01)
        u6 = {}
        for (i,j), c in X10.items():
            u6[(i,j)] = u6.get((i,j), 0) + c
        for (i,j), c in X00.items():
            u6[(i,j)] = u6.get((i,j), 0) - c
        v6 = {}
        for (j,k), c in Y00.items():
            v6[(j,k)] = v6.get((j,k), 0) + c
        for (j,k), c in Y01.items():
            v6[(j,k)] = v6.get((j,k), 0) + c
        w6 = {}
        for (pos, coeff) in result_positions[3]:
            w6[pos] = w6.get(pos, 0) + coeff
        add_term(u6, v6, w6)
        
        # s7 = (X01-X11)(Y10+Y11)
        u7 = {}
        for (i,j), c in X01.items():
            u7[(i,j)] = u7.get((i,j), 0) + c
        for (i,j), c in X11.items():
            u7[(i,j)] = u7.get((i,j), 0) - c
        v7 = {}
        for (j,k), c in Y10.items():
            v7[(j,k)] = v7.get((j,k), 0) + c
        for (j,k), c in Y11.items():
            v7[(j,k)] = v7.get((j,k), 0) + c
        w7 = {}
        for (pos, coeff) in result_positions[0]:
            w7[pos] = w7.get(pos, 0) + coeff
        add_term(u7, v7, w7)
    
    # Now define the 7 products for 4x4 x 4x4
    # A blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B blocks: B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
    
    def block_A(r0, r1, c0, c1):
        d = {}
        for i in range(r0, r1):
            for j in range(c0, c1):
                d[(i,j)] = 1
        return d
    
    def block_B(r0, r1, c0, c1):
        d = {}
        for j in range(r0, r1):
            for k in range(c0, c1):
                d[(j,k)] = 1
        return d
    
    A00 = block_A(0,2,0,2)
    A01 = block_A(0,2,2,4)
    A10 = block_A(2,4,0,2)
    A11 = block_A(2,4,2,4)
    
    B00 = block_B(0,2,0,2)
    B01 = block_B(0,2,2,4)
    B10 = block_B(2,4,0,2)
    B11 = block_B(2,4,2,4)
    
    # C positions for 4x4x4 (k,i) indexing for W
    # C00: i in 0,1; k in 0,1 -> (k,i) for k in 0,1, i in 0,1
    # C01: i in 0,1; k in 2,3
    # C10: i in 2,3; k in 0,1
    # C11: i in 2,3; k in 2,3
    
    def c_block(i0, i1, k0, k1):
        return [(k, i) for k in range(k0, k1) for i in range(i0, i1)]
    
    C00_pos = c_block(0,2,0,2)
    C01_pos = c_block(0,2,2,4)
    C10_pos = c_block(2,4,0,2)
    C11_pos = c_block(2,4,2,4)
    
    # Product 1: X=A00+A11, Y=B00+B11, result: C00, C01, C10, C11
    # P00=C00, P01=C01, P10=C10, P11=C11
    X1 = [
        {**{k: v for k, v in A00.items()}, **{k: v for k, v in A11.items()}},
        {**{k: v for k, v in A00.items()}, **{k: v for k, v in A11.items()}},  # Wait, X01 should be A00[0,1]+A11[0,1]
    ]
    
    # I'm making an error. Let me be more careful.
    # X is a 2x2 matrix where X[r,c] is a 2x2 block.
    # For Product 1: X = A00+A11 means:
    #   X[0,0] = A00[0,0] + A11[0,0] = A[0,0]+A[2,2]
    #   X[0,1] = A00[0,1] + A11[0,1] = A[0,1]+A[2,3]
    #   X[1,0] = A00[1,0] + A11[1,0] = A[1,0]+A[3,2]
    #   X[1,1] = A00[1,1] + A11[1,1] = A[1,1]+A[3,3]
    
    def add_dicts(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) + v
        return d
    
    def sub_dicts(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) - v
        return d
    
    # Product 1: X = A00+A11, Y = B00+B11
    X1_00 = add_dicts(A00, A11)  # This is wrong! A00 and A11 have different indices
    # A00 has keys (0,0),(0,1),(1,0),(1,1)
    # A11 has keys (2,2),(2,3),(3,2),(3,3)
    # So add_dicts just combines them
    
    X1 = [
        add_dicts(A00, A11),  # X[0,0]
        add_dicts(A01, A11),  # X[0,1] -- WRONG! Should be A00[0,1]+A11[0,1]
    ]
    
    # I see my error. X is a 2x2 matrix of 2x2 blocks.
    # X[0,0] = A00[0,0] + A11[0,0]
    # But A00 is already a 2x2 block. So X[0,0] as a scalar is A[0,0]+A[2,2].
    # X[0,1] as a scalar is A[0,1]+A[2,3].
    # X[1,0] as a scalar is A[1,0]+A[3,2].
    # X[1,1] as a scalar is A[1,1]+A[3,3].
    
    # So for the 2x2 Strassen, X00 = A[0,0]+A[2,2], X01 = A[0,1]+A[2,3], etc.
    
    # Let me redefine. For each product, I'll specify X00, X01, X10, X11 as dicts
    # where each dict maps (i,j) to coefficient.
    
    # Product 1: X = A00+A11
    # X00 (scalar X[0,0]) = A[0,0] + A[2,2]
    # X01 (scalar X[0,1]) = A[0,1] + A[2,3]
    # X10 (scalar X[1,0]) = A[1,0] + A[3,2]
    # X11 (scalar X[1,1]) = A[1,1] + A[3,3]
    
    def prod1_X():
        return [
            {(0,0): 1, (2,2): 1},
            {(0,1): 1, (2,3): 1},
            {(1,0): 1, (3,2): 1},
            {(1,1): 1, (3,3): 1}
        ]
    
    def prod1_Y():
        return [
            {(0,0): 1, (2,2): 1},
            {(0,1): 1, (2,3): 1},
            {(1,0): 1, (3,2): 1},
            {(1,1): 1, (3,3): 1}
        ]
    
    # Result: P00=C00, P01=C01, P10=C10, P11=C11
    # result_positions[0] = C00 with coeff +1
    # result_positions[1] = C01 with coeff +1
    # result_positions[2] = C10 with coeff +1
    # result_positions[3] = C11 with coeff +1
    
    rp1 = [
        [(pos, 1) for pos in C00_pos],
        [(pos, 1) for pos in C01_pos],
        [(pos, 1) for pos in C10_pos],
        [(pos, 1) for pos in C11_pos]
    ]
    
    strassen_2x2(prod1_X(), prod1_Y(), rp1)
    
    # Product 2: X = A10+A11, Y = B00
    # X00 = A[2,0]+A[2,2], X01 = A[2,1]+A[2,3], X10 = A[3,0]+A[3,2], X11 = A[3,1]+A[3,3]
    # Y00 = B[0,0], Y01 = B[0,1], Y10 = B[1,0], Y11 = B[1,1]
    # Result: P00 not used, P01 not used, P10=C10, P11=C11
    # Actually: C10 = M2+M4, C11 = M1-M2+M3+M6
    # So product 2 contributes to C10 and C11 only
    
    # Wait, I need to re-examine. The Strassen formula gives:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11) -> product 1
    # M2 = (A10+A11)B00 -> product 2
    # M3 = A00(B01-B11) -> product 3
    # M4 = A11(B10-B00) -> product 4
    # M5 = (A00+A01)B11 -> product 5
    # M6 = (A10-A00)(B00+B01) -> product 6
    # M7 = (A01-A11)(B10+B11) -> product 7
    #
    # So each product contributes to specific C blocks:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    # So for product 2 (M2): contributes to C10 with +1 and C11 with -1
    # In the 2x2 Strassen, P10 and P11 are the relevant outputs.
    # result_positions[2] = C10 with +1, result_positions[3] = C11 with -1
    # result_positions[0] and [1] are empty (not used)
    
    rp2 = [
        [],
        [],
        [(pos, 1) for pos in C10_pos],
        [(pos, -1) for pos in C11_pos]
    ]
    
    X2 = [
        {(2,0): 1, (2,2): 1},
        {(2,1): 1, (2,3): 1},
        {(3,0): 1, (3,2): 1},
        {(3,1): 1, (3,3): 1}
    ]
    Y2 = [
        {(0,0): 1},
        {(0,1): 1},
        {(1,0): 1},
        {(1,1): 1}
    ]
    
    strassen_2x2(X2, Y2, rp2)
    
    # Product 3 (M3): A00(B01-B11), contributes to C01(+1), C11(+1)
    rp3 = [
        [],
        [(pos, 1) for pos in C01_pos],
        [],
        [(pos, 1) for pos in C11_pos]
    ]
    X3 = [
        {(0,0): 1},
        {(0,1): 1},
        {(1,0): 1},
        {(1,1): 1}
    ]
    Y3 = [
        {},  # Y00
        {(0,2): 1, (0,3): 1, (2,2): -1, (2,3): -1},  # Y01 = B01 - B11... wait
    ]
    
    # Y = B01-B11
    # Y00 = B01[0,0] - B11[0,0] = B[0,2] - B[2,2]
    # Y01 = B01[0,1] - B11[0,1] = B[0,3] - B[2,3]
    # Y10 = B01[1,0] - B11[1,0] = B[1,2] - B[3,2]
    # Y11 = B01[1,1] - B11[1,1] = B[1,3] - B[3,3]
    
    Y3 = [
        {(0,2): 1, (2,2): -1},
        {(0,3): 1, (2,3): -1},
        {(1,2): 1, (3,2): -1},
        {(1,3): 1, (3,3): -1}
    ]
    
    strassen_2x2(X3, Y3, rp3)
    
    # Product 4 (M4): A11(B10-B00), contributes to C00(+1), C10(+1)
    rp4 = [
        [(pos, 1) for pos in C00_pos],
        [],
        [(pos, 1) for pos in C10_pos],
        []
    ]
    X4 = [
        {(2,2): 1},
        {(2,3): 1},
        {(3,2): 1},
        {(3,3): 1}
    ]
    # Y = B10-B00
    # Y00 = B10[0,0]-B00[0,0] = B[2,0]-B[0,0]
    # Y01 = B10[0,1]-B00[0,1] = B[2,1]-B[0,1]
    # Y10 = B10[1,0]-B00[1,0] = B[3,0]-B[1,0]
    # Y11 = B10[1,1]-B00[1,1] = B[3,1]-B[1,1]
    Y4 = [
        {(2,0): 1, (0,0): -1},
        {(2,1): 1, (0,1): -1},
        {(3,0): 1, (1,0): -1},
        {(3,1): 1, (1,1): -1}
    ]
    
    strassen_2x2(X4, Y4, rp4)
    
    # Product 5 (M5): (A00+A01)B11, contributes to C00(-1), C01(+1)
    rp5 = [
        [(pos, -1) for pos in C00_pos],
        [(pos, 1) for pos in C01_pos],
        [],
        []
    ]
    # X = A00+A01
    # X00 = A[0,0], X01 = A[0,1]+A[0,2], X10 = A[1,0], X11 = A[1,1]+A[1,2]
    X5 = [
        {(0,0): 1},
        {(0,1): 1, (0,2): 1},
        {(1,0): 1},
        {(1,1): 1, (1,2): 1}
    ]
    # Y = B11
    Y5 = [
        {(2,2): 1},
        {(2,3): 1},
        {(3,2): 1},
        {(3,3): 1}
    ]
    
    strassen_2x2(X5, Y5, rp5)
    
    # Product 6 (M6): (A10-A00)(B00+B01), contributes to C11(+1)
    rp6 = [
        [],
        [],
        [],
        [(pos, 1) for pos in C11_pos]
    ]
    # X = A10-A00
    # X00 = A[2,0]-A[0,0], X01 = A[2,1]-A[0,1], X10 = A[3,0]-A[1,0], X11 = A[3,1]-A[1,1]
    X6 = [
        {(2,0): 1, (0,0): -1},
        {(2,1): 1, (0,1): -1},
        {(3,0): 1, (1,0): -1},
        {(3,1): 1, (1,1): -1}
    ]
    # Y = B00+B01
    # Y00 = B[0,0], Y01 = B[0,1]+B[0,2], Y10 = B[1,0], Y11 = B[1,1]+B[1,2]
    Y6 = [
        {(0,0): 1},
        {(0,1): 1, (0,2): 1},
        {(1,0): 1},
        {(1,1): 1, (1,2): 1}
    ]
    
    strassen_2x2(X6, Y6, rp6)
    
    # Product 7 (M7): (A01-A11)(B10+B11), contributes to C00(+1)
    rp7 = [
        [(pos, 1) for pos in C00_pos],
        [],
        [],
        []
    ]
    # X = A01-A11
    # X00 = A[0,2]-A[2,2], X01 = A[0,3]-A[2,3], X10 = A[1,2]-A[3,2], X11 = A[1,3]-A[3,3]
    X7 = [
        {(0,2): 1, (2,2): -1},
        {(0,3): 1, (2,3): -1},
        {(1,2): 1, (3,2): -1},
        {(1,3): 1, (3,3): -1}
    ]
    # Y = B10+B11
    # Y00 = B[2,0], Y01 = B[2,1]+B[2,2], Y10 = B[3,0], Y11 = B[3,1]+B[3,2]... 
    # Wait: B10 is B[2:4,0:2], B11 is B[2:4,2:4]
    # Y00 = B10[0,0]+B11[0,0] = B[2,0]+B[2,2]
    # Y01 = B10[0,1]+B11[0,1] = B[2,1]+B[2,3]
    # Y10 = B10[1,0]+B11[1,0] = B[3,0]+B[3,2]
    # Y11 = B10[1,1]+B11[1,1] = B[3,1]+B[3,3]
    Y7 = [
        {(2,0): 1, (2,2): 1},
        {(2,1): 1, (2,3): 1},
        {(3,0): 1, (3,2): 1},
        {(3,1): 1, (3,3): 1}
    ]
    
    strassen_2x2(X7, Y7, rp7)
    
    # Now add the 4x4 x 4x1 part (schoolbook)
    # For columns 4,5,6,7,8 of B (i.e., k=4..8)
    # A is 4x4, B_col is 4x1, C_col is 4x1
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                add_term(u, v, w)
    
    rank = len(terms)
    
    # Build U, V, W
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(terms):
        for idx in range(n*m):
            U[idx][c] = u[idx]
        for idx in range(m*p):
            V[idx][c] = v[idx]
        for idx in range(p*n):
            W[idx][c] = w[idx]
    
    return {"U": U, "V": V, "W": W}
