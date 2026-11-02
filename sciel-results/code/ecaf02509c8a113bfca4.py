import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    # We will build the decomposition block by block.
    # Rank 80 decomposition.
    
    # Initialize U, V, W
    rank = 80
    U = [[0] * rank for _ in range(n * m)]  # 12 x 80
    V = [[0] * rank for _ in range(m * p)]  # 28 x 80
    W = [[0] * rank for _ in range(p * n)]  # 21 x 80
    
    col = 0
    
    # Helper to add a term: u_idx, v_idx, w_idx, val
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Index mapping:
    # U[i, j] corresponds to A[i, j]
    # V[j, k] corresponds to B[j, k]
    # W[k, i] corresponds to C[k, i] -> W[k*n + i, r]
    
    def add_term(i, j, k, val):
        nonlocal col
        U[i * m + j][col] = val
        V[j * p + k][col] = val
        W[k * n + i][col] = val
        col += 1

    # --- Block A_top B_left (2x4 * 4x4) -> 2x4 ---
    # A_top = [[A11, A12]] (2x2 blocks)
    # B_left = [[B11, B12], [B21, B22]] (2x2 blocks)
    # Result = A11 B11 + A12 B21 (col 0-1) and A11 B12 + A12 B22 (col 2-3)
    
    # Strassen for A11 B11
    # A11: rows 0-1, cols 0-1 of A
    # B11: rows 0-1, cols 0-1 of B
    # Indices for A11: i in 0-1, j in 0-1
    # Indices for B11: j in 0-1, k in 0-1
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22) -> here A22 is A12? No.
    # Strassen for X(2x2) * Y(2x2):
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # We need to add 7 terms for A11 * B11.
    # x11 is A[0,0], x12 is A[0,1], x21 is A[1,0], x22 is A[1,1]
    # y11 is B[0,0], y12 is B[0,1], y21 is B[1,0], y22 is B[1,1]
    
    # M1: (A00+A11)(B00+B11)
    # This expands to A00B00 + A00B11 + A11B00 + A11B11.
    # We cannot just add products. We must add the tensor terms for the rank decomposition.
    # Strassen decomposition of the tensor T[i,j,k] = X[i,j]Y[j,k].
    # The rank 7 decomposition is:
    # M1: u=(1,1,1,1), v=(1,1,1,1), w=(1,1,1,1) ? No.
    # The vectors are:
    # u1 = (1, 0, 0, 1) -> A00, A11
    # v1 = (1, 0, 0, 1) -> B00, B11
    # w1 = (1, 1, 1, 1) -> C00, C01, C10, C11 ?
    # Wait, the output C is 2x2. W indices are k*n+i.
    # For 2x2*2x2, n=2, p=2. W size 4.
    # C00: k=0, i=0. C01: k=1, i=0. C10: k=0, i=1. C11: k=1, i=1.
    # M1 = (A00+A11)(B00+B11) contributes to C00, C01, C10, C11.
    # C00 = A00B00 + A00B11 + A11B00 + A11B11.
    # C01 = A00B01 + A00B11 + A11B01 + A11B11.
    # C10 = A10B00 + A10B11 + A11B00 + A11B11.
    # C11 = A10B01 + A10B11 + A11B01 + A11B11.
    # The Strassen M1 term is added to C00, C01, C10, C11 with coefficient 1.
    # But M1 is just one product.
    # The decomposition is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M, we add a term to U, V, W.
    # U vector for M1: A00=1, A11=1, others 0.
    # V vector for M1: B00=1, B11=1, others 0.
    # W vector for M1: C00=1, C01=1, C10=1, C11=1.
    
    # Let's define the 7 Strassen terms for a generic 2x2*2x2 block.
    # Inputs: a_base (i,j offsets), b_base (j,k offsets), c_base (k,i offsets)
    # A block is at rows r0..r1, cols c0..c1 of A.
    # B block is at rows r0..r1, cols c0..c1 of B.
    # C block is at rows c0..c1, cols r0..r1 of C.
    
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A indices: (a_r0, a_c0), (a_r0, a_c0+1), (a_r0+1, a_c0), (a_r0+1, a_c0+1)
        # B indices: (b_r0, b_c0), (b_r0, b_c0+1), (b_r0+1, b_c0), (b_r0+1, b_c0+1)
        # C indices: (c_r0, c_c0), (c_r0, c_c0+1), (c_r0+1, c_c0), (c_r0+1, c_c0+1)
        
        # M1: (A00+A11)(B00+B11)
        u = [(a_r0, a_c0, 1), (a_r0+1, a_c0+1, 1)]
        v = [(b_r0, b_c0, 1), (b_r0+1, b_c0+1, 1)]
        w = [(c_r0, c_c0, 1), (c_r0, c_c0+1, 1), (c_r0+1, c_c0, 1), (c_r0+1, c_c0+1, 1)]
        for (i, j, val) in u: U[i*m+j][col] += val
        for (j, k, val) in v: V[j*p+k][col] += val
        for (k, i, val) in w: W[k*n+i][col] += val
        col += 1
        
        # M2: (A10+A11)B00
        u = [(a_r0+1, a_c0, 1), (a_r0+1, a_c0+1, 1)]
        v = [(b_r0, b_c0, 1)]
        w = [(c_r0, c_c0+1, 1), (c_r0+1, c_c0+1, 1)] # C10, C11
        for (i, j, val) in u: U[i*m+j][col] += val
        for (j, k, val) in v: V[j*p+k][col] += val
        for (k, i, val) in w: W[k*n+i][col] += val
        col += 1
        
        # M3: A00(B01-B11)
        u = [(a_r0, a_c0, 1)]
        v = [(b_r0, b_c0+1, 1), (b_r0+1, b_c0+1, -1)]
        w = [(c_r0, c_c0+1, 1), (c_r0+1, c_c0+1, 1)] # C01, C11
        for (i, j, val) in u: U[i*m+j][col] += val
        for (j, k, val) in v: V[j*p+k][col] += val
        for (k, i, val) in w: W[k*n+i][col] += val
        col += 1
        
        # M4: A11(B10-B00)
        u = [(a_r0+1, a_c0+1, 1)]
        v = [(b_r0+1, b_c0, 1), (b_r0, b_c0, -1)]
        w = [(c_r0, c_c0, 1), (c_r0+1, c_c0, 1)] # C00, C10
        for (i, j, val) in u: U[i*m+j][col] += val
        for (j, k, val) in v: V[j*p+k][col] += val
        for (k, i, val) in w: W[k*n+i][col] += val
        col += 1
        
        # M5: (A00+A01)B11
        u = [(a_r0, a_c0, 1), (a_r0, a_c0+1, 1)]
        v = [(b_r0+1, b_c0+1, 1)]
        w = [(c_r0, c_c0, -1), (c_r0, c_c0+1, 1), (c_r0+1, c_c0+1, 1)] # C00(-1), C01, C11
        for (i, j, val) in u: U[i*m+j][col] += val
        for (j, k, val) in v: V[j*p+k][col] += val
        for (k, i, val) in w: W[k*n+i][col] += val
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        u = [(a_r0+1, a_c0, 1), (a_r0, a_c0, -1)]
        v = [(b_r0, b_c0, 1), (b_r0, b_c0+1, 1)]
        w = [(c_r0+1, c_c0+1, 1)] # C11
        for (i, j, val) in u: U[i*m+j][col] += val
        for (j, k, val) in v: V[j*p+k][col] += val
        for (k, i, val) in w: W[k*n+i][col] += val
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        u = [(a_r0, a_c0+1, 1), (a_r0+1, a_c0+1, -1)]
        v = [(b_r0+1, b_c0, 1), (b_r0+1, b_c0+1, 1)]
        w = [(c_r0, c_c0, 1), (c_r0+1, c_c0+1, 1)] # C00, C11
        for (i, j, val) in u: U[i*m+j][col] += val
        for (j, k, val) in v: V[j*p+k][col] += val
        for (k, i, val) in w: W[k*n+i][col] += val
        col += 1

    # Block 1: A11 B11. A rows 0-1, cols 0-1. B rows 0-1, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Block 2: A12 B21. A rows 0-1, cols 2-3. B rows 2-3, cols 0-1. C rows 0-1, cols 0-1.
    # A12 is at (0-1, 2-3). B21 is at (2-3, 0-1).
    # Wait, B21 is rows 2-3 of B, cols 0-1 of B.
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Block 3: A11 B12. A rows 0-1, cols 0-1. B rows 0-1, cols 2-3. C rows 2-3, cols 0-1.
    add_strassen(0, 0, 0, 2, 2, 0)
    
    # Block 4: A12 B22. A rows 0-1, cols 2-3. B rows 2-3, cols 2-3. C rows 2-3, cols 0-1.
    add_strassen(0, 2, 2, 2, 2, 0)
    
    # --- Block A_top B_right (2x4 * 4x3) -> 2x3 ---
    # Schoolbook.
    # A11 (2x2) * B13 (2x3). Rank 12.
    # A12 (2x2) * B23 (2x3). Rank 12.
    def add_schoolbook_2x2x3(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[(a_r0+i)*m + (a_c0+j)][col] = 1
                    V[(b_r0+j)*p + (b_c0+k)][col] = 1
                    W[(c_r0+k)*n + (c_c0+i)][col] = 1
                    col += 1

    # A11: r0=0, c0=0. B13: r0=0, c0=4. C13: r0=4, c0=0.
    add_schoolbook_2x2x3(0, 0, 0, 4, 4, 0)
    # A12: r0=0, c0=2. B23: r0=2, c0=4. C13: r0=4, c0=0.
    add_schoolbook_2x2x3(0, 2, 2, 4, 4, 0)
    
    # --- Block A_bot B_left (1x4 * 4x4) -> 1x4 ---
    # Schoolbook.
    # A21 (1x2) * B11 (2x2). Rank 4.
    # A22 (1x2) * B21 (2x2). Rank 4.
    # A21 (1x2) * B12 (2x2). Rank 4.
    # A22 (1x2) * B22 (2x2). Rank 4.
    def add_schoolbook_1x2x2(a_r, a_c0, b_r0, b_c0, c_r0, c_c):
        nonlocal col
        for j in range(2):
            for k in range(2):
                U[a_r*m + (a_c0+j)][col] = 1
                V[(b_r0+j)*p + (b_c0+k)][col] = 1
                W[(c_r0+k)*n + a_r][col] = 1
                col += 1

    # A21: r=2, c0=0. B11: r0=0, c0=0. C: r0=0, c=2.
    add_schoolbook_1x2x2(2, 0, 0, 0, 0, 2)
    # A22: r=2, c0=2. B21: r0=2, c0=0. C: r0=0, c=2.
    add_schoolbook_1x2x2(2, 2, 2, 0, 0, 2)
    # A21: r=2, c0=0. B12: r0=0, c0=2. C: r0=2, c=2.
    add_schoolbook_1x2x2(2, 0, 0, 2, 2, 2)
    # A22: r=2, c0=2. B22: r0=2, c0=2. C: r0=2, c=2.
    add_schoolbook_1x2x2(2, 2, 2, 2, 2, 2)
    
    # --- Block A_bot B_right (1x4 * 4x3) -> 1x3 ---
    # Schoolbook.
    # A21 (1x2) * B13 (2x3). Rank 6.
    # A22 (1x2) * B23 (2x3). Rank 6.
    def add_schoolbook_1x2x3(a_r, a_c0, b_r0, b_c0, c_r0, c_c):
        nonlocal col
        for j in range(2):
            for k in range(3):
                U[a_r*m + (a_c0+j)][col] = 1
                V[(b_r0+j)*p + (b_c0+k)][col] = 1
                W[(c_r0+k)*n + a_r][col] = 1
                col += 1

    # A21: r=2, c0=0. B13: r0=0, c0=4. C: r0=4, c=2.
    add_schoolbook_1x2x3(2, 0, 0, 4, 4, 2)
    # A22: r=2, c0=2. B23: r0=2, c0=4. C: r0=4, c=2.
    add_schoolbook_1x2x3(2, 2, 2, 4, 4, 2)
    
    return {"U": U, "V": V, "W": W}

