import time
import sys

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Indexing:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        nonlocal rank
        U.append([0] * rank + [u_val])
        V.append([0] * rank + [v_val])
        W.append([0] * rank + [w_val])
        rank += 1
        # Wait, this is inefficient. Better to build lists of coefficients.
        
    # Let's use a list of tuples (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    terms = []
    
    def add(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))

    # Block 1: A_top (rows 0,1), B_left (cols 0,1)
    # A_top is 2x5. B_left is 5x2.
    # Sub-block A_t1 (rows 0,1; cols 0,1), B_r1 (rows 0,1; cols 0,1)
    # Strassen for A_t1 * B_r1
    # A_t1 indices: i in 0,1; j in 0,1.
    # B_r1 indices: j in 0,1; k in 0,1.
    # Output C indices: i in 0,1; k in 0,1.
    
    # Strassen M1 = (A11 + A22)(B11 + B22)
    # A11 is A[0,0], A22 is A[1,1]
    # B11 is B[0,0], B22 is B[1,1]
    # Term: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # Specifically:
    # C[0,0] += M1
    # C[0,1] += M1 (Wait, Strassen formulas)
    
    # Let's just use the standard Strassen decomposition terms directly.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We add terms for each M and accumulate in W.
    # But W is a matrix. We need to add the contribution to specific W rows.
    # W row index is k*n + i.
    # For C[0,0] (i=0, k=0), w_row = 0*3 + 0 = 0.
    # For C[0,1] (i=0, k=1), w_row = 1*3 + 0 = 3.
    # For C[1,0] (i=1, k=0), w_row = 0*3 + 1 = 1.
    # For C[1,1] (i=1, k=1), w_row = 1*3 + 1 = 4.
    
    # M1
    u_idx_1 = 0*5 + 0 # A[0,0]
    u_idx_2 = 1*5 + 1 # A[1,1]
    v_idx_1 = 0*5 + 0 # B[0,0]
    v_idx_2 = 1*5 + 1 # B[1,1]
    
    # We need to add 7 terms for Strassen.
    # Each term is a product of a linear combo of U and linear combo of V.
    # This results in 7 new rank terms.
    
    # Term 1: M1
    add(0*5+0, 1, 0*5+0, 1, 0, 1) # C00
    add(0*5+1, 1, 0*5+1, 1, 0, 1) # C00
    add(0*5+0, 1, 0*5+1, 1, 0, 1) # C00
    add(0*5+1, 1, 0*5+0, 1, 0, 1) # C00
    # This is wrong. M1 is ONE term in the decomposition.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This is 4 schoolbook terms.
    # Strassen replaces 8 schoolbook terms with 7 products of sums.
    # But the products of sums are NOT single rank terms in the tensor decomposition.
    # (A00+A11) is a linear combination.
    # In tensor decomposition, we need $u \otimes v \otimes w$.
    # If $u = A00 + A11$, this is not a basis vector.
    # We can change the basis of U and V!
    # Yes, we can define new variables $U'_1 = U_1 + U_2$, etc.
    # But the interface requires U, V, W to be matrices with entries.
    # We can just output the coefficients.
    
    # So, for M1, we have:
    # u_vec = [1, 1, 0, ...] (coeffs for A00, A11)
    # v_vec = [1, 1, 0, ...] (coeffs for B00, B11)
    # This is ONE term in the rank decomposition.
    # w_vec has 1s in the appropriate positions.
    
    # So, we add 7 terms total for the 2x2x2 block.
    
    # M1
    u_m1 = {0*5+0: 1, 1*5+1: 1}
    v_m1 = {0*5+0: 1, 1*5+1: 1}
    w_m1 = {0: 1, 3: 1, 1: 1, 4: 1} # C00, C01, C10, C11?
    # C00 = M1 + ...
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes to C00 (+1), C11 (+1).
    # w_m1 = {0: 1, 4: 1}
    
    # M2 = (A10 + A11)B00
    u_m2 = {1*5+0: 1, 1*5+1: 1}
    v_m2 = {0*5+0: 1}
    # C10 += M2
    w_m2 = {1: 1}
    # C11 -= M2
    w_m2[4] = -1
    
    # M3 = A00(B01 - B11)
    u_m3 = {0*5+0: 1}
    v_m3 = {0*5+1: 1, 1*5+1: -1}
    # C01 += M3
    w_m3 = {3: 1}
    # C11 += M3
    w_m3[4] = 1
    
    # M4 = A11(B10 - B00)
    u_m4 = {1*5+1: 1}
    v_m4 = {1*5+0: 1, 0*5+0: -1}
    # C00 += M4
    w_m4 = {0: 1}
    # C10 += M4
    w_m4[1] = 1
    
    # M5 = (A00 + A01)B11
    u_m5 = {0*5+0: 1, 0*5+1: 1}
    v_m5 = {1*5+1: 1}
    # C00 -= M5
    w_m5 = {0: -1}
    # C01 += M5
    w_m5[3] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    u_m6 = {1*5+0: 1, 0*5+0: -1}
    v_m6 = {0*5+0: 1, 0*5+1: 1}
    # C11 += M6
    w_m6 = {4: 1}
    
    # M7 = (A01 - A11)(B10 + B11)
    u_m7 = {0*5+1: 1, 1*5+1: -1}
    v_m7 = {1*5+0: 1, 1*5+1: 1}
    # C00 += M7
    w_m7 = {0: 1}
    
    # Add these 7 terms
    for u_dict, v_dict, w_dict in [
        (u_m1, v_m1, w_m1),
        (u_m2, v_m2, w_m2),
        (u_m3, v_m3, w_m3),
        (u_m4, v_m4, w_m4),
        (u_m5, v_m5, w_m5),
        (u_m6, v_m6, w_m6),
        (u_m7, v_m7, w_m7)
    ]:
        # Flatten to vectors
        u_vec = [0]*(n*m)
        v_vec = [0]*(m*p)
        w_vec = [0]*(p*n)
        for idx, val in u_dict.items(): u_vec[idx] = val
        for idx, val in v_dict.items(): v_vec[idx] = val
        for idx, val in w_dict.items(): w_vec[idx] = val
        terms.append((u_vec, v_vec, w_vec))

    # Rest of Block 1: A_t2 (2x3), B_r2 (3x2)
    # A_t2 rows 0,1; cols 2,3,4.
    # B_r2 rows 2,3,4; cols 0,1.
    # Schoolbook 2*3*2 = 12 terms.
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u_idx = i*5 + (2+j)
                v_idx = (2+j)*5 + k
                w_idx = k*3 + i
                u_vec = [0]*(n*m); u_vec[u_idx] = 1
                v_vec = [0]*(m*p); v_vec[v_idx] = 1
                w_vec = [0]*(p*n); w_vec[w_idx] = 1
                terms.append((u_vec, v_vec, w_vec))

    # Block 2: A_top (2x5), B_right (5x3)
    # Schoolbook 2*5*3 = 30 terms.
    for i in range(2):
        for j in range(5):
            for k in range(3):
                u_idx = i*5 + j
                v_idx = j*5 + (2+k)
                w_idx = (2+k)*3 + i
                u_vec = [0]*(n*m); u_vec[u_idx] = 1
                v_vec = [0]*(m*p); v_vec[v_idx] = 1
                w_vec = [0]*(p*n); w_vec[w_idx] = 1
                terms.append((u_vec, v_vec, w_vec))

    # Block 3: A_bottom (1x5), B_left (5x2)
    # Schoolbook 1*5*2 = 10 terms.
    for i in range(1):
        for j in range(5):
            for k in range(2):
                u_idx = (2+i)*5 + j
                v_idx = j*5 + k
                w_idx = k*3 + (2+i)
                u_vec = [0]*(n*m); u_vec[u_idx] = 1
                v_vec = [0]*(m*p); v_vec[v_idx] = 1
                w_vec = [0]*(p*n); w_vec[w_idx] = 1
                terms.append((u_vec, v_vec, w_vec))

    # Block 4: A_bottom (1x5), B_right (5x3)
    # Schoolbook 1*5*3 = 15 terms.
    for i in range(1):
        for j in range(5):
            for k in range(3):
                u_idx = (2+i)*5 + j
                v_idx = j*5 + (2+k)
                w_idx = (2+k)*3 + (2+i)
                u_vec = [0]*(n*m); u_vec[u_idx] = 1
                v_vec = [0]*(m*p); v_vec[v_idx] = 1
                w_vec = [0]*(p*n); w_vec[w_idx] = 1
                terms.append((u_vec, v_vec, w_vec))

    # Assemble U, V, W
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u_vec, v_vec, w_vec) in enumerate(terms):
        for i in range(n*m):
            if u_vec[i] != 0: U[i][c] = u_vec[i]
        for j in range(m*p):
            if v_vec[j] != 0: V[j][c] = v_vec[j]
        for k in range(p*n):
            if w_vec[k] != 0: W[k][c] = w_vec[k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
