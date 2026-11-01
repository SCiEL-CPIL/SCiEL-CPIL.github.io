import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 43 decomposition.
    # Split A (3x3) into:
    # A11 (2x2), A12 (2x1)
    # A21 (1x2), A22 (1x1)
    # Split B (3x5) into:
    # B1 (2x5), B2 (1x5)
    # Split B1 (2x5) into:
    # B1a (2x2), B1b (2x2), B1c (2x1)
    
    # Indices for A (3x3): i in 0..2, j in 0..2
    # A11: i in 0..1, j in 0..1
    # A12: i in 0..1, j in 2
    # A21: i in 2, j in 0..1
    # A22: i in 2, j in 2
    
    # Indices for B (3x5): j in 0..2, k in 0..4
    # B1: j in 0..1, k in 0..4
    # B2: j in 2, k in 0..4
    
    # B1a: j in 0..1, k in 0..1
    # B1b: j in 0..1, k in 2..3
    # B1c: j in 0..1, k in 4
    
    # C (3x5): i in 0..2, k in 0..4
    # C11 = A11 B1a (2x2) -> C[i, 0..1] for i in 0..1
    # C12 = A11 B1b (2x2) -> C[i, 2..3] for i in 0..1
    # C13 = A11 B1c (2x1) -> C[i, 4] for i in 0..1
    # C14 = A12 B2 (2x5) -> C[i, 0..4] for i in 0..1
    # C21 = A21 B1 (1x5) -> C[2, 0..4]
    # C22 = A22 B2 (1x5) -> C[2, 0..4]
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # Helper to create zero vectors
    dim_U = n * m # 9
    dim_V = m * p # 15
    dim_W = p * n # 15
    
    # 1. A11 B1a (2x2 by 2x2) using Strassen (7 terms)
    # A11 indices in U: i in 0..1, j in 0..1 -> i*m+j -> 0,1,3,4
    # B1a indices in V: j in 0..1, k in 0..1 -> j*p+k -> 0,1,5,6
    # C indices in W: k*n+i -> k*3+i
    
    # Strassen terms for 2x2x2:
    # M1 = (A11 + A22) (B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B22 - B11)
    # M4 = A22 (B11 - B21)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    # Note: My A11 is 2x2, so A11[0,0], A11[0,1], A11[1,0], A11[1,1]
    # Let's map A11 to local indices 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # In global U: 
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 3
    # (1,1) -> 4
    # B1a to local indices 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # In global V:
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 5
    # (1,1) -> 6
    # C (2x2) to local indices 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # In global W:
    # (0,0) -> 0
    # (0,1) -> 3
    # (1,0) -> 1
    # (1,1) -> 4
    
    def strassen_2x2x2(a_idx, b_idx, c_idx, sign_a=1, sign_b=1):
        # a_idx, b_idx, c_idx are lists of 4 global indices
        # We add 7 terms
        # M1
        u1 = [0]*dim_U; v1 = [0]*dim_V; w1 = [0]*dim_W
        for i in range(4):
            u1[a_idx[i]] = sign_a
            v1[b_idx[i]] = sign_b
            w1[c_idx[i]] = 1
        # Wait, Strassen involves sums/differences.
        # We need to add terms to U, V, W that represent the bilinear forms.
        # Term 1: (A00+A11)(B00+B11) -> A00B00 + A00B11 + A11B00 + A11B11
        # This is not a single rank 1 term. It's a sum of 4 rank 1 terms?
        # No, Strassen replaces 8 rank 1 terms with 7 rank 1 terms of *combined* inputs.
        # The inputs to the multiplication are linear combinations of the original matrix entries.
        # So U, V, W must reflect these linear combinations.
        
        pass

    # This is getting complicated to implement Strassen inline with the global index mapping.
    # I will use the schoolbook for the 2x2x2 part as well to ensure correctness, 
    # which gives rank 8 for that part.
    # Total rank: 8 (A11B1a) + 8 (A11B1b) + 4 (A11B1c) + 10 (A12B2) + 10 (A21B1) + 5 (A22B2) = 45.
    # This is the same as the seed.
    
    # To improve, I must implement Strassen correctly.
    # Let's hardcode the Strassen decomposition for the 2x2x2 block.
    
    # A11 global indices: [0, 1, 3, 4]
    # B1a global indices: [0, 1, 5, 6]
    # C11 global indices (W): [0, 3, 1, 4] (k*n+i: 0*3+0=0, 0*3+1=1? No. W[k*n+i].
    # C is 3x5. i is row, k is col.
    # C11 is top-left 2x2 of C.
    # i in 0..1, k in 0..1.
    # W index: k*n + i.
    # (0,0) -> 0*3+0=0
    # (0,1) -> 0*3+1=1? No. W[k*n+i]. k is col, i is row.
    # C[0,0] -> W[0*3+0] = W[0]
    # C[0,1] -> W[1*3+0] = W[3]
    # C[1,0] -> W[0*3+1] = W[1]
    # C[1,1] -> W[1*3+1] = W[4]
    # So c_idx = [0, 3, 1, 4]
    
    # Let's implement the 7 Strassen terms for A11 B1a.
    # Local indices: A: 0,1,2,3. B: 0,1,2,3. C: 0,1,2,3.
    # M1: (A0+A3)(B0+B3) -> C0+=M1, C1-=M1, C2-=M1, C3+=M1
    # M2: (A2+A3)B0 -> C2+=M2
    # M3: A0(B3-B0) -> C0+=M3, C3+=M3
    # M4: A3(B0-B2) -> C1+=M4, C3+=M4
    # M5: (A0+A1)B3 -> C1+=M5, C3+=M5
    # M6: (A2-A0)(B0+B1) -> C2+=M6
    # M7: (A1-A3)(B2+B3) -> C1+=M7
    
    # We will add 7 terms to U, V, W.
    # Each term is a rank 1 tensor.
    # Term 1: u has 1s at A0, A3. v has 1s at B0, B3. w has 1 at C0, -1 at C1, -1 at C2, 1 at C3.
    # But W is a single vector. We can have coefficients in W.
    
    def add_strassen_term(a_coeffs, b_coeffs, c_coeffs):
        u = [0]*dim_U
        v = [0]*dim_V
        w = [0]*dim_W
        for idx, coeff in zip(a_idx, a_coeffs):
            u[idx] = coeff
        for idx, coeff in zip(b_idx, b_coeffs):
            v[idx] = coeff
        for idx, coeff in zip(c_idx, c_coeffs):
            w[idx] = coeff
        add_term(u, v, w)
        
    a_idx = [0, 1, 3, 4] # A00, A01, A10, A11
    b_idx = [0, 1, 5, 6] # B00, B01, B10, B11
    c_idx = [0, 3, 1, 4] # C00, C01, C10, C11 (W indices)
    
    # M1
    add_strassen_term([1, 0, 0, 1], [1, 0, 0, 1], [1, -1, -1, 1])
    # M2
    add_strassen_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])
    # M3
    add_strassen_term([1, 0, 0, 0], [1, 0, 0, -1], [1, 0, 0, 1])
    # M4
    add_strassen_term([0, 0, 0, 1], [1, 0, -1, 0], [0, 1, 0, 1])
    # M5
    add_strassen_term([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1])
    # M6
    add_strassen_term([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0])
    # M7
    add_strassen_term([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0])
    
    # 2. A11 B1b (2x2 by 2x2) using Strassen
    # B1b: j in 0..1, k in 2..3
    # B indices: j*p+k.
    # (0,2) -> 2
    # (0,3) -> 3
    # (1,2) -> 7
    # (1,3) -> 8
    b_idx_2 = [2, 3, 7, 8]
    # C12: i in 0..1, k in 2..3
    # C indices: k*n+i.
    # (0,2) -> 2*3+0 = 6
    # (0,3) -> 3*3+0 = 9
    # (1,2) -> 2*3+1 = 7
    # (1,3) -> 3*3+1 = 10
    c_idx_2 = [6, 9, 7, 10]
    
    add_strassen_term([1, 0, 0, 1], [1, 0, 0, 1], [1, -1, -1, 1])
    add_strassen_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])
    add_strassen_term([1, 0, 0, 0], [1, 0, 0, -1], [1, 0, 0, 1])
    add_strassen_term([0, 0, 0, 1], [1, 0, -1, 0], [0, 1, 0, 1])
    add_strassen_term([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1])
    add_strassen_term([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0])
    add_strassen_term([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0])
    # Note: The a_idx and c_idx patterns are the same, only b_idx changed.
    # Wait, the a_idx in add_strassen_term is hardcoded to the first block's a_idx?
    # No, I passed coefficients, but the function uses global a_idx.
    # I need to pass a_idx as well or make it generic.
    
    # Let's refactor add_strassen_term to accept indices.
    
    # I will restart the construction with a generic function.
    
    U.clear()
    V.clear()
    W.clear()
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    def add_block_strassen(a_indices, b_indices, c_indices):
        # a_indices: [A00, A01, A10, A11]
        # b_indices: [B00, B01, B10, B11]
        # c_indices: [C00, C01, C10, C11]
        
        def term(a_c, b_c, c_c):
            u = [0]*dim_U
            v = [0]*dim_V
            w = [0]*dim_W
            for i in range(4):
                u[a_indices[i]] = a_c[i]
                v[b_indices[i]] = b_c[i]
                w[c_indices[i]] = c_c[i]
            add_term(u, v, w)
            
        term([1, 0, 0, 1], [1, 0, 0, 1], [1, -1, -1, 1])
        term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])
        term([1, 0, 0, 0], [1, 0, 0, -1], [1, 0, 0, 1])
        term([0, 0, 0, 1], [1, 0, -1, 0], [0, 1, 0, 1])
        term([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1])
        term([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0])
        term([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0])
        
    # Block 1: A11 B1a
    a11_idx = [0, 1, 3, 4]
    b1a_idx = [0, 1, 5, 6]
    c11_idx = [0, 3, 1, 4]
    add_block_strassen(a11_idx, b1a_idx, c11_idx)
    
    # Block 2: A11 B1b
    b1b_idx = [2, 3, 7, 8]
    c12_idx = [6, 9, 7, 10]
    add_block_strassen(a11_idx, b1b_idx, c12_idx)
    
    # Block 3: A11 B1c (2x2 by 2x1) - Schoolbook (4 terms)
    # B1c: j in 0..1, k in 4
    # (0,4) -> 4
    # (1,4) -> 9
    b1c_idx = [4, 9]
    # C13: i in 0..1, k in 4
    # (0,4) -> 4*3+0 = 12
    # (1,4) -> 4*3+1 = 13
    c13_idx = [12, 13]
    
    # A11 B1c:
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Term 1: A00 B0 -> C0
    u = [0]*dim_U; v = [0]*dim_V; w = [0]*dim_W
    u[a11_idx[0]] = 1; v[b1c_idx[0]] = 1; w[c13_idx[0]] = 1
    add_term(u, v, w)
    
    # Term 2: A01 B1 -> C0
    u = [0]*dim_U; v = [0]*dim_V; w = [0]*dim_W
    u[a11_idx[1]] = 1; v[b1c_idx[1]] = 1; w[c13_idx[0]] = 1
    add_term(u, v, w)
    
    # Term 3: A10 B0 -> C1
    u = [0]*dim_U; v = [0]*dim_V; w = [0]*dim_W
    u[a11_idx[2]] = 1; v[b1c_idx[0]] = 1; w[c13_idx[1]] = 1
    add_term(u, v, w)
    
    # Term 4: A11 B1 -> C1
    u = [0]*dim_U; v = [0]*dim_V; w = [0]*dim_W
    u[a11_idx[3]] = 1; v[b1c_idx[1]] = 1; w[c13_idx[1]] = 1
    add_term(u, v, w)
    
    # Block 4: A12 B2 (2x1 by 1x5) - Schoolbook (10 terms)
    # A12: i in 0..1, j in 2
    # (0,2) -> 2
    # (1,2) -> 5
    a12_idx = [2, 5]
    # B2: j in 2, k in 0..4
    # (2,0) -> 10
    # (2,1) -> 11
    # (2,2) -> 12
    # (2,3) -> 13
    # (2,4) -> 14
    b2_idx = [10, 11, 12, 13, 14]
    # C14: i in 0..1, k in 0..4
    # (0,0) -> 0
    # (0,1) -> 3
    # (0,2) -> 6
    # (0,3) -> 9
    # (0,4) -> 12
    # (1,0) -> 1
    # (1,1) -> 4
    # (1,2) -> 7
    # (1,3) -> 10
    # (1,4) -> 13
    c14_idx = [0, 3, 6, 9, 12, 1, 4, 7, 10, 13]
    
    for i in range(2):
        for k in range(5):
            u = [0]*dim_U; v = [0]*dim_V; w = [0]*dim_W
            u[a12_idx[i]] = 1
            v[b2_idx[k]] = 1
            w[c14_idx[i*5+k]] = 1
            add_term(u, v, w)
            
    # Block 5: A21 B1 (1x2 by 2x5) - Schoolbook (10 terms)
    # A21: i in 2, j in 0..1
    # (2,0) -> 6
    # (2,1) -> 7
    a21_idx = [6, 7]
    # B1: j in 0..1, k in 0..4
    # (0,0) -> 0, (0,1) -> 1, (0,2) -> 2, (0,3) -> 3, (0,4) -> 4
    # (1,0) -> 5, (1,1) -> 6, (1,2) -> 7, (1,3) -> 8, (1,4) -> 9
    b1_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # C21: i in 2, k in 0..4
    # (2,0) -> 0*3+2 = 2
    # (2,1) -> 1*3+2 = 5
    # (2,2) -> 2*3+2 = 8
    # (2,3) -> 3*3+2 = 11
    # (2,4) -> 4*3+2 = 14
    c21_idx = [2, 5, 8, 11, 14]
    
    for j in range(2):
        for k in range(5):
            u = [0]*dim_U; v = [0]*dim_V; w = [0]*dim_W
            u[a21_idx[j]] = 1
            v[b1_idx[j*5+k]] = 1
            w[c21_idx[k]] = 1
            add_term(u, v, w)
            
    # Block 6: A22 B2 (1x1 by 1x5) - Schoolbook (5 terms)
    # A22: i in 2, j in 2
    # (2,2) -> 8
    a22_idx = [8]
    # B2: [10, 11, 12, 13, 14]
    # C22: i in 2, k in 0..4
    # (2,0) -> 2
    # (2,1) -> 5
    # (2,2) -> 8
    # (2,3) -> 11
    # (2,4) -> 14
    c22_idx = [2, 5, 8, 11, 14]
    
    for k in range(5):
        u = [0]*dim_U; v = [0]*dim_V; w = [0]*dim_W
        u[a22_idx[0]] = 1
        v[b2_idx[k]] = 1
        w[c22_idx[k]] = 1
        add_term(u, v, w)
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
