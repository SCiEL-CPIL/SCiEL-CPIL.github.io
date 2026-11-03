import numpy as np

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Build the target tensor (sparse)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    
    # We'll build the rank-80 schoolbook decomposition as a baseline,
    # then try to improve it.
    
    # Actually, let's try a smarter approach:
    # We know that for 4x4x4, the best known rank is 49 (or possibly lower).
    # For 4x4x5, we can think of it as 4x4x4 + 4x4x1.
    
    # Let's first build the 4x4x4 part using Strassen's algorithm (rank 7 for 2x2, so 7*7=49 for 4x4).
    # Then add 16 terms for the 4x4x1 part.
    
    # Strassen's 2x2x2 decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we use 7x7 = 49 terms.
    # The 4x4 matrix is divided into 2x2 blocks, each of size 2x2.
    # Block (r,c) of A has indices i in {2r, 2r+1}, j in {2c, 2c+1}
    # Block (r,c) of B has indices j in {2c, 2c+1}, k in {2r, 2r+1} (wait, B is 4x4, so k in 0..3)
    # Actually, B is 4x4, so block (r,c) of B has indices j in {2c, 2c+1}, k in {2r, 2r+1}
    # Block (r,c) of C has indices i in {2r, 2r+1}, k in {2c, 2c+1}
    
    # Let's define the Strassen factors for 2x2:
    # For each of the 7 terms, we have vectors u, v, w of length 4 (for the 2x2 matrix flattened)
    # u is for A (4 entries: A11, A12, A21, A22)
    # v is for B (4 entries: B11, B12, B21, B22)
    # w is for C (4 entries: C11, C12, C21, C22)
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # u1 = [1, 0, 0, 1]  (A11 + A22)
    # v1 = [1, 0, 0, 1]  (B11 + B22)
    # w1 = [1, 0, 0, 1]  (C11 + C22)
    
    # Term 2: M2 = (A21+A22)B11
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # w2 = [0, 0, 1, 0]  (C21)
    
    # Term 3: M3 = A11(B21-B22)
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 0, 1, -1]
    # w3 = [0, 1, 0, 0]  (C12)
    
    # Term 4: M4 = A22(B12-B11)
    # u4 = [0, 0, 0, 1]
    # v4 = [-1, 1, 0, 0]
    # w4 = [1, 0, 1, 0]  (C11 + C21)
    
    # Term 5: M5 = (A11+A12)B22
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # w5 = [-1, 1, 0, 0]  (-C11 + C12)
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # u6 = [-1, 0, 1, 0]
    # v6 = [1, 1, 0, 0]
    # w6 = [0, 0, 0, 1]  (C22)
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # u7 = [0, 1, 0, -1]
    # v7 = [0, 0, 1, 1]
    # w7 = [1, 0, 0, -1]  (C11 - C22)
    
    # Now for 4x4x4, we have 7 outer terms and 7 inner terms.
    # For outer term o and inner term i, the global term index is o*7 + i.
    
    # Let's define the block structure:
    # A is 4x4, blocks are 2x2.
    # Block (r,c) of A: rows 2r..2r+1, cols 2c..2c+1
    # In the flattened A (16 entries), block (r,c) starts at index (2r)*4 + 2c = 8r + 2c
    # The 4 entries of block (r,c) are at indices 8r+2c, 8r+2c+1, 8r+2c+2, 8r+2c+3
    
    # Similarly for B (4x4) and C (4x4).
    
    # For the 4x4x5 part, we need to handle the 5th column of B separately.
    # Actually, the task is 4x4 * 4x5 = 4x5.
    # So B is 4x5, C is 4x5.
    
    # Let's think of this as:
    # C[:, 0:4] = A * B[:, 0:4]  (4x4 * 4x4 = 4x4)
    # C[:, 4] = A * B[:, 4]      (4x4 * 4x1 = 4x1)
    
    # For the 4x4 part, we use 49 terms.
    # For the 4x1 part, we need 4*4*1 = 16 terms (schoolbook).
    
    # Total rank = 49 + 16 = 65.
    
    # Let's build this.
    
    # Strassen factors for 2x2:
    strassen_u = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_v = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    strassen_w = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [0, 1, 0, 0],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, -1]  # M7
    ]
    
    # Now build the 49 terms for 4x4x4.
    # For outer term o and inner term i:
    # The U vector is 16 entries (for A, 4x4).
    # The V vector is 16 entries (for B, 4x4).
    # The W vector is 16 entries (for C, 4x4).
    
    # For a given outer term o, we need to determine which blocks of A, B, C are involved.
    # Let's define the outer Strassen terms for 4x4:
    # The 4x4 matrix is divided into 4 blocks of size 2x2.
    # Let's label the blocks as:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Outer Strassen terms:
    # O1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # O2 = (A10+A11)B00 -> contributes to C10
    # O3 = A00(B10-B11) -> contributes to C01
    # O4 = A11(B01-B00) -> contributes to C00 and C10
    # O5 = (A00+A01)B11 -> contributes to -C00 and C01
    # O6 = (A10-A00)(B00+B01) -> contributes to C11
    # O7 = (A01-A11)(B10+B11) -> contributes to C00 and -C11
    
    # For each outer term, we have a linear combination of blocks.
    # For outer term o, let's define:
    # a_blocks: dict from block index (0..3) to coefficient
    # b_blocks: dict from block index (0..3) to coefficient
    # c_blocks: dict from block index (0..3) to coefficient
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    outer_terms = [
        # O1
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # O2
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1}},
        # O3
        {'a': {0: 1}, 'b': {2: 1, 3: -1}, 'c': {1: 1}},
        # O4
        {'a': {3: 1}, 'b': {1: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # O5
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # O6
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # O7
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1, 3: -1}},
    ]
    
    # Now build the 49 terms.
    U_44 = []
    V_44 = []
    W_44 = []
    
    for o in range(7):
        outer = outer_terms[o]
        for i in range(7):
            u_inner = strassen_u[i]
            v_inner = strassen_v[i]
            w_inner = strassen_w[i]
            
            # Build U vector (16 entries)
            u_global = [0] * 16
            for block, coeff_a in outer['a'].items():
                # Block (r,c) where r = block // 2, c = block % 2
                r = block // 2
                c = block % 2
                # The 4 entries of this block in the flattened A:
                # Index 0: (2r, 2c) -> 2r*4 + 2c = 8r + 2c
                # Index 1: (2r, 2c+1) -> 2r*4 + 2c+1 = 8r + 2c + 1
                # Index 2: (2r+1, 2c) -> (2r+1)*4 + 2c = 8r + 4 + 2c
                # Index 3: (2r+1, 2c+1) -> (2r+1)*4 + 2c+1 = 8r + 4 + 2c + 1
                base = 8 * r + 2 * c
                u_global[base] += coeff_a * u_inner[0]
                u_global[base + 1] += coeff_a * u_inner[1]
                u_global[base + 4] += coeff_a * u_inner[2]
                u_global[base + 5] += coeff_a * u_inner[3]
            
            # Build V vector (16 entries)
            v_global = [0] * 16
            for block, coeff_b in outer['b'].items():
                r = block // 2
                c = block % 2
                base = 8 * r + 2 * c
                v_global[base] += coeff_b * v_inner[0]
                v_global[base + 1] += coeff_b * v_inner[1]
                v_global[base + 4] += coeff_b * v_inner[2]
                v_global[base + 5] += coeff_b * v_inner[3]
            
            # Build W vector (16 entries)
            w_global = [0] * 16
            for block, coeff_c in outer['c'].items():
                r = block // 2
                c = block % 2
                base = 8 * r + 2 * c
                w_global[base] += coeff_c * w_inner[0]
                w_global[base + 1] += coeff_c * w_inner[1]
                w_global[base + 4] += coeff_c * w_inner[2]
                w_global[base + 5] += coeff_c * w_inner[3]
            
            U_44.append(u_global)
            V_44.append(v_global)
            W_44.append(w_global)
    
    # Now build the 16 terms for the 4x4x1 part.
    # C[:, 4] = A * B[:, 4]
    # For each i, k: C[i, 4] += A[i, j] * B[j, 4]
    # So for each (i, j, k=4), we have a term.
    # U: A[i, j] -> index i*4 + j
    # V: B[j, 4] -> index j*5 + 4
    # W: C[i, 4] -> index 4*4 + i = 16 + i (wait, C is 4x5, so C[i, k] -> i*5 + k)
    # Actually, C is 4x5, so C[i, k] -> i*5 + k
    # W: C[i, 4] -> i*5 + 4
    
    U_41 = []
    V_41 = []
    W_41 = []
    
    for i in range(4):
        for j in range(4):
            u = [0] * 16
            u[i * 4 + j] = 1
            
            v = [0] * 20
            v[j * 5 + 4] = 1
            
            w = [0] * 20
            w[i * 5 + 4] = 1
            
            U_41.append(u)
            V_41.append(v)
            W_41.append(w)
    
    # Combine: total rank = 49 + 16 = 65
    rank = 65
    U = [row + [0] * 4 for row in U_44] + [row + [0] * 4 for row in U_41]
    # Wait, U_44 has 16 entries, but U should have 16 entries (for A, 4x4).
    # U_41 also has 16 entries.
    # So U is just U_44 + U_41, each row has 16 entries.
    
    # Let me redo this.
    U = U_44 + U_41
    V = V_44 + V_41
    W = W_44 + W_41
    
    # Now verify.
    # Build the target tensor.
    T_target = np.zeros((16, 20, 20))
    for i in range(4):
        for j in range(4):
            for k in range(5):
                a = i * 4 + j
                b = j * 5 + k
                c = k * 4 + i
                T_target[a, b, c] = 1.0
    
    # Compute the actual tensor from U, V, W.
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    T_actual = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    error = np.sum((T_actual - T_target) ** 2)
    
    if error < 1e-10:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank 80
        rank_sb = 80
        U_sb = [[0] * rank_sb for _ in range(16)]
        V_sb = [[0] * rank_sb for _ in range(20)]
        W_sb = [[0] * rank_sb for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    a = i * 4 + j
                    b = j * 5 + k
                    c = k * 4 + i
                    U_sb[a][col] = 1
                    V_sb[b][col] = 1
                    W_sb[c][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
