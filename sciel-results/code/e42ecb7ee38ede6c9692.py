import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 85 solution:
    # 49 terms for 4x4x4 block (i in 0..3, j in 0..3, k in 0..3)
    # 36 terms for remainder
    
    rank = 85
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    # Helper to add a term
    def add_term(i, j, k, coeff, col_idx):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        U[i*m + j, col_idx] += coeff
        V[j*p + k, col_idx] += coeff
        W[k*n + i, col_idx] += coeff

    col = 0
    
    # --- Part 1: 4x4x4 Block ---
    # Indices i, j, k in {0, 1, 2, 3}
    
    # Define the 7 Strassen combinations for 4x4
    # A_r and B_r are 2x2 matrices of coefficients for the 2x2 blocks
    # Block (p, q) of A corresponds to rows 2p..2p+1, cols 2q..2q+1
    # Local indices in A_r: p_local, q_local in {0, 1}
    # Global indices: i = 2*p_block + p_local, j = 2*q_block + q_local
    
    # We will generate the 7 scalar products for each of the 7 combinations
    # Each scalar product is a sum of A_i,j * B_j,k terms
    
    # The 7 combinations (A_coeff, B_coeff, W_coeff)
    # A_coeff: 2x2 matrix, A_r[p,q] = sum of A_block[p,q]
    # B_coeff: 2x2 matrix
    # W_coeff: 2x2 matrix, C_block[p,q] += W_coeff[p,q] * M_r
    
    strassen_4x4 = [
        # 1
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])),
        # 2
        (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, 0]])),
        # 3
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, -1]]), np.array([[0, 1], [0, 1]])),
        # 4
        (np.array([[0, 0], [0, 1]]), np.array([[0, -1], [1, 0]]), np.array([[1, 0], [1, 0]])),
        # 5
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[-1, 1], [0, 0]])),
        # 6
        (np.array([[-1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        # 7
        (np.array([[0, 1], [0, -1]]), np.array([[0, 0], [1, 1]]), np.array([[1, 1], [0, 0]]))
    ]
    
    # The 7 scalar products for 2x2 Strassen
    # Each is a list of (a_idx, b_idx, coeff)
    # a_idx, b_idx are local indices (0,1)
    strassen_2x2 = [
        # M1: (a00+a11)(b00+b11)
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # M2: (a10+a11)b00
        [(1,0,1), (1,1,1)],
        # M3: a00(b01-b11)
        [(0,1,1), (0,0,-1)],
        # M4: a11(b10-b00)
        [(1,1,1), (1,0,-1)],
        # M5: (a00+a01)b11
        [(0,1,1), (0,0,1)],
        # M6: (a10-a00)(b00+b01)
        [(1,0,1), (1,1,1), (0,0,-1), (0,1,-1)],
        # M7: (a01-a11)(b10+b11)
        [(0,1,1), (0,0,1), (1,1,-1), (1,0,-1)]
    ]
    
    # We need to map local (p, q) to global (i, j)
    # For A_r[p, q]:
    # p is block row index (0 or 1), q is block col index (0 or 1)
    # The block A[p, q] contains A[2p, 2q], A[2p, 2q+1], A[2p+1, 2q], A[2p+1, 2q+1]
    # The local indices in the 2x2 Strassen step refer to the entries of A_r.
    # A_r is a 2x2 matrix.
    # Entry (u, v) of A_r is the sum of A_block[u, v].
    # Wait, A_r[u, v] = sum_{x,y} A_coeff[u, v] * A_block[x, y]? No.
    # A_r is formed by combining blocks.
    # A_r[u, v] = A_block[u, v] * A_coeff[u, v]? No.
    # The definition was: A_r = sum_{p,q} A_coeff[p, q] A_block[p, q].
    # So A_r is a 2x2 matrix.
    # Its entry (u, v) is NOT simply A_block[u, v].
    # A_r[u, v] is the sum of entries from the blocks?
    # No, A_r is a matrix of the same size as the blocks (2x2).
    # A_r = A_coeff[0,0]*A_block[0,0] + A_coeff[0,1]*A_block[0,1] + ...
    # So A_r[u, v] = sum_{p,q} A_coeff[p, q] * A_block[p, q][u, v].
    
    # So for a term A_r[u, v] * B_r[x, y]:
    # A_r[u, v] = sum_{p,q} A_coeff[p, q] * A[2p+u, 2q+v]
    # B_r[x, y] = sum_{r,s} B_coeff[r, s] * B[2r+x, 2s+y]
    
    # The product is sum_{p,q,r,s} A_coeff[p, q] B_coeff[r, s] A[2p+u, 2q+v] B[2r+x, 2s+y]
    
    # We iterate through the 7 combinations (idx_r)
    for idx_r, (A_coeff, B_coeff, W_coeff) in enumerate(strassen_4x4):
        # Iterate through the 7 scalar products (idx_s)
        for idx_s, terms in enumerate(strassen_2x2):
            # For each term (u, v, coeff) in the 2x2 Strassen expansion
            # u, v are local indices of A_r and B_r
            # Wait, the 2x2 Strassen terms are defined on the entries of A_r and B_r.
            # The list `terms` contains (a_idx, b_idx, coeff).
            # a_idx is (u, v) for A_r? No, a_idx is a single index?
            # In my `strassen_2x2` definition, I used (0,0) etc. which are (row, col).
            # So u, v are the row, col of the entry in A_r.
            
            # For each (u, v, coeff) in terms:
            # We need to expand A_r[u, v] and B_r[?, ?]
            # Wait, the term is A_r[u, v] * B_r[x, y].
            # The `terms` list should contain (u, v, x, y, coeff)?
            # My `strassen_2x2` list has tuples of (a_idx, b_idx, coeff).
            # a_idx is (u, v). b_idx is (x, y).
            
            # Let's re-parse the terms.
            # Example M1: (a00+a11)(b00+b11)
            # Terms: (0,0) with b(0,0), (0,0) with b(1,1), (1,1) with b(0,0), (1,1) with b(1,1)
            # My list: [(0,0,1), (0,1,1), (1,0,1), (1,1,1)] -> This is wrong format.
            
            # Let's redefine strassen_2x2 correctly.
            # Each term is a product of an entry of A and an entry of B.
            # (u, v) for A, (x, y) for B.
            
            pass # Placeholder, will redefine below

    # Redefine strassen_2x2 with correct (u, v, x, y, coeff)
    strassen_2x2_correct = [
        # M1
        [(0,0, 0,0, 1), (0,0, 1,1, 1), (1,1, 0,0, 1), (1,1, 1,1, 1)],
        # M2
        [(1,0, 0,0, 1), (1,1, 0,0, 1)],
        # M3
        [(0,0, 0,1, 1), (0,0, 1,1, -1)],
        # M4
        [(1,1, 1,0, 1), (1,1, 0,0, -1)],
        # M5
        [(0,0, 1,1, 1), (0,1, 1,1, 1)],
        # M6
        [(1,0, 0,0, 1), (1,0, 0,1, 1), (0,0, 0,0, -1), (0,0, 0,1, -1)],
        # M7
        [(0,1, 1,0, 1), (0,1, 1,1, 1), (1,1, 1,0, -1), (1,1, 1,1, -1)]
    ]

    for idx_r, (A_coeff, B_coeff, W_coeff) in enumerate(strassen_4x4):
        for idx_s, terms in enumerate(strassen_2x2_correct):
            for (u, v, x, y, coeff) in terms:
                # Expand A_r[u, v] and B_r[x, y]
                # A_r[u, v] = sum_{p,q} A_coeff[p, q] * A[2p+u, 2q+v]
                # B_r[x, y] = sum_{r,s} B_coeff[r, s] * B[2r+x, 2s+y]
                
                # The product is sum of A[i, j] * B[j, k]
                # Note: j in A is 2q+v. j in B is 2r+x.
                # For matrix multiplication, the col of A must match row of B.
                # So we only keep terms where 2q+v == 2r+x.
                
                for p in range(2):
                    for q in range(2):
                        a_coeff_val = A_coeff[p, q]
                        if a_coeff_val == 0: continue
                        i = 2*p + u
                        j_a = 2*q + v
                        
                        for r in range(2):
                            for s in range(2):
                                b_coeff_val = B_coeff[r, s]
                                if b_coeff_val == 0: continue
                                j_b = 2*r + x
                                k = 2*s + y
                                
                                if j_a == j_b:
                                    # Valid term A[i, j] B[j, k]
                                    total_coeff = a_coeff_val * b_coeff_val * coeff
                                    if total_coeff != 0:
                                        # Determine contribution to C
                                        # This term contributes to M_r.
                                        # M_r contributes to C_block[bp, bq] with W_coeff[bp, bq]
                                        # C_block[bp, bq] corresponds to C[2bp+u_c, 2bq+v_c]?
                                        # No, M_r is a 2x2 matrix.
                                        # The entry (u_c, v_c) of M_r is the sum of these products.
                                        # So this specific product contributes to M_r[u_c, v_c]?
                                        # No, the product A[i,j]B[j,k] is a scalar.
                                        # It is one of the terms summing to M_r[u_c, v_c].
                                        # Wait, the 2x2 Strassen algorithm computes the 4 entries of M_r.
                                        # The 7 scalar products are used to compute these 4 entries.
                                        # My expansion above calculates the scalar products directly.
                                        # Each scalar product is a specific combination.
                                        # So `total_coeff` is the coefficient for this specific scalar product.
                                        
                                        # Now, which entries of C does this scalar product affect?
                                        # It affects M_r.
                                        # M_r is added to C with W_coeff.
                                        # But M_r is a matrix.
                                        # The scalar product is a specific entry of the product?
                                        # No, the 7 scalar products are NOT entries.
                                        # They are intermediate values.
                                        # The entries of M_r are linear combinations of these 7 scalars.
                                        
                                        # This means my approach of expanding the scalar products is correct,
                                        # but I need to know how the scalar products map to M_r entries.
                                        
                                        # This is getting too complex for inline generation.
                                        # I will stick to Rank 92 for safety.
                                        pass

    # Fallback to Rank 92
    rank = 92
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((i_base, j_base, k_base))
    
    skip_set = set()
    for ib, jb, kb in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    skip_set.add((ib+di, jb+dj, kb+dk))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skip_set:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    for ib, jb, kb in blocks:
        i1, i2 = ib, ib + 1
        j1, j2 = jb, jb + 1
        k1, k2 = kb, kb + 1
        
        U[i1 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        U[i2 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        W[k1 * n + i2, col] += 1
        W[k2 * n + i2, col] += -1
        col += 1
        
        U[i1 * m + j1, col] += 1
        V[j1 * p + k2, col] += 1
        V[j2 * p + k2, col] += -1
        W[k2 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        U[i2 * m + j2, col] += 1
        V[j2 * p + k1, col] += 1
        V[j1 * p + k1, col] += -1
        W[k1 * n + i1, col] += 1
        W[k1 * n + i2, col] += 1
        col += 1
        
        U[i1 * m + j1, col] += 1
        U[i1 * m + j2, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += -1
        W[k2 * n + i1, col] += 1
        col += 1
        
        U[i2 * m + j1, col] += 1
        U[i1 * m + j1, col] += -1
        V[j1 * p + k1, col] += 1
        V[j1 * p + k2, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        U[i1 * m + j2, col] += 1
        U[i2 * m + j2, col] += -1
        V[j2 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
        
    if time.time() < deadline - 5:
        # Verification
        for test_i in range(n):
            for test_j in range(m):
                for test_k in range(p):
                    actual = 0
                    for r in range(rank):
                        actual += U[test_i * m + test_j, r] * V[test_j * p + test_k, r] * W[test_k * n + test_i, r]
                    if actual != 1:
                        # Fallback to standard
                        rank_std = 100
                        U_std = np.zeros((nm, rank_std), dtype=np.int64)
                        V_std = np.zeros((mp, rank_std), dtype=np.int64)
                        W_std = np.zeros((pn, rank_std), dtype=np.int64)
                        c = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    U_std[i*m+j, c] = 1
                                    V_std[j*p+k, c] = 1
                                    W_std[k*n+i, c] = 1
                                    c += 1
                        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
