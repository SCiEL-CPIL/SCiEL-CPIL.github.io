"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 56
    # 2 blocks of (3x2x5).
    # Each 3x2x5 splits into:
    # - 2x2x2 (Strassen, rank 7) for k=0,1
    # - 2x2x2 (Strassen, rank 7) for k=2,3
    # - 2x2x1 (Schoolbook, rank 4) for k=4
    # - 1x2x2 (Schoolbook, rank 4) for k=0,1
    # - 1x2x2 (Schoolbook, rank 4) for k=2,3
    # - 1x2x1 (Schoolbook, rank 2) for k=4
    # Total per 3x2x5: 7+7+4+4+4+2 = 28.
    # Total rank: 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    def add_strassen_2x2x2(u_indices, v_indices, w_indices):
        # u_indices: [u00, u01, u10, u11]
        # v_indices: [v00, v01, v10, v11]
        # w_indices: [w00, w01, w10, w11]
        nonlocal col
        
        # 1: (a00+a11)(v00+v11) -> w00+w11
        add_term(u_indices[0], v_indices[0], w_indices[0], 1)
        add_term(u_indices[0], v_indices[1], w_indices[0], 1) # Wait, Strassen terms are products.
        # The add_term function adds a rank-1 term: U[u,c]*V[v,c]*W[w,c].
        # Strassen term 1: (A00+A11)*(B00+B11) contributes to C00 and C11.
        # This is not a single rank-1 term in the tensor sense if we expand.
        # Tensor decomposition: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
        # Strassen exploits this.
        # Term 1: U part: A00+A11. V part: B00+B11. W part: C00+C11.
        # This requires 3 rank-1 terms? No.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is not a single rank 1 tensor.
        # Strassen works by defining new variables M1..M7.
        # M1 = (A00+A11)B00.
        # M2 = A11(B01-B00).
        # ...
        # Each M is a product of a linear combination of A and a linear combination of B.
        # In the tensor decomposition T = sum U V W, each term r corresponds to a "multiplication" M_r.
        # U[:, r] is the coefficients of A in M_r.
        # V[:, r] is the coefficients of B in M_r.
        # W[:, r] is the coefficients of C in M_r.
        
        # So for Strassen 2x2x2:
        # M1 = (A00+A11) * B00. 
        # U: A00:1, A11:1. V: B00:1. W: C00:1, C11:1.
        # This is NOT a valid rank-1 term in the standard U*V*W form because W has two non-zeros?
        # Yes, it is. U is a vector, V is a vector, W is a vector.
        # The term is sum_i sum_j sum_k U[i,r] V[j,r] W[k,r] e_i e_j e_k.
        # If W has two non-zeros, it adds to two positions in C. That's fine.
        
        # Let's re-verify the Strassen terms for 2x2x2.
        # M1 = (A00+A11)B00 -> C00 += M1, C11 += M1.
        # U: [1, 0, 0, 1] (A00, A01, A10, A11)
        # V: [1, 0, 0, 0] (B00, B01, B10, B11)
        # W: [1, 0, 0, 1] (C00, C01, C10, C11)
        
        # M2 = A11(B01-B00) -> C10 += M2, C11 -= M2.
        # U: [0, 0, 0, 1]
        # V: [-1, 1, 0, 0]
        # W: [0, 0, 1, -1]
        
        # M3 = A00(B11-B10) -> C01 += M3, C00 -= M3.
        # U: [1, 0, 0, 0]
        # V: [0, 0, -1, 1]
        # W: [-1, 1, 0, 0]
        
        # M4 = (A10+A00)B10 -> C10 += M4, C00 += M4.
        # U: [1, 0, 1, 0]
        # V: [0, 0, 1, 0]
        # W: [1, 0, 1, 0]
        
        # M5 = (A01-A10)(B10+B11) -> C11 += M5, C01 -= M5.
        # U: [0, 1, -1, 0]
        # V: [0, 0, 1, 1]
        # W: [0, -1, 0, 1]
        
        # M6 = (A11-A01)(B00+B01) -> C10 += M6, C11 += M6.
        # U: [0, -1, 0, 1]
        # V: [1, 1, 0, 0]
        # W: [0, 0, 1, 1]
        
        # M7 = (A00-A11)(B11-B10) -> C01 += M7, C11 -= M7.
        # U: [1, 0, 0, -1]
        # V: [0, 0, -1, 1]
        # W: [0, 1, 0, -1]
        
        # Let's check M1:
        # (A00+A11)B00 = A00B00 + A11B00.
        # C00 gets A00B00 + ...
        # C11 gets A11B00 + ...
        # Standard Strassen:
        # M1 = (A00+A11)B00. C00 += M1, C11 += M1.
        # M2 = A11(B01-B00). C10 += M2, C11 -= M2.
        # M3 = A00(B11-B10). C01 += M3, C00 -= M3.
        # M4 = (A10+A00)B10. C10 += M4, C00 += M4.
        # M5 = (A01-A10)(B10+B11). C11 += M5, C01 -= M5.
        # M6 = (A11-A01)(B00+B01). C10 += M6, C11 += M6.
        # M7 = (A00-A11)(B11-B10). C01 += M7, C11 -= M7.
        
        # Let's add these 7 terms.
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1]),
            ([0, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 0, -1, 1], [-1, 1, 0, 0]),
            ([1, 0, 1, 0], [0, 0, 1, 0], [1, 0, 1, 0]),
            ([0, 1, -1, 0], [0, 0, 1, 1], [0, -1, 0, 1]),
            ([0, -1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]),
            ([1, 0, 0, -1], [0, 0, -1, 1], [0, 1, 0, -1])
        ]
        
        for u_vals, v_vals, w_vals in terms:
            for u_idx, val in zip(u_indices, u_vals):
                if val != 0:
                    U[u_idx][col] = val
            for v_idx, val in zip(v_indices, v_vals):
                if val != 0:
                    V[v_idx][col] = val
            for w_idx, val in zip(w_indices, w_vals):
                if val != 0:
                    W[w_idx][col] = val
            col += 1

    def add_schoolbook(u_indices, v_indices, w_indices):
        # u_indices: list of U indices for A elements
        # v_indices: list of V indices for B elements
        # w_indices: list of W indices for C elements
        # Assumes 1-to-1 mapping for schoolbook.
        # For n x m x p schoolbook, we need n*m*p terms.
        # But here we are passing specific subsets.
        # This helper is for small blocks where we iterate all i,j,k.
        pass # We will use add_term directly for schoolbook parts to be safe.

    # Helper to add a generic schoolbook block
    def add_schoolbook_block(n_loc, m_loc, p_loc, u_map, v_map, w_map):
        # u_map[i,j] = global U index
        # v_map[j,k] = global V index
        # w_map[k,i] = global W index
        nonlocal col
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    u_idx = u_map[i][j]
                    v_idx = v_map[j][k]
                    w_idx = w_map[k][i]
                    add_term(u_idx, v_idx, w_idx, 1)

    # Block 1: j in {0, 1}
    # Sub-block 1.1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 Strassen
    u_11 = [0, 1, 4, 5] # i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1
    v_11 = [0, 1, 5, 6] # j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
    w_11 = [0, 1, 3, 4] # k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
    add_strassen_2x2x2(u_11, v_11, w_11)

    # Sub-block 1.2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2 Strassen
    u_12 = [0, 1, 4, 5]
    v_12 = [2, 3, 7, 8] # j=0,k=2; j=0,k=3; j=1,k=2; j=1,k=3
    w_12 = [6, 7, 9, 10] # k=2,i=0; k=2,i=1; k=3,i=0; k=3,i=1
    add_strassen_2x2x2(u_12, v_12, w_12)

    # Sub-block 1.3: i in {0,1}, j in {0,1}, k in {4} -> 2x2x1 Schoolbook
    # i=0,1; j=0,1; k=4
    u_13 = [0, 1, 4, 5]
    v_13 = [4, 9] # j=0,k=4; j=1,k=4
    w_13 = [12, 13] # k=4,i=0; k=4,i=1
    # Schoolbook:
    # i=0, j=0, k=4: U[0], V[4], W[12]
    # i=0, j=1, k=4: U[1], V[9], W[12]
    # i=1, j=0, k=4: U[4], V[4], W[13]
    # i=1, j=1, k=4: U[5], V[9], W[13]
    add_term(0, 4, 12, 1)
    add_term(1, 9, 12, 1)
    add_term(4, 4, 13, 1)
    add_term(5, 9, 13, 1)

    # Sub-block 1.4: i in {2}, j in {0,1}, k in {0,1} -> 1x2x2 Schoolbook
    # i=2; j=0,1; k=0,1
    # U: i=2, j=0 -> 8; i=2, j=1 -> 9
    # V: j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 5; j=1,k=1 -> 6
    # W: k=0,i=2 -> 2; k=1,i=2 -> 5
    add_term(8, 0, 2, 1) # i=2, j=0, k=0
    add_term(8, 1, 5, 1) # i=2, j=0, k=1
    add_term(9, 5, 2, 1) # i=2, j=1, k=0
    add_term(9, 6, 5, 1) # i=2, j=1, k=1

    # Sub-block 1.5: i in {2}, j in {0,1}, k in {2,3} -> 1x2x2 Schoolbook
    # i=2; j=0,1; k=2,3
    # U: 8, 9
    # V: j=0,k=2 -> 2; j=0,k=3 -> 3; j=1,k=2 -> 7; j=1,k=3 -> 8
    # W: k=2,i=2 -> 8; k=3,i=2 -> 11
    add_term(8, 2, 8, 1)
    add_term(8, 3, 11, 1)
    add_term(9, 7, 8, 1)
    add_term(9, 8, 11, 1)

    # Sub-block 1.6: i in {2}, j in {0,1}, k in {4} -> 1x2x1 Schoolbook
    # i=2; j=0,1; k=4
    # U: 8, 9
    # V: j=0,k=4 -> 4; j=1,k=4 -> 9
    # W: k=4,i=2 -> 14
    add_term(8, 4, 14, 1)
    add_term(9, 9, 14, 1)

    # Block 2: j in {2, 3}
    # Sub-block 2.1: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2 Strassen
    u_21 = [2, 3, 6, 7] # i=0,j=2; i=0,j=3; i=1,j=2; i=1,j=3
    v_21 = [10, 11, 15, 16] # j=2,k=0; j=2,k=1; j=3,k=0; j=3,k=1
    w_21 = [0, 1, 3, 4] # k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
    add_strassen_2x2x2(u_21, v_21, w_21)

    # Sub-block 2.2: i in {0,1}, j in {2,3}, k in {2,3} -> 2x2x2 Strassen
    u_22 = [2, 3, 6, 7]
    v_22 = [12, 13, 17, 18] # j=2,k=2; j=2,k=3; j=3,k=2; j=3,k=3
    w_22 = [6, 7, 9, 10]
    add_strassen_2x2x2(u_22, v_22, w_22)

    # Sub-block 2.3: i in {0,1}, j in {2,3}, k in {4} -> 2x2x1 Schoolbook
    u_23 = [2, 3, 6, 7]
    v_23 = [14, 19] # j=2,k=4; j=3,k=4
    w_23 = [12, 13]
    add_term(2, 14, 12, 1) # i=0, j=2, k=4
    add_term(3, 19, 12, 1) # i=0, j=3, k=4
    add_term(6, 14, 13, 1) # i=1, j=2, k=4
    add_term(7, 19, 13, 1) # i=1, j=3, k=4

    # Sub-block 2.4: i in {2}, j in {2,3}, k in {0,1} -> 1x2x2 Schoolbook
    # i=2; j=2,3; k=0,1
    # U: i=2, j=2 -> 10; i=2, j=3 -> 11
    # V: j=2,k=0 -> 10; j=2,k=1 -> 11; j=3,k=0 -> 15; j=3,k=1 -> 16
    # W: k=0,i=2 -> 2; k=1,i=2 -> 5
    add_term(10, 10, 2, 1)
    add_term(10, 11, 5, 1)
    add_term(11, 15, 2, 1)
    add_term(11, 16, 5, 1)

    # Sub-block 2.5: i in {2}, j in {2,3}, k in {2,3} -> 1x2x2 Schoolbook
    # i=2; j=2,3; k=2,3
    # U: 10, 11
    # V: j=2,k=2 -> 12; j=2,k=3 -> 13; j=3,k=2 -> 17; j=3,k=3 -> 18
    # W: k=2,i=2 -> 8; k=3,i=2 -> 11
    add_term(10, 12, 8, 1)
    add_term(10, 13, 11, 1)
    add_term(11, 17, 8, 1)
    add_term(11, 18, 11, 1)

    # Sub-block 2.6: i in {2}, j in {2,3}, k in {4} -> 1x2x1 Schoolbook
    # i=2; j=2,3; k=4
    # U: 10, 11
    # V: j=2,k=4 -> 14; j=3,k=4 -> 19
    # W: k=4,i=2 -> 14
    add_term(10, 14, 14, 1)
    add_term(11, 19, 14, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
