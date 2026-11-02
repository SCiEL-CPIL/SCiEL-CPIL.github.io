def construct():
    import itertools
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n

    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []

    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        # Append to existing rows or create new ones if needed?
        # It's easier to build lists of columns (terms) and then transpose.
        pass

    # We will build a list of tuples (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    # representing the rank-1 tensors.
    terms = []

    # Block 1: A1 (2x3) * B11 (3x3) -> C11 (2x3)
    # i in 0..1, j in 0..2, k in 0..2
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 2 + i
                terms.append((u_idx, 1, v_idx, 1, w_idx, 1))

    # Block 2: A1 (2x3) * B12 (3x2) -> C12 (2x2)
    # i in 0..1, j in 0..2, k in 3..4
    for i in range(2):
        for j in range(3):
            for k in range(3, 5):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 2 + i
                terms.append((u_idx, 1, v_idx, 1, w_idx, 1))

    # Block 3: A2 (2x2) * B21 (2x3) -> C21 (2x3)
    # i in 0..1, j in 3..4, k in 0..2
    for i in range(2):
        for j in range(3, 5):
            for k in range(3):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 2 + i
                terms.append((u_idx, 1, v_idx, 1, w_idx, 1))

    # Block 4: A2 (2x2) * B22 (2x2) -> C22 (2x2) using Strassen
    # i in 0..1, j in 3..4, k in 3..4
    # Local indices: i_l in 0..1, j_l in 0..1, k_l in 0..1
    # Global: i_g = i_l, j_g = 3 + j_l, k_g = 3 + k_l
    
    # Strassen formulas
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We need to express each M_t as a sum of rank-1 terms?
    # No, Strassen defines the linear combinations for the inputs and outputs.
    # The rank-1 terms are the 7 multiplications M_t.
    # For each M_t, we have a specific U vector, V vector, and W vector.
    
    # U vector for M_t: coefficients for A elements (i, j_g)
    # V vector for M_t: coefficients for B elements (j_g, k_g)
    # W vector for M_t: coefficients for C elements (k_g, i)
    
    # Let's define the 7 terms explicitly.
    # A elements: A[0,3], A[0,4], A[1,3], A[1,4]
    # B elements: B[3,3], B[3,4], B[4,3], B[4,4]
    # C elements: C[3,0], C[3,1], C[4,0], C[4,1] (Note: C index is k*n+i = k*2+i)
    # C[3,0] is k=3, i=0. C[3,1] is k=3, i=1.
    # C[4,0] is k=4, i=0. C[4,1] is k=4, i=1.
    
    # M1: (A[0,3]+A[1,4]) * (B[3,3]+B[4,4])
    # U: A[0,3]=1, A[1,4]=1
    # V: B[3,3]=1, B[4,4]=1
    # W: c00 += 1, c11 += 1
    # c00 is C[3,0] (k=3, i=0) -> idx 3*2+0 = 6
    # c11 is C[4,1] (k=4, i=1) -> idx 4*2+1 = 9
    terms.append((0*5+3, 1, 3*5+3, 1, 6, 1)) # This is wrong, terms are rank-1.
    # Wait, a rank-1 term is a single scalar multiplication.
    # Strassen replaces 8 scalar multiplications with 7 "complex" ones.
    # But in the tensor decomposition, each "complex" multiplication M_t is itself a rank-1 term in the *bilinear* form if we consider the linear combinations as part of the U and V matrices.
    # Yes! U, V, W can have coefficients.
    # So for M1, we add ONE term to the decomposition.
    # U has 1 at (0,3) and 1 at (1,4).
    # V has 1 at (3,3) and 1 at (4,4).
    # W has 1 at (3,0) and 1 at (4,1).
    
    # Let's re-verify the W indices.
    # C is 2x5.
    # c00 corresponds to row 0, col 3 of C? No.
    # C = A * B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # For Block 4, i is 0,1. k is 3,4.
    # c00 in local Strassen notation is C[0, 3] (i=0, k=3).
    # c01 in local Strassen notation is C[0, 4] (i=0, k=4).
    # c10 in local Strassen notation is C[1, 3] (i=1, k=3).
    # c11 in local Strassen notation is C[1, 4] (i=1, k=4).
    
    # Global W index is k*n + i = k*2 + i.
    # C[0, 3] -> k=3, i=0 -> idx 6.
    # C[0, 4] -> k=4, i=0 -> idx 8.
    # C[1, 3] -> k=3, i=1 -> idx 7.
    # C[1, 4] -> k=4, i=1 -> idx 9.
    
    # Let's define the 7 Strassen terms for Block 4.
    
    # M1: (A[0,3]+A[1,4]) * (B[3,3]+B[4,4])
    # U: (0,3):1, (1,4):1
    # V: (3,3):1, (4,4):1
    # W: C[0,3] (idx 6): +1, C[1,4] (idx 9): +1
    terms.append((0*5+3, 1, 3*5+3, 1, 6, 1)) # This format is for single indices.
    # I need a format that allows multiple indices in U, V, W for one term.
    
    # Let's change the structure of `terms` to hold dicts or lists of (idx, val).
    
    # Reset terms
    terms = []
    
    def add_block(n_sub, m_sub, p_sub, i_off, j_off, k_off, use_strassen=False):
        # n_sub, m_sub, p_sub are dimensions of the sub-block
        # i_off, j_off, k_off are offsets in the global matrices
        # A is n x m, B is m x p, C is p x n
        # Global A[i, j] is at i*m + j
        # Global B[j, k] is at j*p + k
        # Global C[k, i] is at k*n + i
        
        if use_strassen:
            if not (n_sub == 2 and m_sub == 2 and p_sub == 2):
                raise ValueError("Strassen only for 2x2x2")
            
            # Local indices 0,1
            # A_local[i_l, j_l] -> Global A[i_off+i_l, j_off+j_l]
            # B_local[j_l, k_l] -> Global B[j_off+j_l, k_off+k_l]
            # C_local[k_l, i_l] -> Global C[k_off+k_l, i_off+i_l]
            
            # M1
            u = {(i_off+0, j_off+0): 1, (i_off+1, j_off+1): 1}
            v = {(j_off+0, k_off+0): 1, (j_off+1, k_off+1): 1}
            w = {(k_off+0, i_off+0): 1, (k_off+1, i_off+1): 1}
            terms.append((u, v, w))
            
            # M2: (A[1,0]+A[1,1]) * B[0,0]
            u = {(i_off+1, j_off+0): 1, (i_off+1, j_off+1): 1}
            v = {(j_off+0, k_off+0): 1}
            w = {(k_off+0, i_off+1): 1, (k_off+1, i_off+1): -1}
            terms.append((u, v, w))
            
            # M3: A[0,0] * (B[1,0]-B[1,1])
            u = {(i_off+0, j_off+0): 1}
            v = {(j_off+1, k_off+0): 1, (j_off+1, k_off+1): -1}
            w = {(k_off+0, i_off+0): 1, (k_off+1, i_off+0): 1}
            terms.append((u, v, w))
            
            # M4: A[1,1] * (B[0,1]-B[0,0])
            u = {(i_off+1, j_off+1): 1}
            v = {(j_off+0, k_off+1): 1, (j_off+0, k_off+0): -1}
            w = {(k_off+0, i_off+0): 1, (k_off+0, i_off+1): 1}
            terms.append((u, v, w))
            
            # M5: (A[0,0]+A[0,1]) * B[1,1]
            u = {(i_off+0, j_off+0): 1, (i_off+0, j_off+1): 1}
            v = {(j_off+1, k_off+1): 1}
            w = {(k_off+0, i_off+0): -1, (k_off+1, i_off+0): 1}
            terms.append((u, v, w))
            
            # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
            u = {(i_off+1, j_off+0): 1, (i_off+0, j_off+0): -1}
            v = {(j_off+0, k_off+0): 1, (j_off+0, k_off+1): 1}
            w = {(k_off+1, i_off+1): 1}
            terms.append((u, v, w))
            
            # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
            u = {(i_off+0, j_off+1): 1, (i_off+1, j_off+1): -1}
            v = {(j_off+1, k_off+0): 1, (j_off+1, k_off+1): 1}
            w = {(k_off+0, i_off+0): 1, (k_off+1, i_off+1): 1}
            terms.append((u, v, w))
            
        else:
            for i_l in range(n_sub):
                for j_l in range(m_sub):
                    for k_l in range(p_sub):
                        u = {(i_off+i_l, j_off+j_l): 1}
                        v = {(j_off+j_l, k_off+k_l): 1}
                        w = {(k_off+k_l, i_off+i_l): 1}
                        terms.append((u, v, w))

    # Block 1: A1 (2x3) * B11 (3x3)
    # i_off=0, j_off=0, k_off=0
    add_block(2, 3, 3, 0, 0, 0)
    
    # Block 2: A1 (2x3) * B12 (3x2)
    # i_off=0, j_off=0, k_off=3
    add_block(2, 3, 2, 0, 0, 3)
    
    # Block 3: A2 (2x2) * B21 (2x3)
    # i_off=0, j_off=3, k_off=0
    add_block(2, 2, 3, 0, 3, 0)
    
    # Block 4: A2 (2x2) * B22 (2x2)
    # i_off=0, j_off=3, k_off=3
    add_block(2, 2, 2, 0, 3, 3, use_strassen=True)
    
    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    
    for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][t_idx] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][t_idx] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][t_idx] = val
            
    return {"U": U, "V": V, "W": W}
