import time

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a rank-55 decomposition for matrix multiplication of dimensions 2x3 and 3x10.
    Strategy:
    Split the multiplication into two parts based on the middle dimension (size 3).
    Part 1: Multiply A[:, 0:2] (2x2) by B[0:2, :] (2x10).
            This consists of 5 independent 2x2x2 matrix multiplications (one for each pair of columns in B).
            Use Strassen's algorithm (rank 7) for each 2x2x2 multiplication.
            Total rank for Part 1: 5 * 7 = 35.
    Part 2: Multiply A[:, 2] (2x1) by B[2, :] (1x10).
            This is an outer product of a 2-vector and a 10-vector.
            Each of the 20 output entries requires 1 scalar multiplication.
            Total rank for Part 2: 20.
    Total rank: 35 + 20 = 55.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We will build the decomposition column by column.
    # U: (n*m, rank) -> (6, rank)
    # V: (m*p, rank) -> (30, rank)
    # W: (p*n, rank) -> (20, rank)
    
    # We don't know the final rank in advance for the list of lists, 
    # but we know it's 55. 
    # However, to be safe and flexible, we can append columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_column(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to create zero vectors
    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)
    
    # Indices for A (n x m): i * m + j
    # Indices for B (m x p): j * p + k
    # Indices for W (p x n trace): k * n + i
    
    # --- Part 1: Strassen for 2x2 x 2x10 ---
    # A_sub is A[:, 0:2] -> indices i*3 + 0, i*3 + 1
    # B_sub is B[0:2, :] -> indices 0*10 + k, 1*10 + k
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    
    # We have 5 pairs of columns in B: (0,1), (2,3), (4,5), (6,7), (8,9)
    for pair_idx in range(5):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        # Define indices for the 2x2 subproblem
        # A indices:
        # a = A[0,0] -> 0*3+0 = 0
        # b = A[0,1] -> 0*3+1 = 1
        # c = A[1,0] -> 1*3+0 = 3
        # d = A[1,1] -> 1*3+1 = 4
        
        # B indices:
        # e = B[0,k1] -> 0*10+k1
        # f = B[0,k2] -> 0*10+k2
        # g = B[1,k1] -> 1*10+k1
        # h = B[1,k2] -> 1*10+k2
        
        # W indices (k*n + i):
        # C11 -> C[0,k1] -> k1*2+0
        # C12 -> C[0,k2] -> k2*2+0
        # C21 -> C[1,k1] -> k1*2+1
        # C22 -> C[1,k2] -> k2*2+1
        
        w_11 = k1 * 2 + 0
        w_12 = k2 * 2 + 0
        w_21 = k1 * 2 + 1
        w_22 = k2 * 2 + 1
        
        idx_a = 0
        idx_b = 1
        idx_c = 3
        idx_d = 4
        
        idx_e = k1
        idx_f = k2
        idx_g = 10 + k1
        idx_h = 10 + k2
        
        # M1 = (a+d)(e+h)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[idx_a] = 1; u[idx_d] = 1
        v[idx_e] = 1; v[idx_h] = 1
        w[w_11] += 1; w[w_22] += 1
        add_column(u, v, w)
        
        # M2 = (c+d)e
        u = zero_u(); v = zero_v(); w = zero_w()
        u[idx_c] = 1; u[idx_d] = 1
        v[idx_e] = 1
        w[w_21] += 1; w[w_22] -= 1
        add_column(u, v, w)
        
        # M3 = a(f-h)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[idx_a] = 1
        v[idx_f] = 1; v[idx_h] = -1
        w[w_12] += 1; w[w_22] += 1
        add_column(u, v, w)
        
        # M4 = d(g-e)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[idx_d] = 1
        v[idx_g] = 1; v[idx_e] = -1
        w[w_11] -= 1; w[w_12] += 1
        add_column(u, v, w)
        
        # M5 = (a+b)h
        u = zero_u(); v = zero_v(); w = zero_w()
        u[idx_a] = 1; u[idx_b] = 1
        v[idx_h] = 1
        w[w_11] += 1; w[w_21] += 1
        add_column(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[idx_c] = 1; u[idx_a] = -1
        v[idx_e] = 1; v[idx_f] = 1
        w[w_22] += 1
        add_column(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[idx_b] = 1; u[idx_d] = -1
        v[idx_g] = 1; v[idx_h] = 1
        w[w_11] -= 1
        add_column(u, v, w)
        
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
        # Return whatever we have, though it will be invalid if incomplete
        # But the loop is very fast, so this is unlikely.
        pass

    # --- Part 2: Schoolbook for 2x1 x 1x10 ---
    # A_sub is A[:, 2] -> indices i*3 + 2
    # B_sub is B[2, :] -> indices 2*10 + k
    
    for i in range(n):
        for k in range(p):
            u = zero_u(); v = zero_v(); w = zero_w()
            # A index: i * 3 + 2
            u[i * 3 + 2] = 1
            # B index: 2 * 10 + k
            v[2 * 10 + k] = 1
            # W index: k * 2 + i
            w[k * 2 + i] = 1
            add_column(u, v, w)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert lists of columns to matrices (rows x cols)
    # The interface expects U, V, W to be of shapes (n*m, rank), etc.
    # My U_cols is a list of columns, each of length n*m.
    # So U is the transpose of the matrix formed by U_cols.
    
    rank = len(U_cols)
    
    # If we ran out of time, rank might be less than 55.
    # But 55 columns is trivial to generate.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
            
    return {"U": U, "V": V, "W": W}
