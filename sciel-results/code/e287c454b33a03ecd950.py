import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 2, 2, 11
    rank = 39
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Define Strassen terms for 2x2 block
    # Each term is (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # Indices in U are i*m + j (i=0,1; j=0,1) -> 0,1,2,3
    # Indices in V are j*p + k (j=0,1; k=0,1) -> 0,1,11,12 (local to block)
    # Indices in W are k*n + i (k=0,1; i=0,1) -> 0,1,2,3 (local to block)
    
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # We will iterate over 5 blocks of 2 columns
    for block in range(5):
        k_offset = block * 2
        # V indices for this block: 
        # v_e = 0 * p + k_offset = k_offset
        # v_f = 0 * p + (k_offset+1) = k_offset + 1
        # v_g = 1 * p + k_offset = p + k_offset
        # v_h = 1 * p + (k_offset+1) = p + k_offset + 1
        
        # W indices for this block:
        # w_c00 = k_offset * n + 0 = 2 * k_offset
        # w_c10 = k_offset * n + 1 = 2 * k_offset + 1
        # w_c01 = (k_offset+1) * n + 0 = 2 * k_offset + 2
        # w_c11 = (k_offset+1) * n + 1 = 2 * k_offset + 3
        
        v_e = k_offset
        v_f = k_offset + 1
        v_g = p + k_offset
        v_h = p + k_offset + 1
        
        w_c00 = 2 * k_offset
        w_c10 = 2 * k_offset + 1
        w_c01 = 2 * k_offset + 2
        w_c11 = 2 * k_offset + 3
        
        # Term 1: m1
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # Term 2: m2
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # Term 3: m3
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # Term 4: m4
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # Term 5: m5
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # Term 6: m6
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # Term 7: m7
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
        
        if time.time() > deadline:
            # If time runs out, we still have a valid partial result? 
            # No, rank must be fixed. 
            # But this is just a safety guard.
            pass

    # Last column k=10
    if r < rank:
        k2 = 10
        v_e2 = k2
        v_g2 = p + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        # Schoolbook for 2x2 * 1x2 (effectively 2x2x1)
        # a * B[0, 10] -> C[0, 10]
        U[0][r] += 1
        V[v_e2][r] += 1
        W[w_c02][r] += 1
        r += 1
        
        # b * B[1, 10] -> C[0, 10]
        U[1][r] += 1
        V[v_g2][r] += 1
        W[w_c02][r] += 1
        r += 1
        
        # c * B[0, 10] -> C[1, 10]
        U[2][r] += 1
        V[v_e2][r] += 1
        W[w_c12][r] += 1
        r += 1
        
        # d * B[1, 10] -> C[1, 10]
        U[3][r] += 1
        V[v_g2][r] += 1
        W[w_c12][r] += 1
        r += 1

    return {"U": U, "V": V, "W": W}
