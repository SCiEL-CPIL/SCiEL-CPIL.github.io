import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    # Target: 4x5x6
    # Strategy: 4 blocks of 2x5x3, each with rank 28. Total rank 112.
    
    total_rank = 112
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define Strassen 2x2x2 terms
    # Local indices for 2x2x2: i,j,k in 0,1
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [A11, A12, A21, A22] -> indices 0,1,2,3
        # V: [B11, B12, B21, B22] -> indices 0,1,2,3
        # W: [C11, C12, C21, C22] -> indices 0,1,2,3
        
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]), # C11=+1, C22=-1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # C21=+1
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),   # C12=+1
        # M4
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 0, 1]),   # C11=+1, C22=+1
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),   # C11=-1
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # C22=+1
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # C11=+1
    ]
    
    # Helper to add a term to U, V, W
    # u_idx, v_idx, w_idx are local indices within the sub-block
    # offset_u, offset_v, offset_w are global offsets
    # coeff is the multiplier
    def add_term(col, u_local, v_local, w_local, coeff, u_off, v_off, w_off):
        if coeff != 0:
            U[u_local + u_off][col] += coeff
            V[v_local + v_off][col] += coeff
            W[w_local + w_off][col] += coeff

    col = 0
    
    # Loop over 4 blocks
    for block_i in range(2): # 0 for rows 0-1, 1 for rows 2-3
        for block_k in range(2): # 0 for cols 0-2, 1 for cols 3-5
            
            # Global offsets
            i_off = block_i * 2
            k_off = block_k * 3
            
            # Sub-block 2x5x3
            # A is 2x5, B is 5x3
            # Local A row i (0,1), col j (0..4) -> Global U index: (i_off+i)*m + j
            # Local B row j (0..4), col k (0..2) -> Global V index: j*p + (k_off+k)
            # Local C row i (0,1), col k (0..2) -> Global W index: (k_off+k)*n + (i_off+i)
            
            # Term Group 1: A1 B1a (2x2x2 Strassen)
            # A cols 0-1, B rows 0-1, B cols 0-1
            for t in range(7):
                u_c, v_c, w_c = strassen_terms[t]
                # u_c corresponds to A[0,0], A[0,1], A[1,0], A[1,1]
                # v_c corresponds to B[0,0], B[0,1], B[1,0], B[1,1]
                # w_c corresponds to C[0,0], C[0,1], C[1,0], C[1,1]
                
                # Map local 2x2 indices to 2x5x3 indices
                # A local (r, c) -> global (i_off+r, c)
                # B local (r, c) -> global (r, k_off+c)
                # C local (r, c) -> global (i_off+r, k_off+c)
                
                # A indices: 0->0, 1->1, 2->5, 3->6? No.
                # A is 2x2 block at cols 0-1.
                # Local A idx 0 (0,0) -> Global U (i_off+0)*m + 0
                # Local A idx 1 (0,1) -> Global U (i_off+0)*m + 1
                # Local A idx 2 (1,0) -> Global U (i_off+1)*m + 0
                # Local A idx 3 (1,1) -> Global U (i_off+1)*m + 1
                
                # B is 2x2 block at rows 0-1, cols 0-1.
                # Local B idx 0 (0,0) -> Global V 0*p + (k_off+0)
                # Local B idx 1 (0,1) -> Global V 0*p + (k_off+1)
                # Local B idx 2 (1,0) -> Global V 1*p + (k_off+0)
                # Local B idx 3 (1,1) -> Global V 1*p + (k_off+1)
                
                # C is 2x2 block at rows 0-1, cols 0-1.
                # Local C idx 0 (0,0) -> Global W (k_off+0)*n + (i_off+0)
                # Local C idx 1 (0,1) -> Global W (k_off+1)*n + (i_off+0)
                # Local C idx 2 (1,0) -> Global W (k_off+0)*n + (i_off+1)
                # Local C idx 3 (1,1) -> Global W (k_off+1)*n + (i_off+1)
                
                for local_idx, coeff in enumerate(u_c):
                    if coeff != 0:
                        r = local_idx // 2
                        c = local_idx % 2
                        g_u = (i_off + r) * m + c
                        U[g_u][col] += coeff
                        
                for local_idx, coeff in enumerate(v_c):
                    if coeff != 0:
                        r = local_idx // 2
                        c = local_idx % 2
                        g_v = r * p + (k_off + c)
                        V[g_v][col] += coeff
                        
                for local_idx, coeff in enumerate(w_c):
                    if coeff != 0:
                        r = local_idx // 2
                        c = local_idx % 2
                        g_w = (k_off + c) * n + (i_off + r)
                        W[g_w][col] += coeff
                        
                col += 1
                
            # Term Group 2: A1 B1b (2x2x1 Schoolbook)
            # A cols 0-1, B rows 0-1, B col 2
            # 4 terms
            for a_r in range(2):
                for b_r in range(2):
                    # A[a_r, 0..1] * B[b_r, 2] -> C[a_r, 2]
                    # Wait, B1b is 2x1. B rows 0-1, Col 2.
                    # A1 is 2x2.
                    # Product is 2x1.
                    # C[a_r, 2] += A[a_r, b_r] * B[b_r, 2]
                    # This is 4 products.
                    
                    # U: A[a_r, b_r] -> (i_off+a_r)*m + b_r
                    # V: B[b_r, 2] -> b_r*p + (k_off+2)
                    # W: C[a_r, 2] -> (k_off+2)*n + (i_off+a_r)
                    
                    U[(i_off + a_r) * m + b_r][col] = 1
                    V[b_r * p + (k_off + 2)][col] = 1
                    W[(k_off + 2) * n + (i_off + a_r)][col] = 1
                    col += 1
                    
            # Term Group 3: A2 B2a (2x2x2 Strassen)
            # A cols 2-3, B rows 2-3, B cols 0-1
            for t in range(7):
                u_c, v_c, w_c = strassen_terms[t]
                
                for local_idx, coeff in enumerate(u_c):
                    if coeff != 0:
                        r = local_idx // 2
                        c = local_idx % 2
                        # A col offset 2
                        g_u = (i_off + r) * m + (2 + c)
                        U[g_u][col] += coeff
                        
                for local_idx, coeff in enumerate(v_c):
                    if coeff != 0:
                        r = local_idx // 2
                        c = local_idx % 2
                        # B row offset 2
                        g_v = (2 + r) * p + (k_off + c)
                        V[g_v][col] += coeff
                        
                for local_idx, coeff in enumerate(w_c):
                    if coeff != 0:
                        r = local_idx // 2
                        c = local_idx % 2
                        g_w = (k_off + c) * n + (i_off + r)
                        W[g_w][col] += coeff
                        
                col += 1
                
            # Term Group 4: A2 B2b (2x2x1 Schoolbook)
            # A cols 2-3, B rows 2-3, B col 2
            for a_r in range(2):
                for b_r in range(2):
                    # A[a_r, 2+b_r] * B[2+b_r, 2] -> C[a_r, 2]
                    U[(i_off + a_r) * m + (2 + b_r)][col] = 1
                    V[(2 + b_r) * p + (k_off + 2)][col] = 1
                    W[(k_off + 2) * n + (i_off + a_r)][col] = 1
                    col += 1
                    
            # Term Group 5: A3 B3a (2x1x2 Schoolbook)
            # A col 4, B row 4, B cols 0-1
            # 2 terms? No, 2x1x2 means A(2x1) * B(1x2) -> C(2x2).
            # A is 2x1 (col 4). B is 1x2 (row 4, cols 0-1).
            # C[i, k] = A[i, 4] * B[4, k]
            # 2 rows for A, 2 cols for B -> 4 terms.
            for a_r in range(2):
                for b_c in range(2):
                    U[(i_off + a_r) * m + 4][col] = 1
                    V[4 * p + (k_off + b_c)][col] = 1
                    W[(k_off + b_c) * n + (i_off + a_r)][col] = 1
                    col += 1
                    
            # Term Group 6: A3 B3b (2x1x1 Schoolbook)
            # A col 4, B row 4, B col 2
            # 2 terms
            for a_r in range(2):
                U[(i_off + a_r) * m + 4][col] = 1
                V[4 * p + (k_off + 2)][col] = 1
                W[(k_off + 2) * n + (i_off + a_r)][col] = 1
                col += 1
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
