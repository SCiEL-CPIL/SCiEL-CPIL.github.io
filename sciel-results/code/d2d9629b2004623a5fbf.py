import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: i * m + j
    # v_idx: j * p + k
    # w_idx: k * n + i
    def add_term(u, v, w, val=1):
        U[u][term_idx] += val
        V[v][term_idx] += val
        W[w][term_idx] += val

    term_idx = 0
    
    # 1. Strassen for A1 (2x2) * B1 (2x2) -> C11 (2x2)
    # A1: i in {0,1}, j in {0,1}
    # B1: j in {0,1}, k in {0,1}
    # C11: i in {0,1}, k in {0,1}
    
    # M1 = (A11+A22)(B11+B22)
    # A11: u=0*3+0=0, A22: u=1*3+1=4
    # B11: v=0*4+0=0, B22: v=1*4+1=5
    # C11: w=0*3+0=0, C22: w=1*3+1=4
    # C12: w=0*3+1=1, C21: w=1*3+0=3
    
    # M1
    add_term(0, 0, 0, 1) # A11 B11 C11
    add_term(0, 5, 0, 1) # A11 B22 C11
    add_term(4, 0, 0, 1) # A22 B11 C11
    add_term(4, 5, 0, 1) # A22 B22 C11
    # Wait, Strassen combines sums in U and V.
    # U vector for M1: A11 + A22. V vector: B11 + B22.
    # W vector: C11 + C22 - C12 + C21? No.
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # Let's just build the 7 rank-1 tensors for Strassen directly.
    # T1: (A11+A22) (B11+B22) (C11+C22-C12+C21) ? No, the W part is the output coefficients.
    # The decomposition is sum_r U_r V_r W_r.
    # For Strassen:
    # r=1: U=(A11+A22), V=(B11+B22), W=(C11+C22-C12+C21) is WRONG.
    # W_r is the vector of coefficients for C.
    # M1 contributes to:
    # C11: +1
    # C22: +1
    # C12: -1
    # C21: +1
    # So W1 has 1 at C11, 1 at C22, -1 at C12, 1 at C21.
    
    def add_strassen_block(a_off, b_off, c_off, start_idx):
        # a_off: (i_start, j_start)
        # b_off: (j_start, k_start)
        # c_off: (i_start, k_start)
        
        i_s, j_s = a_off
        j_b, k_s = b_off
        i_c, k_c = c_off
        
        # Indices in U, V, W
        # U: i, j
        # V: j, k
        # W: k, i
        
        # M1: (A11+A22)(B11+B22)
        # U: A11(i_s,j_s), A22(i_s+1,j_s+1)
        # V: B11(j_s,k_s), B22(j_s+1,k_s+1)
        # W: C11(k_s,i_s), C22(k_s+1,i_s+1), C12(k_s+1,i_s), C21(k_s,i_s+1)
        
        # Term 1
        U[i_s*m + j_s][start_idx] += 1
        U[(i_s+1)*m + (j_s+1)][start_idx] += 1
        V[j_s*p + k_s][start_idx] += 1
        V[(j_s+1)*p + (k_s+1)][start_idx] += 1
        W[k_s*n + i_s][start_idx] += 1      # C11
        W[(k_s+1)*n + (i_s+1)][start_idx] += 1 # C22
        W[(k_s+1)*n + i_s][start_idx] -= 1   # C12
        W[k_s*n + (i_s+1)][start_idx] += 1   # C21
        
        # M2: (A21+A22)B11
        # U: A21(i_s+1,j_s), A22(i_s+1,j_s+1)
        # V: B11(j_s,k_s)
        # W: C21(k_s,i_s+1), C22(k_s+1,i_s+1)
        
        U[(i_s+1)*m + j_s][start_idx+1] += 1
        U[(i_s+1)*m + (j_s+1)][start_idx+1] += 1
        V[j_s*p + k_s][start_idx+1] += 1
        W[k_s*n + (i_s+1)][start_idx+1] += 1   # C21
        W[(k_s+1)*n + (i_s+1)][start_idx+1] += 1 # C22
        
        # M3: A11(B12-B22)
        # U: A11(i_s,j_s)
        # V: B12(j_s,k_s+1), B22(j_s+1,k_s+1)
        # W: C12(k_s+1,i_s), C22(k_s+1,i_s+1)
        
        U[i_s*m + j_s][start_idx+2] += 1
        V[j_s*p + (k_s+1)][start_idx+2] += 1
        V[(j_s+1)*p + (k_s+1)][start_idx+2] -= 1
        W[(k_s+1)*n + i_s][start_idx+2] += 1   # C12
        W[(k_s+1)*n + (i_s+1)][start_idx+2] += 1 # C22
        
        # M4: A22(B21-B11)
        # U: A22(i_s+1,j_s+1)
        # V: B21(j_s+1,k_s), B11(j_s,k_s)
        # W: C11(k_s,i_s), C21(k_s,i_s+1)
        
        U[(i_s+1)*m + (j_s+1)][start_idx+3] += 1
        V[(j_s+1)*p + k_s][start_idx+3] += 1
        V[j_s*p + k_s][start_idx+3] -= 1
        W[k_s*n + i_s][start_idx+3] += 1      # C11
        W[k_s*n + (i_s+1)][start_idx+3] += 1   # C21
        
        # M5: (A11+A12)B22
        # U: A11(i_s,j_s), A12(i_s,j_s+1)
        # V: B22(j_s+1,k_s+1)
        # W: C12(k_s+1,i_s), C22(k_s+1,i_s+1)
        
        U[i_s*m + j_s][start_idx+4] += 1
        U[i_s*m + (j_s+1)][start_idx+4] += 1
        V[(j_s+1)*p + (k_s+1)][start_idx+4] += 1
        W[(k_s+1)*n + i_s][start_idx+4] += 1   # C12
        W[(k_s+1)*n + (i_s+1)][start_idx+4] += 1 # C22
        
        # M6: (A21-A11)(B11+B12)
        # U: A21(i_s+1,j_s), A11(i_s,j_s)
        # V: B11(j_s,k_s), B12(j_s,k_s+1)
        # W: C22(k_s+1,i_s+1)
        
        U[(i_s+1)*m + j_s][start_idx+5] += 1
        U[i_s*m + j_s][start_idx+5] -= 1
        V[j_s*p + k_s][start_idx+5] += 1
        V[j_s*p + (k_s+1)][start_idx+5] += 1
        W[(k_s+1)*n + (i_s+1)][start_idx+5] += 1 # C22
        
        # M7: (A12-A22)(B21+B22)
        # U: A12(i_s,j_s+1), A22(i_s+1,j_s+1)
        # V: B21(j_s+1,k_s), B22(j_s+1,k_s+1)
        # W: C11(k_s,i_s)
        
        U[i_s*m + (j_s+1)][start_idx+6] += 1
        U[(i_s+1)*m + (j_s+1)][start_idx+6] -= 1
        V[(j_s+1)*p + k_s][start_idx+6] += 1
        V[(j_s+1)*p + (k_s+1)][start_idx+6] += 1
        W[k_s*n + i_s][start_idx+6] += 1      # C11
        
        return start_idx + 7

    # Block 1: A1(0,0) B1(0,0) -> C11(0,0)
    term_idx = add_strassen_block((0,0), (0,0), (0,0), 0)
    
    # Block 2: A2(0,2) B3(2,0) -> C11(0,0)
    # A2 is 2x1 (i 0-1, j 2). B3 is 1x2 (j 2, k 0-1). C11 is 2x2.
    # Rank 4.
    # C_ik = A_i2 B_2k
    for i in range(2):
        for k in range(2):
            u = i*m + 2
            v = 2*p + k
            w = k*n + i
            U[u][term_idx] = 1
            V[v][term_idx] = 1
            W[w][term_idx] = 1
            term_idx += 1

    # Block 3: A1(0,0) B2(0,2) -> C12(0,2)
    # B2 is 2x2 (j 0-1, k 2-3). C12 is 2x2 (i 0-1, k 2-3).
    term_idx = add_strassen_block((0,0), (0,2), (0,2), term_idx)
    
    # Block 4: A2(0,2) B4(2,2) -> C12(0,2)
    # A2 is 2x1. B4 is 1x2 (j 2, k 2-3). C12 is 2x2.
    for i in range(2):
        for k in range(2):
            u = i*m + 2
            v = 2*p + (2+k)
            w = (2+k)*n + i
            U[u][term_idx] = 1
            V[v][term_idx] = 1
            W[w][term_idx] = 1
            term_idx += 1

    # Block 5: A3(2,0) B1(0,0) -> C21(2,0)
    # A3 is 1x2 (i 2, j 0-1). B1 is 2x2. C21 is 1x2 (i 2, k 0-1).
    # C_k = sum_j A_2j B_jk
    # Rank 4.
    for j in range(2):
        for k in range(2):
            u = 2*m + j
            v = j*p + k
            w = k*n + 2
            U[u][term_idx] = 1
            V[v][term_idx] = 1
            W[w][term_idx] = 1
            term_idx += 1

    # Block 6: A3(2,0) B2(0,2) -> C22(2,2)
    # A3 is 1x2. B2 is 2x2 (j 0-1, k 2-3). C22 is 1x2 (i 2, k 2-3).
    for j in range(2):
        for k in range(2):
            u = 2*m + j
            v = j*p + (2+k)
            w = (2+k)*n + 2
            U[u][term_idx] = 1
            V[v][term_idx] = 1
            W[w][term_idx] = 1
            term_idx += 1

    # Block 7: A4(2,2) B3(2,0) -> C21(2,0)
    # A4 is 1x1. B3 is 1x2 (j 2, k 0-1). C21 is 1x2.
    # C_k = A_22 B_2k
    for k in range(2):
        u = 2*m + 2
        v = 2*p + k
        w = k*n + 2
        U[u][term_idx] = 1
        V[v][term_idx] = 1
        W[w][term_idx] = 1
        term_idx += 1

    # Block 8: A4(2,2) B4(2,2) -> C22(2,2)
    # A4 is 1x1. B4 is 1x2 (j 2, k 2-3). C22 is 1x2.
    for k in range(2):
        u = 2*m + 2
        v = 2*p + (2+k)
        w = (2+k)*n + 2
        U[u][term_idx] = 1
        V[v][term_idx] = 1
        W[w][term_idx] = 1
        term_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
