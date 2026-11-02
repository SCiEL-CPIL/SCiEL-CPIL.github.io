"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 53: 7 Strassen blocks of rank 7 + 1 schoolbook block of rank 4
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Process 7 blocks of 2 columns each
    for l in range(7):
        # Columns in B are 2*l and 2*l+1
        k1 = 2 * l
        k2 = 2 * l + 1
        
        # Indices in V (m*p = 30)
        # B[0, k] is at index k
        # B[1, k] is at index 15 + k
        v_0_k1 = k1
        v_1_k1 = 15 + k1
        v_0_k2 = k2
        v_1_k2 = 15 + k2
        
        # Indices in W (p*n = 30)
        # C[i, k] is at index k*n + i = k*2 + i
        w_0_k1 = k1 * 2 + 0
        w_1_k1 = k1 * 2 + 1
        w_0_k2 = k2 * 2 + 0
        w_1_k2 = k2 * 2 + 1
        
        # Term 1: M1 = (A00 + A11) * (B0k1 + B1k2)
        # U: A00, A11
        U[0][col_idx] += 1
        U[3][col_idx] += 1
        # V: B0k1, B1k2
        V[v_0_k1][col_idx] += 1
        V[v_1_k2][col_idx] += 1
        # W: C0k1, C1k2
        W[w_0_k1][col_idx] += 1
        W[w_1_k2][col_idx] += 1
        col_idx += 1
        
        # Term 2: M2 = (A10 + A11) * B0k1
        U[2][col_idx] += 1
        U[3][col_idx] += 1
        V[v_0_k1][col_idx] += 1
        W[w_1_k1][col_idx] += 1
        col_idx += 1
        
        # Term 3: M3 = A00 * (B0k2 - B1k2)
        U[0][col_idx] += 1
        V[v_0_k2][col_idx] += 1
        V[v_1_k2][col_idx] -= 1
        W[w_0_k2][col_idx] += 1
        col_idx += 1
        
        # Term 4: M4 = A11 * (B1k1 - B0k1)
        U[3][col_idx] += 1
        V[v_1_k1][col_idx] += 1
        V[v_0_k1][col_idx] -= 1
        W[w_0_k1][col_idx] += 1
        W[w_1_k1][col_idx] += 1
        col_idx += 1
        
        # Term 5: M5 = (A00 + A01) * B1k2
        U[0][col_idx] += 1
        U[1][col_idx] += 1
        V[v_1_k2][col_idx] += 1
        W[w_0_k1][col_idx] -= 1
        W[w_0_k2][col_idx] += 1
        col_idx += 1
        
        # Term 6: M6 = (A10 - A00) * (B0k1 + B0k2)
        U[2][col_idx] += 1
        U[0][col_idx] -= 1
        V[v_0_k1][col_idx] += 1
        V[v_0_k2][col_idx] += 1
        W[w_1_k2][col_idx] += 1
        col_idx += 1
        
        # Term 7: M7 = (A01 - A11) * (B1k1 + B1k2)
        U[1][col_idx] += 1
        U[3][col_idx] -= 1
        V[v_1_k1][col_idx] += 1
        V[v_1_k2][col_idx] += 1
        W[w_0_k1][col_idx] += 1
        col_idx += 1

    # Process the last column (index 14) with schoolbook (4 terms)
    k_last = 14
    v_0_last = k_last
    v_1_last = 15 + k_last
    w_0_last = k_last * 2 + 0
    w_1_last = k_last * 2 + 1
    
    # Term 8: A00 * B0_last -> C0_last
    U[0][col_idx] += 1
    V[v_0_last][col_idx] += 1
    W[w_0_last][col_idx] += 1
    col_idx += 1
    
    # Term 9: A01 * B1_last -> C0_last
    U[1][col_idx] += 1
    V[v_1_last][col_idx] += 1
    W[w_0_last][col_idx] += 1
    col_idx += 1
    
    # Term 10: A10 * B0_last -> C1_last
    U[2][col_idx] += 1
    V[v_0_last][col_idx] += 1
    W[w_1_last][col_idx] += 1
    col_idx += 1
    
    # Term 11: A11 * B1_last -> C1_last
    U[3][col_idx] += 1
    V[v_1_last][col_idx] += 1
    W[w_1_last][col_idx] += 1
    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
