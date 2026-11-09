"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    # Rank 15 decomposition:
    # 1 term for A_R B_B (j=2)
    # 7 terms for A_L B_T, k in {0,1} (Strassen)
    # 7 terms for A_L B_T, k in {2,3} (Strassen)
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # 1. A_R B_B term (j=2)
    # U index: i*m + j = i*3 + 2
    # V index: j*p + k = 2*4 + k = 8 + k
    # W index: k*n + i = k*2 + i
    col = 0
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1

    # 2. Strassen for k in {0, 1}
    # A_L is 2x2 (i=0,1; j=0,1)
    # B_T is 2x2 (j=0,1; k=0,1)
    # Strassen factors for M_1..M_7
    
    # M1: (A00+A11)(B00+B11)
    # U: A00, A11
    # V: B00, B11
    # W: C00, C01, C10, C11
    # Standard Strassen mapping:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to fill U, V, W for these 7 terms.
    # Note: U, V, W are linear combinations.
    
    # Term 1: M1
    # U: A00 (i=0,j=0), A11 (i=1,j=1)
    # V: B00 (j=0,k=0), B11 (j=1,k=1)
    # W: C00, C01, C10, C11
    # Wait, the W index is k*n+i.
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    
    # For M1:
    # U[0*3+0] = 1, U[1*3+1] = 1
    # V[0*4+0] = 1, V[1*4+1] = 1
    # W[0] = 1, W[2] = 1, W[1] = 1, W[3] = 1
    # But W is a single vector per term.
    # The term is U * V * W.
    # So W must be the same for all C entries?
    # No, W is a vector of length rank.
    # For a fixed term c, W[k*n+i, c] is a scalar.
    # So for M1, W[0, c]=1, W[1, c]=1, W[2, c]=1, W[3, c]=1.
    
    # Let's define the 7 Strassen terms for k=0,1 block.
    # Indices for A: i*3+j
    # Indices for B: j*4+k
    # Indices for C: k*2+i
    
    def add_strassen_block(k_offset, col_start):
        # k_offset is 0 or 2
        # k_local is 0, 1
        # k_global = k_offset + k_local
        
        # M1: (A00+A11)(B00+B11)
        c = col_start
        U[0*3+0][c] = 1
        U[1*3+1][c] = 1
        V[0*4+k_offset+0][c] = 1
        V[1*4+k_offset+1][c] = 1
        W[(k_offset+0)*2+0][c] = 1
        W[(k_offset+1)*2+0][c] = 1
        W[(k_offset+0)*2+1][c] = 1
        W[(k_offset+1)*2+1][c] = 1
        
        # M2: (A10+A11)B00
        c = col_start + 1
        U[1*3+0][c] = 1
        U[1*3+1][c] = 1
        V[0*4+k_offset+0][c] = 1
        W[(k_offset+0)*2+1][c] = 1
        W[(k_offset+1)*2+1][c] = -1 # C11 has -M2
        
        # M3: A00(B01-B11)
        c = col_start + 2
        U[0*3+0][c] = 1
        V[0*4+k_offset+1][c] = 1
        V[1*4+k_offset+1][c] = -1
        W[(k_offset+1)*2+0][c] = 1
        W[(k_offset+1)*2+1][c] = 1
        
        # M4: A11(B10-B00)
        c = col_start + 3
        U[1*3+1][c] = 1
        V[1*4+k_offset+0][c] = 1
        V[0*4+k_offset+0][c] = -1
        W[(k_offset+0)*2+0][c] = 1
        W[(k_offset+1)*2+1][c] = 1
        
        # M5: (A00+A01)B11
        c = col_start + 4
        U[0*3+0][c] = 1
        U[0*3+1][c] = 1
        V[1*4+k_offset+1][c] = 1
        W[(k_offset+0)*2+0][c] = -1
        W[(k_offset+1)*2+0][c] = 1
        
        # M6: (A10-A00)(B00+B01)
        c = col_start + 5
        U[1*3+0][c] = 1
        U[0*3+0][c] = -1
        V[0*4+k_offset+0][c] = 1
        V[0*4+k_offset+1][c] = 1
        W[(k_offset+1)*2+1][c] = 1
        
        # M7: (A01-A11)(B10+B11)
        c = col_start + 6
        U[0*3+1][c] = 1
        U[1*3+1][c] = -1
        V[1*4+k_offset+0][c] = 1
        V[1*4+k_offset+1][c] = 1
        W[(k_offset+0)*2+0][c] = -1
        W[(k_offset+1)*2+1][c] = -1

    add_strassen_block(0, 1)
    add_strassen_block(2, 8)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
