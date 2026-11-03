import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    
    # We will construct a decomposition with rank 49 (for 4x4x4 part) + 48 (for 4x4x3 part) = 97.
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x3 using schoolbook (rank 48)
    
    total_rank = 49 + 48
    
    # U: (n*m) x rank = 16 x 97
    # V: (m*p) x rank = 28 x 97
    # W: (p*n) x rank = 28 x 97
    # Note: The task description says W is (p*n, rank). 
    # Standard matrix mult C = A B, A is n x m, B is m x p, C is n x p.
    # Tensor T_{i,j,k} = sum U_{i,j,rank} V_{j,k,rank} W_{k,i,rank}? 
    # Let's check the index convention in the prompt: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies W index is (k, i) where k is col of C (0..p-1) and i is row of C (0..n-1).
    # So W shape is (p*n, rank).
    # U index is (i, j) where i is row of A (0..n-1) and j is col of A (0..m-1). Shape (n*m, rank).
    # V index is (j, k) where j is row of B (0..m-1) and k is col of B (0..p-1). Shape (m*p, rank).
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen ---
    # We decompose the 4x4x4 multiplication tensor.
    # A is 4x4, B is 4x4 (cols 0-3), C is 4x4 (cols 0-3).
    
    # Strassen 2x2x2 base case
    # U2, V2, W2 are 4x7 matrices
    # Indices for 2x2 matrices are (r, c) -> r*2+c
    def get_strassen_2x2():
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # A00=U[0], A11=U[3], B00=V[0], B11=V[3]
        # C00 += M1, C11 += M1
        # U: A00, A11
        U2[0][0] = 1; U2[3][0] = 1
        # V: B00, B11
        V2[0][0] = 1; V2[3][0] = 1
        # W: C00, C11
        W2[0][0] = 1; W2[3][0] = 1
        
        # M2 = (A10+A11)B00
        # A10=U[2], A11=U[3], B00=V[0]
        # C10 += M2
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[2][1] = 1
        
        # M3 = A00(B01-B11)
        # A00=U[0], B01=V[1], B11=V[3]
        # C01 += M3
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[1][2] = 1
        
        # M4 = A11(B10-B00)
        # A11=U[3], B10=V[2], B00=V[0]
        # C00 += M4, C10 += M4
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = 1; W2[2][3] = 1
        
        # M5 = (A00+A01)B11
        # A00=U[0], A01=U[1], B11=V[3]
        # C01 += M5, C11 -= M5
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[1][4] = 1; W2[3][4] = -1
        
        # M6 = (A10-A00)(B00+B01)
        # A10=U[2], A00=U[0], B00=V[0], B01=V[1]
        # C11 += M6
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01=U[1], A11=U[3], B10=V[2], B11=V[3]
        # C00 += M7, C11 -= M7
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1; W2[3][6] = -1
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2()
    
    # 4x4 Strassen
    # We split A, B, C into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 Strassen products at the block level are:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Each Si is a 2x2 matrix product. We expand each Si using U2, V2, W2.
    # For each block-level step s in 0..6:
    #   The A-part is a linear combination of A-blocks.
    #   The B-part is a linear combination of B-blocks.
    #   The C-part is a linear combination of C-blocks.
    
    # Define the block-level coefficients
    # a_coeffs[s] is a dict mapping block (br, bc) to coeff
    # b_coeffs[s] is a dict mapping block (br, bc) to coeff
    # c_coeffs[s] is a dict mapping block (br, bc) to coeff
    
    a_coeffs = [
        {(0,0): 1, (1,1): 1}, # S1
        {(1,0): 1, (1,1): 1}, # S2
        {(0,0): 1},           # S3
        {(1,1): 1},           # S4
        {(0,0): 1, (0,1): 1}, # S5
        {(1,0): 1, (0,0): -1},# S6
        {(0,1): 1, (1,1): -1} # S7
    ]
    b_coeffs = [
        {(0,0): 1, (1,1): 1}, # S1
        {(0,0): 1},           # S2
        {(0,1): 1, (1,1): -1},# S3
        {(1,0): 1, (0,0): -1},# S4
        {(1,1): 1},           # S5
        {(0,0): 1, (0,1): 1}, # S6
        {(1,0): 1, (1,1): 1}  # S7
    ]
    c_coeffs = [
        {(0,0): 1, (1,1): 1}, # S1
        {(1,0): 1},           # S2
        {(0,1): 1},           # S3
        {(0,0): 1, (1,0): 1}, # S4
        {(0,1): 1, (1,1): -1},# S5
        {(1,1): 1},           # S6
        {(0,0): 1, (1,1): -1} # S7
    ]
    
    # Global index helpers
    # A global idx: row * m + col. Row = br*2 + lr, Col = bc*2 + lc
    def a_glob(br, bc, lr, lc):
        return (br*2 + lr) * m + (bc*2 + lc)
        
    # B global idx: row * p + col. Row = br*2 + lr, Col = bc*2 + lc (for 4x4 part, col < 4)
    def b_glob(br, bc, lr, lc):
        return (br*2 + lr) * p + (bc*2 + lc)
        
    # W global idx: col * n + row. Col = bc*2 + lc, Row = br*2 + lr
    def w_glob(br, bc, lr, lc):
        return (bc*2 + lc) * n + (br*2 + lr)
        
    # Fill U, V, W for the 49 terms
    for s in range(7):
        for base_term in range(7):
            rank_idx = s * 7 + base_term
            
            # U part
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    coeff_u_base = U2[local_idx][base_term]
                    if coeff_u_base == 0:
                        continue
                    for (br, bc), coeff_a_block in a_coeffs[s].items():
                        if coeff_a_block == 0:
                            continue
                        g_idx = a_glob(br, bc, lr, lc)
                        U[g_idx][rank_idx] += coeff_u_base * coeff_a_block
                        
            # V part
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    coeff_v_base = V2[local_idx][base_term]
                    if coeff_v_base == 0:
                        continue
                    for (br, bc), coeff_b_block in b_coeffs[s].items():
                        if coeff_b_block == 0:
                            continue
                        g_idx = b_glob(br, bc, lr, lc)
                        V[g_idx][rank_idx] += coeff_v_base * coeff_b_block
                        
            # W part
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    coeff_w_base = W2[local_idx][base_term]
                    if coeff_w_base == 0:
                        continue
                    for (br, bc), coeff_c_block in c_coeffs[s].items():
                        if coeff_c_block == 0:
                            continue
                        g_idx = w_glob(br, bc, lr, lc)
                        W[g_idx][rank_idx] += coeff_w_base * coeff_c_block

    # --- Part 2: 4x4x3 Schoolbook ---
    # Columns 4, 5, 6 of B.
    # i in 0..3 (rows of A/C)
    # j in 0..3 (cols of A / rows of B)
    # k in 4..6 (cols of B/C)
    
    schoolbook_start = 49
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                rank_idx = schoolbook_start + t
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][rank_idx] = 1
                V[v_idx][rank_idx] = 1
                W[w_idx][rank_idx] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
