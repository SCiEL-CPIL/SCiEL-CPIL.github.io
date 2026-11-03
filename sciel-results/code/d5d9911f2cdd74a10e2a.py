import time
import sys

# EVOLVE-BLOCK-START

def get_strassen_2x2_UVW():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    A: 00, 01, 10, 11 -> 0, 1, 2, 3
    B: 00, 01, 10, 11 -> 0, 1, 2, 3
    C: 00, 01, 10, 11 -> 0, 1, 2, 3
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def get_strassen_4x4_UVW():
    """
    Returns U, V, W for 4x4x4 Strassen (Rank 49).
    U: 16x49, V: 16x49, W: 16x49
    Indices:
    A: row-major 4x4 -> 0..15
    B: row-major 4x4 -> 0..15
    C: row-major 4x4 -> 0..15
    """
    n = 4
    rank = 49
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    base_U, base_V, base_W = get_strassen_2x2_UVW()
    
    # Strassen steps for 4x4 using 2x2 blocks
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # Define block coefficients for each of the 7 Strassen terms
    # Format: list of 7 tuples. Each tuple contains:
    # a_blocks: list of (block_row, block_col, coeff)
    # b_blocks: list of (block_row, block_col, coeff)
    # c_blocks: list of (block_row, block_col, coeff)
    
    strassen_steps = [
        # M1 = (A00+A11)(B00+B11)
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]
        },
        # M2 = (A10+A11)B00
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1)]
        },
        # M3 = A00(B01-B11)
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'c': [(0,1,1)]
        },
        # M4 = A11(B10-B00)
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'c': [(0,0,1), (1,0,1)]
        },
        # M5 = (A00+A01)B11
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,1,1), (1,1,-1)]
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,-1)]
        }
    ]
    
    # Helper to map block (br, bc) and local (lr, lc) to global index
    def idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
    
    for step_idx, step in enumerate(strassen_steps):
        # Each step corresponds to 7 sub-terms from the base Strassen
        for base_term in range(7):
            col = step_idx * 7 + base_term
            
            # Process U (A part)
            # Local A element (lr, lc) in base_U corresponds to a specific 2x2 position
            # We need to sum contributions from all A blocks involved in this Strassen step
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr*2 + lc
                    coeff_u_base = base_U[local_idx][base_term]
                    if coeff_u_base == 0: continue
                    
                    for br, bc, coeff_a in step['a']:
                        global_idx = idx(br, bc, lr, lc)
                        U[global_idx][col] += coeff_u_base * coeff_a
                        
            # Process V (B part)
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr*2 + lc
                    coeff_v_base = base_V[local_idx][base_term]
                    if coeff_v_base == 0: continue
                    
                    for br, bc, coeff_b in step['b']:
                        global_idx = idx(br, bc, lr, lc)
                        V[global_idx][col] += coeff_v_base * coeff_b
                        
            # Process W (C part)
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr*2 + lc
                    coeff_w_base = base_W[local_idx][base_term]
                    if coeff_w_base == 0: continue
                    
                    for br, bc, coeff_c in step['c']:
                        global_idx = idx(br, bc, lr, lc)
                        W[global_idx][col] += coeff_w_base * coeff_c
                        
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    # Part 1: 4x4x4 (first 4 cols of B and C)
    # Part 2: 4x4x3 (last 3 cols of B and C)
    
    U_44, V_44, W_44 = get_strassen_4x4_UVW()
    rank_44 = 49
    
    # Schoolbook for 4x4x3
    rank_43 = n * m * 3 # 48
    
    total_rank = rank_44 + rank_43
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Fill Part 1 (4x4x4) ---
    # U: 16x49
    # V: 16x49 (maps to B cols 0-3)
    # W: 16x49 (maps to C cols 0-3)
    
    for i in range(16):
        for k in range(rank_44):
            U[i][k] = U_44[i][k]
    
    # V mapping: local B index j*4+k -> global V index j*7+k
    for j in range(4):
        for k in range(4):
            local_v_idx = j*4 + k
            global_v_idx = j*7 + k
            for r in range(rank_44):
                V[global_v_idx][r] = V_44[local_v_idx][r]
                
    # W mapping: local C index k*4+i -> global W index k*4+i (since p*n indexing is k*n+i, and for first 4 cols k is 0..3)
    # Wait, W index is k*n + i.
    # Local C is 4x4. Row k (0..3), Col i (0..3).
    # Global C has p=7 rows (k=0..6), n=4 cols (i=0..3).
    # So for k in 0..3, global index is k*4 + i. This matches local index.
    for k in range(4):
        for i in range(4):
            local_w_idx = k*4 + i
            global_w_idx = k*4 + i
            for r in range(rank_44):
                W[global_w_idx][r] = W_44[local_w_idx][r]

    # --- Fill Part 2 (4x4x3) ---
    # Schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
    # Rank 48.
    # U: A[i,j]
    # V: B[j,k] where k is 4,5,6
    # W: C[i,k] where k is 4,5,6
    
    t = rank_44
    for i in range(n):
        for j in range(m):
            for k in range(3): # k corresponds to global k = 4, 5, 6
                global_k = k + 4
                
                u_idx = i * m + j
                v_idx = j * p + global_k
                w_idx = global_k * n + i
                
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
