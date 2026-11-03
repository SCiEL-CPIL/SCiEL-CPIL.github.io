import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # Total rank calculation
    rank_part2 = n * p  # 4 * 6 = 24
    rank_part1a = 49    # 7 * 7 for 4x4x4 Strassen
    rank_part1b = n * 4 * 2  # 4 * 4 * 2 = 32 for 4x4x2 schoolbook
    
    total_rank = rank_part2 + rank_part1a + rank_part1b
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 2: A2 (4x1) * B2 (1x6)
    # A2 is A[:, 4], B2 is B[4, :]
    # C += A[:, 4] * B[4, :]
    for i in range(n):
        for k in range(p):
            # U index: i, j=4
            u_idx = i * m + 4
            # V index: j=4, k
            v_idx = 4 * p + k
            # W index: k, i
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 1a: A1 (4x4) * B11 (4x4) using Strassen
    # A1 is A[:, 0:4], B11 is B[0:4, 0:4]
    # We need to implement recursive Strassen for 4x4x4
    
    # Helper to add a 2x2x2 Strassen decomposition
    # offsets: 
    # u_offset: list of (i, j) global coords for the 4 local (i', j')
    # v_offset: list of (j, k) global coords for the 4 local (j', k')
    # w_offset: list of (k, i) global coords for the 4 local (k', i')
    # The local indices are 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    
    # Strassen 2x2x2 vectors
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, -1, 0],
        [0, 1, -1, 0]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Strassen 4x4x4 combines 7 terms
    # Each term is a product of sums of 2x2 blocks.
    # We define the 7 terms of 4x4x4 Strassen in terms of 2x2 blocks.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # Term 1: (A00 + A11)(B00 + B11)
    # Term 2: (A10 + A11)B00
    # Term 3: A00(B01 - B11)
    # Term 4: A11(B10 - B00)
    # Term 5: (A00 + A01)B11
    # Term 6: (A10 - A00)(B00 + B01)
    # Term 7: (A01 - A11)(B10 + B11)
    
    # For each term, we define the coefficients for the 4 A-blocks and 4 B-blocks.
    # Then we apply the 2x2x2 Strassen decomposition to the resulting 2x2x2 "scalar" multiplication.
    
    # Coefficients for A blocks (A00, A01, A10, A11) for the 7 terms
    a_coeffs = [
        [1, 0, 0, 1], # T1
        [0, 0, 1, 1], # T2
        [1, 0, 0, 0], # T3
        [0, 0, 0, 1], # T4
        [1, 1, 0, 0], # T5
        [1, 0, -1, 0],# T6
        [0, 1, 0, -1] # T7
    ]
    
    # Coefficients for B blocks (B00, B01, B10, B11) for the 7 terms
    b_coeffs = [
        [1, 0, 0, 1], # T1
        [1, 0, 0, 0], # T2
        [0, 1, 0, -1],# T3
        [0, -1, 1, 0],# T4
        [0, 0, 0, 1], # T5
        [1, 1, 0, 0], # T6
        [0, 0, 1, 1]  # T7
    ]
    
    # Coefficients for C blocks (C00, C01, C10, C11) for the 7 terms
    # From:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    c_coeffs = [
        [1, 0, 0, 1], # T1
        [0, 0, 1, -1],# T2
        [0, 1, 0, 1], # T3
        [1, 0, 1, 0], # T4
        [-1, 1, 0, 0],# T5
        [0, 0, 0, 1], # T6
        [1, 0, 0, 0]  # T7
    ]
    
    # Now, for each of the 7 major terms (t in 0..6):
    # We have a "virtual" 2x2 matrix X = sum a_coeffs[t][b] * A_b
    # and Y = sum b_coeffs[t][b] * B_b
    # We compute Z = X Y using 2x2x2 Strassen.
    # The result Z contributes to C blocks with coefficients c_coeffs[t].
    
    # The 2x2x2 Strassen decomposition has 7 sub-terms (r in 0..6).
    # For each sub-term r:
    # u_r, v_r, w_r are vectors of length 4 (local 2x2 indices).
    # We need to map these to global indices.
    
    # Local 2x2 index l (0..3) corresponds to (i', j') where i'=l//2, j'=l%2.
    # A_b block has global rows i = 2*i_b + i', cols j = 2*j_b + j'.
    # So for a given local index l in u_r, value val:
    # We add val * a_coeffs[t][b] to U[2*i_b + i', 2*j_b + j'] for each block b.
    
    for t in range(7):
        for r in range(7):
            # u_r, v_r, w_r are the 2x2x2 Strassen vectors
            u_vec = s_u[r]
            v_vec = s_v[r]
            w_vec = s_w[r]
            
            # Process U
            for l in range(4):
                val = u_vec[l]
                if val == 0: continue
                i_prime = l // 2
                j_prime = l % 2
                for b in range(4):
                    coeff = a_coeffs[t][b]
                    if coeff == 0: continue
                    i_b = b // 2
                    j_b = b % 2
                    i_global = 2 * i_b + i_prime
                    j_global = 2 * j_b + j_prime
                    u_idx = i_global * m + j_global
                    U[u_idx][col] += val * coeff
            
            # Process V
            for l in range(4):
                val = v_vec[l]
                if val == 0: continue
                j_prime = l // 2
                k_prime = l % 2
                for b in range(4):
                    coeff = b_coeffs[t][b]
                    if coeff == 0: continue
                    j_b = b // 2
                    k_b = b % 2
                    j_global = 2 * j_b + j_prime
                    k_global = 2 * k_b + k_prime
                    v_idx = j_global * p + k_global
                    V[v_idx][col] += val * coeff
            
            # Process W
            for l in range(4):
                val = w_vec[l]
                if val == 0: continue
                k_prime = l // 2
                i_prime = l % 2
                for b in range(4):
                    coeff = c_coeffs[t][b]
                    if coeff == 0: continue
                    k_b = b // 2
                    i_b = b % 2
                    k_global = 2 * k_b + k_prime
                    i_global = 2 * i_b + i_prime
                    w_idx = k_global * n + i_global
                    W[w_idx][col] += val * coeff
            
            col += 1
            
    # Part 1b: A1 (4x4) * B12 (4x2) using Schoolbook
    # A1 is A[:, 0:4], B12 is B[0:4, 4:6]
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
