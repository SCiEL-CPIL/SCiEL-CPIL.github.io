"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Part 1: 2x1x6 -> 12
    # Part 2: 2x4x6 -> 2 * Rank(2x2x6)
    # Rank(2x2x6) using 3 Strassen blocks -> 3 * 7 = 21
    # Total Rank = 12 + 2 * 21 = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0
    # A[0,0] * B[0,k]
    # A[1,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            # U index: i * m + 0
            U[i * m][col] = 1
            # V index: 0 * p + k
            V[k][col] = 1
            # W index: k * n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in 1..4
    # Split into two blocks of 2x2x6
    # Block 1: j in 1, 2
    # Block 2: j in 3, 4
    
    for block_idx in range(2):
        # Local j' 0, 1 map to Global j
        j_base = 1 + block_idx * 2
        
        # For each Strassen instance (3 pairs of columns)
        for s in range(3):
            # Local k 0, 1 map to Global k
            k_base = s * 2
            
            # Define the 7 Strassen terms
            # Term (u_coeffs, v_coeffs, w_coeffs)
            # u: (A00, A01, A10, A11)
            # v: (B00, B01, B10, B11)
            # w: (C00, C01, C10, C11)
            
            # T1: (a+d)(e+h)
            # a=A00, d=A11, e=B00, h=B11
            # C00 += 1, C11 += 1
            terms = [
                # (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
                # Indices are local (0..3 for A, 0..3 for B, 0..3 for C)
                
                # T1
                ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
                # T2: (c+d)e
                # c=A10, d=A11, e=B00
                # C10 += 1, C11 -= 1
                ([2, 3], [1, 1], [0], [1], [2], [1], [3], [-1]), 
                # Note: The tuple structure needs to be consistent.
            ]
            
            # Let's define a helper to apply a term
            # u: list of (local_i, local_j, val)
            # v: list of (local_j, local_k, val)
            # w: list of (local_i, local_k, val)
            
            strassen_terms = [
                # (u_list, v_list, w_list)
                (
                    [(0,0,1), (1,1,1)], # a, d
                    [(0,0,1), (1,1,1)], # e, h
                    [(0,0,1), (1,1,1)]  # C00, C11
                ),
                (
                    [(1,0,1), (1,1,1)], # c, d
                    [(0,0,1)],          # e
                    [(1,0,1), (1,1,-1)] # C10, C11
                ),
                (
                    [(0,0,1)],          # a
                    [(0,1,1), (1,1,-1)],# f, -h
                    [(0,1,1), (1,1,1)]  # C01, C11
                ),
                (
                    [(1,1,1)],          # d
                    [(1,0,1), (0,0,-1)],# g, -e
                    [(0,0,1), (1,0,1)]  # C00, C10
                ),
                (
                    [(0,0,1), (0,1,1)], # a, b
                    [(1,1,1)],          # h
                    [(0,0,-1), (0,1,1)] # C00, C01
                ),
                (
                    [(1,0,1), (0,0,-1)],# c, -a
                    [(0,0,1), (0,1,1)], # e, f
                    [(1,1,1)]           # C11
                ),
                (
                    [(0,1,1), (1,1,-1)],# b, -d
                    [(1,0,1), (1,1,1)], # g, h
                    [(0,0,1)]           # C00
                ),
            ]
            
            for term in strassen_terms:
                u_list, v_list, w_list = term
                
                # Apply U
                for (li, lj, val) in u_list:
                    global_i = li
                    global_j = j_base + lj
                    U[global_i * m + global_j][col] += val
                    
                # Apply V
                for (lj, lk, val) in v_list:
                    global_j = j_base + lj
                    global_k = k_base + lk
                    V[global_j * p + global_k][col] += val
                    
                # Apply W
                for (li, lk, val) in w_list:
                    global_i = li
                    global_k = k_base + lk
                    W[global_k * n + global_i][col] += val
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
