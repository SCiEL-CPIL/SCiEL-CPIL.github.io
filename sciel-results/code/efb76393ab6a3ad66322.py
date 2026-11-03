"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using hierarchical Strassen
    # 8 blocks of 2x2x2 Strassen (rank 7 each)
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Order: A00, A01, A10, A11 and B00, B01, B10, B11
    # Output: C00, C01, C10, C11
    strassen_terms = [
        # L_A, L_B, R_C
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 0]),  # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 0, 1]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),  # M7
    ]
    
    col = 0
    # 4 column groups (x) and 2 row groups (part)
    for x in range(4):
        for part in range(2):
            # part 0: j in {0,1}, part 1: j in {2,3}
            # x: k in {2x, 2x+1}
            
            for term in strassen_terms:
                L_A, L_B, R_C = term
                
                # Map local A indices (0,1,2,3) to global i, j
                # Local 0: i=0, j_local=0
                # Local 1: i=0, j_local=1
                # Local 2: i=1, j_local=0
                # Local 3: i=1, j_local=1
                for l_idx in range(4):
                    coeff = L_A[l_idx]
                    if coeff != 0:
                        i = l_idx // 2
                        j_loc = l_idx % 2
                        j_glob = 2 * part + j_loc
                        u_idx = i * m + j_glob
                        U[u_idx][col] += coeff
                        
                # Map local B indices (0,1,2,3) to global j, k
                # Local 0: j_local=0, k_loc=0
                # Local 1: j_local=0, k_loc=1
                # Local 2: j_local=1, k_loc=0
                # Local 3: j_local=1, k_loc=1
                for l_idx in range(4):
                    coeff = L_B[l_idx]
                    if coeff != 0:
                        j_loc = l_idx // 2
                        k_loc = l_idx % 2
                        j_glob = 2 * part + j_loc
                        k_glob = 2 * x + k_loc
                        v_idx = j_glob * p + k_glob
                        V[v_idx][col] += coeff
                        
                # Map local C indices (0,1,2,3) to global k, i
                # Local 0: k_loc=0, i=0
                # Local 1: k_loc=0, i=1
                # Local 2: k_loc=1, i=0
                # Local 3: k_loc=1, i=1
                # Note: W index is k*n + i
                for l_idx in range(4):
                    coeff = R_C[l_idx]
                    if coeff != 0:
                        k_loc = l_idx // 2
                        i = l_idx % 2
                        k_glob = 2 * x + k_loc
                        w_idx = k_glob * n + i
                        W[w_idx][col] += coeff
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
