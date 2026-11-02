"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x4 * 4x6
    # Rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 7 blocks
    # c: A11, A12, A21, A22
    # d: B11, B12, B21, B22
    C_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    D_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Targets for each M_k: (i_target, block_col_target, coeff)
    # block_col 0: k_global 0-2, block_col 1: k_global 3-5
    Targets = [
        [(0, 0, 1), (1, 1, 1)], # M1: C11(+), C22(+)
        [(1, 0, 1), (1, 1, -1)],# M2: C21(+), C22(-)
        [(0, 1, 1), (1, 1, 1)], # M3: C12(+), C22(+)
        [(0, 0, 1), (1, 0, 1)], # M4: C11(+), C21(+)
        [(0, 0, -1), (0, 1, 1)],# M5: C11(-), C12(+)
        [(1, 1, 1)],            # M6: C22(+)
        [(0, 0, 1)]             # M7: C11(+)
    ]
    
    for t in range(rank):
        k = t // 6
        j_sub = (t % 6) // 3
        k_out = t % 3
        
        c = C_coeffs[k]
        d = D_coeffs[k]
        
        # Fill U
        # U[i*m + j_global, t]
        # j_sub corresponds to column in 1x2 virtual A
        # A11 (i=0, j=0,1), A12 (i=0, j=2,3), A21 (i=1, j=0,1), A22 (i=1, j=2,3)
        # c[0] is A11, c[1] is A12, c[2] is A21, c[3] is A22
        
        # i=0
        U[0 * m + j_sub][t] = c[0]
        U[0 * m + j_sub + 2][t] = c[1]
        # i=1
        U[1 * m + j_sub][t] = c[2]
        U[1 * m + j_sub + 2][t] = c[3]
        
        # Fill V
        # V[j_global*p + k_global, t]
        # j_sub is row in 2x3 virtual B
        # k_out is column in 2x3 virtual B
        # B11 (j=0,1; k=0,1,2), B12 (j=0,1; k=3,4,5)
        # B21 (j=2,3; k=0,1,2), B22 (j=2,3; k=3,4,5)
        # d[0] is B11, d[1] is B12, d[2] is B21, d[3] is B22
        
        # j_global = j_sub
        V[j_sub * p + k_out][t] = d[0]
        V[j_sub * p + k_out + 3][t] = d[1]
        # j_global = j_sub + 2
        V[(j_sub + 2) * p + k_out][t] = d[2]
        V[(j_sub + 2) * p + k_out + 3][t] = d[3]
        
        # Fill W
        # W[k_global*n + i, t]
        targets = Targets[k]
        for i_t, bc_t, coeff in targets:
            k_global = k_out + 3 * bc_t
            W[k_global * n + i_t][t] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
