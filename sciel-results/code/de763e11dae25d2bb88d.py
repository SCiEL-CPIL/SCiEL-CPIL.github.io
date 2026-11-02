"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose p=16 into 8 blocks of size 2
    num_blocks = p // 2
    block_size = 2
    # Rank of 2x2x2 multiplication is 7
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2), B (2x2), C (2x2)
    # M1 = (A11+A22)(B11+B22) -> C11+, C22+
    # M2 = (A21+A22)B22       -> C21+, C22-
    # M3 = A11(B12-B22)       -> C12+, C22+
    # M4 = A22(B21-B11)       -> C11+, C21+
    # M5 = (A11+A12)B11       -> C11-, C12+
    # M6 = (A21+A11)(B22-B12) -> C22+
    # M7 = (A12+A22)(B21-B11) -> C11+
    
    # Define the 7 terms as (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {(i,j): coeff}
    # B_coeffs: dict {(j,k): coeff} where k is local 0 or 1
    # C_coeffs: dict {(i,k): coeff} where k is local 0 or 1
    
    terms = [
        # M1
        (
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}
        ),
        # M2
        (
            {(1,0): 1, (1,1): 1}, 
            {(1,1): 1}, 
            {(1,0): 1, (1,1): -1}
        ),
        # M3
        (
            {(0,0): 1}, 
            {(0,1): 1, (1,1): -1}, 
            {(0,1): 1, (1,1): 1}
        ),
        # M4
        (
            {(1,1): 1}, 
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (1,0): 1}
        ),
        # M5
        (
            {(0,0): 1, (0,1): 1}, 
            {(0,0): 1}, 
            {(0,0): -1, (0,1): 1}
        ),
        # M6
        (
            {(1,0): 1, (0,0): 1}, 
            {(1,1): 1, (0,1): -1}, 
            {(1,1): 1}
        ),
        # M7
        (
            {(0,1): 1, (1,1): 1}, 
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1}
        )
    ]
    
    for b in range(num_blocks):
        base_col = b * rank_per_block
        for r_idx in range(rank_per_block):
            col = base_col + r_idx
            a_coeffs, b_coeffs, c_coeffs = terms[r_idx]
            
            # Fill U
            for (i, j), val in a_coeffs.items():
                U[i * m + j][col] += val
                
            # Fill V
            # Local k 0 -> global k = 2*b
            # Local k 1 -> global k = 2*b + 1
            for (j, k_local), val in b_coeffs.items():
                k_global = 2 * b + k_local
                V[j * p + k_global][col] += val
                
            # Fill W
            # Local k 0 -> global k = 2*b
            # Local k 1 -> global k = 2*b + 1
            # W index: k * n + i
            for (i, k_local), val in c_coeffs.items():
                k_global = 2 * b + k_local
                W[k_global * n + i][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
