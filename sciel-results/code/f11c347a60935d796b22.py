"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (12, rank)
    
    # Rank calculation:
    # A_L (2x2) * B_L (2x6) -> 3 Strassen blocks (2x2) -> 3 * 7 = 21
    # A_R (2x1) * B_R (1x6) -> Outer product -> 12
    # Total Rank = 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # Part 1: A_L * B_L
    # A_L uses A[0,0], A[0,1], A[1,0], A[1,1]
    # Indices in U:
    # A[0,0] -> 0*3+0 = 0
    # A[0,1] -> 0*3+1 = 1
    # A[1,0] -> 1*3+0 = 3
    # A[1,1] -> 1*3+1 = 4
    
    # B_L uses B[0,0..5], B[1,0..5]
    # B_L is split into 3 blocks of 2 columns:
    # Block 0: cols 0, 1
    # Block 1: cols 2, 3
    # Block 2: cols 4, 5
    
    # Strassen coefficients for A_L (2x2) and B_block (2x2)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Output C (2x2) for each block:
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5 + M7
    # C21 = M2 + M4 - M6
    # C22 = M1 - M3 + M2 + M5
    
    # We need to map these to global indices.
    # For block b (0, 1, 2):
    # B columns are 2*b and 2*b+1.
    # B11 is B[0, 2*b] -> V index 0*6 + 2*b
    # B12 is B[0, 2*b+1] -> V index 0*6 + 2*b+1
    # B21 is B[1, 2*b] -> V index 1*6 + 2*b
    # B22 is B[1, 2*b+1] -> V index 1*6 + 2*b+1
    
    # Output C columns are 2*b and 2*b+1.
    # C11 is C[0, 2*b] -> W index (2*b)*2 + 0
    # C12 is C[0, 2*b+1] -> W index (2*b+1)*2 + 0
    # C21 is C[1, 2*b] -> W index (2*b)*2 + 1
    # C22 is C[1, 2*b+1] -> W index (2*b+1)*2 + 1
    
    for b in range(3):
        v_b11 = 2*b
        v_b12 = 2*b + 1
        v_b21 = 6 + 2*b
        v_b22 = 6 + 2*b + 1
        
        w_c11 = (2*b)*2 + 0
        w_c12 = (2*b+1)*2 + 0
        w_c21 = (2*b)*2 + 1
        w_c22 = (2*b+1)*2 + 1
        
        # M1
        add_term([0, 4], [1, 1], [v_b11, v_b22], [1, 1], 
                 [w_c11, w_c12, w_c21, w_c22], [1, 1, 1, 1])
        
        # M2
        add_term([3, 4], [1, 1], [v_b11], [1], 
                 [w_c21, w_c22], [1, 1])
        
        # M3
        add_term([0], [1], [v_b12, v_b22], [1, -1], 
                 [w_c12, w_c22], [1, -1])
        
        # M4
        add_term([4], [1], [v_b21, v_b11], [1, -1], 
                 [w_c11, w_c21], [-1, 1])
        
        # M5
        add_term([0, 1], [1, 1], [v_b22], [1], 
                 [w_c12, w_c22], [1, 1])
        
        # M6
        add_term([3, 0], [1, -1], [v_b11, v_b12], [1, 1], 
                 [w_c11, w_c21], [1, -1])
        
        # M7
        add_term([1, 4], [1, -1], [v_b21, v_b22], [1, 1], 
                 [w_c11, w_c12], [-1, 1])

    # Part 2: A_R * B_R
    # A_R uses A[0,2], A[1,2]
    # A[0,2] -> U index 2
    # A[1,2] -> U index 5
    
    # B_R uses B[2,0..5]
    # B[2,k] -> V index 2*6 + k = 12 + k
    
    # Output C_R is C[0,0..5] and C[1,0..5]
    # C[0,k] -> W index k*2 + 0
    # C[1,k] -> W index k*2 + 1
    
    # Outer product:
    # C[0,k] = A[0,2] * B[2,k]
    # C[1,k] = A[1,2] * B[2,k]
    
    for k in range(6):
        # Term for C[0,k]
        add_term([2], [1], [12 + k], [1], [k*2 + 0], [1])
        # Term for C[1,k]
        add_term([5], [1], [12 + k], [1], [k*2 + 1], [1])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
