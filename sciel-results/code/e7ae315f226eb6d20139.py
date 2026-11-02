"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # X coeffs for m1..m7
    X_coeffs = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [-1, 0, 1, 0], # m6
        [0, 1, 0, -1]  # m7
    ]
    # Y coeffs for m1..m7
    Y_coeffs = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, -1, 1, 0], # m3
        [-1, 1, 0, 0], # m4
        [0, 0, 0, 1], # m5
        [1, 0, 1, 0], # m6
        [0, 1, 0, 1]  # m7
    ]
    # Output coeffs for z00, z01, z10, z11 from m1..m7
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    Z_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # z00
        [0, 0, 1, 0, 1, 0, 0],  # z01
        [0, 1, 0, 1, 0, 0, 0],  # z10
        [1, -1, 1, 0, 0, 1, 0]  # z11
    ]
    
    # Define the 4 blocks
    # Block 0: A_L * B_LL -> C_LL (part of C_L)
    # Block 1: A_R * B_RL -> C_RL (part of C_L)
    # Block 2: A_L * B_LR -> C_LR (part of C_R)
    # Block 3: A_R * B_RR -> C_RR (part of C_R)
    
    # A indices
    # A_L: 0,1,2,3
    # A_R: 4,5,6,7
    A_block_indices = [
        [0, 1, 2, 3], # A_L
        [4, 5, 6, 7], # A_R
        [0, 1, 2, 3], # A_L
        [4, 5, 6, 7]  # A_R
    ]
    
    # B indices
    # B_LL: 0,1,4,5
    # B_RL: 8,9,12,13
    # B_LR: 2,3,6,7
    # B_RR: 10,11,14,15
    B_block_indices = [
        [0, 1, 4, 5],   # B_LL
        [8, 9, 12, 13], # B_RL
        [2, 3, 6, 7],   # B_LR
        [10, 11, 14, 15] # B_RR
    ]
    
    # C indices (W indices)
    # C_L (cols 0,1): 0,2,1,3
    # C_R (cols 2,3): 4,6,5,7
    # Note: W index is k*n + i = k*2 + i
    # C_L corresponds to k=0,1
    # C_00 (i=0,k=0) -> 0
    # C_01 (i=0,k=1) -> 2
    # C_10 (i=1,k=0) -> 1
    # C_11 (i=1,k=1) -> 3
    # C_R corresponds to k=2,3
    # C_02 (i=0,k=2) -> 4
    # C_03 (i=0,k=3) -> 6
    # C_12 (i=1,k=2) -> 5
    # C_13 (i=1,k=3) -> 7
    
    C_block_indices = [
        [0, 2, 1, 3], # C_L (z00, z01, z10, z11)
        [0, 2, 1, 3], # C_L
        [4, 6, 5, 7], # C_R
        [4, 6, 5, 7]  # C_R
    ]
    
    col = 0
    for block in range(4):
        a_idx = A_block_indices[block]
        b_idx = B_block_indices[block]
        c_idx = C_block_indices[block]
        
        for s in range(7):
            # Fill U
            for local_i in range(4):
                U[a_idx[local_i]][col] += X_coeffs[s][local_i]
            
            # Fill V
            for local_j in range(4):
                V[b_idx[local_j]][col] += Y_coeffs[s][local_j]
            
            # Fill W
            for local_k in range(4):
                W[c_idx[local_k]][col] += Z_coeffs[local_k][s]
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
