"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen's algorithm for 2x2x2 blocks.
    # p=10 is split into four 2x2 blocks and one 2x2 block? 
    # 10 is not divisible by 2? Yes it is. 10 = 4 * 2 + 2.
    # So 5 blocks of size 2.
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (U)
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_coeffs = [
        [1, 0, 0, 1],  # M1: (A00+A11)
        [0, 0, 1, 1],  # M2: (A10+A11)
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: (A00+A01)
        [-1, 0, 1, 0], # M6: (A10-A00)
        [0, 1, 0, -1]  # M7: (A01-A11)
    ]
    
    # Strassen coefficients for B block (V)
    # B block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V_coeffs = [
        [1, 0, 0, 1],  # M1: (B00+B11)
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: (B01-B11)
        [0, 0, 1, -1], # M4: (B10-B00)
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: (B00+B01)
        [0, 0, 1, 1]   # M7: (B10+B11)
    ]
    
    # Strassen coefficients for C block (W)
    # C block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    W_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],  # P00
        [0, 0, 1, 0, 1, 0, 0],   # P01
        [0, 1, 0, 1, 0, 0, 0],   # P10
        [1, -1, 1, 0, 0, 1, 0]   # P11
    ]
    
    for t in range(num_blocks):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Map local B indices to global V indices
        # Local 0 (0,0) -> j=0, k=k0 -> idx k0
        # Local 1 (0,1) -> j=0, k=k1 -> idx k1
        # Local 2 (1,0) -> j=1, k=k0 -> idx p + k0
        # Local 3 (1,1) -> j=1, k=k1 -> idx p + k1
        v_map = [k0, k1, p + k0, p + k1]
        
        # Map local C indices to global W indices
        # Local 0 (0,0) -> i=0, k=k0 -> idx k0*n + 0
        # Local 1 (0,1) -> i=0, k=k1 -> idx k1*n + 0
        # Local 2 (1,0) -> i=1, k=k0 -> idx k0*n + 1
        # Local 3 (1,1) -> i=1, k=k1 -> idx k1*n + 1
        w_map = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
        
        for s in range(rank_per_block):
            col = t * rank_per_block + s
            
            # Fill U
            for i in range(n * m):
                U[i][col] = U_coeffs[s][i]
                
            # Fill V
            for local_idx, val in enumerate(V_coeffs[s]):
                global_idx = v_map[local_idx]
                V[global_idx][col] = val
                
            # Fill W
            for local_idx, val in enumerate(W_coeffs[local_idx]):
                # Wait, W_coeffs is indexed by output position (0..3)
                # We need to fill W for this column 'col'
                # W_coeffs[out_pos][s] is the coeff for M_s in P_out_pos
                pass
            
            # Correct W filling:
            # For each output position in the block (0..3), get coeff from W_coeffs[out_pos][s]
            for out_pos in range(4):
                coeff = W_coeffs[out_pos][s]
                global_idx = w_map[out_pos]
                W[global_idx][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
