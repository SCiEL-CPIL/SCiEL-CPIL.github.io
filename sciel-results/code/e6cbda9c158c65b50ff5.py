"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: two 4x4x4 blocks and one 4x4x1 block.
    # Trivial rank for 4x4x4 is 64.
    # Trivial rank for 4x4x1 is 4.
    # Total rank = 64 + 64 + 4 = 132.
    
    R_4 = 64
    R_1 = 4
    R = 2 * R_4 + R_1
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Helper to fill trivial 4x4x4 decomposition
    def fill_4x4x4(U, V, W, offset_U, offset_V_rows, offset_W, offset_R):
        # U is n*m x R, V is m*p x R, W is p*n x R
        # For 4x4x4, indices i,j,k in 0..3
        # U index i*4+j, V index j*4+k (local), W index k*4+i (local)
        # But V in the full problem has index j*p+k.
        # We need to map local V index j*4+k to global V index j*p+k.
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = offset_R + (i * 16 + j * 4 + k)
                    
                    # U
                    U[i * 4 + j][r] = 1
                    
                    # V
                    # Local V index is j*4+k? No, local V is 4x4.
                    # In the 4x4x4 problem, V is 4x4. Index j*4+k.
                    # In the full problem, V is 4x9. Index j*9+k.
                    # We are filling the block for k in 0..3 (or 4..7).
                    # offset_V_rows is not used for V index calculation directly, 
                    # we use the global j, k.
                    # But we need to know which k range.
                    # Let's pass k_offset.
                    
                    # Wait, the function signature needs to know the k offset.
                    pass
                    
    # Let's just write it out explicitly for clarity and correctness.
    
    # Block 1: k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                r = i * 16 + j * 4 + k
                U[i * 4 + j][r] = 1
                V[j * 9 + k][r] = 1
                W[k * 4 + i][r] = 1
                
    # Block 2: k in 4..7
    for i in range(4):
        for j in range(4):
            for k in range(4, 8):
                r = 64 + (i * 16 + j * 4 + (k - 4))
                U[i * 4 + j][r] = 1
                V[j * 9 + k][r] = 1
                W[k * 4 + i][r] = 1
                
    # Block 3: k in 8
    for i in range(4):
        for j in range(4):
            r = 128 + (i * 4 + j)
            U[i * 4 + j][r] = 1
            V[j * 9 + 8][r] = 1
            W[8 * 4 + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
