"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrix multiplication (Rank 7)
    # Indices 0..3 correspond to [0,0], [0,1], [1,0], [1,1]
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, -1, 0, 0, 1, 1]   # C11
    ]
    
    # Verify 2x2 decomposition
    # C = A * B
    # C00 = A00*B00 + A01*B10
    # Using W2[0]: 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 + 1*M7
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M7 = (A01-A11)(B10+B11)
    # Sum = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A01B10. Correct.
    
    n, m, p = DIMENSIONS
    # We are building a 4x4x4 decomposition using Kronecker product of two 2x2x2 Strassen decompositions.
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)] # 16 x 49
    V = [[0] * rank for _ in range(m * p)] # 16 x 49
    W = [[0] * rank for _ in range(p * n)] # 16 x 49
    
    # Map 2x2 block indices to 4x4 scalar indices
    # Block (R, C) in 2x2 view of 4x4 matrix.
    # R, C in {0, 1}
    # Local indices r, c in {0, 1}
    # Global row = R*2 + r
    # Global col = C*2 + c
    # Scalar index = row*4 + col
    
    for k1 in range(7):
        for k2 in range(7):
            k = k1 * 7 + k2
            # Iterate over all scalar positions in 4x4 matrices
            for R in range(2):
                for C in range(2):
                    u_outer = U2[C * 2 + R][k1] # U2 index is [row, col] in 2x2 block? 
                    # Wait, U2 is defined for the 2x2 matrix indices.
                    # In the 2x2 Strassen, U2[i, k] where i is the index in the 2x2 matrix.
                    # My U2 definition:
                    # U2[0] is A00, U2[1] is A01, U2[2] is A10, U2[3] is A11.
                    # So index i in U2 corresponds to (i//2, i%2).
                    # Here R is block row, C is block col.
                    # So the index in U2 is R*2 + C.
                    u_outer_val = U2[R * 2 + C][k1]
                    v_outer_val = V2[R * 2 + C][k1]
                    w_outer_val = W2[R * 2 + C][k1]
                    
                    if u_outer_val == 0 and v_outer_val == 0 and w_outer_val == 0:
                        continue
                    
                    for r in range(2):
                        for c in range(2):
                            # Inner indices r, c correspond to the element within the block.
                            # In the 2x2 Strassen decomposition of the block, the index is r*2+c.
                            u_inner_val = U2[r * 2 + c][k2]
                            v_inner_val = V2[r * 2 + c][k2]
                            w_inner_val = W2[r * 2 + c][k2]
                            
                            # Global indices
                            row = R * 2 + r
                            col = C * 2 + c
                            idx = row * 4 + col
                            
                            U[idx][k] += u_outer_val * u_inner_val
                            V[idx][k] += v_outer_val * v_inner_val
                            W[idx][k] += w_outer_val * w_inner_val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
