"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Decompose into four 2x3x2 sub-problems (column pairs of B)
    # For each 2x3x2 block, use a rank-6 Winograd/Strassen-like decomposition
    # Total rank: 4 * 6 = 24
    
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 16 x 24
    
    # For a 2x3x2 matrix multiplication:
    # A is 2x3: [[a00, a01, a02], [a10, a11, a12]]
    # B is 3x2: [[b00, b01], [b10, b11], [b20, b21]]
    # C is 2x2: C[0,0] = a00*b00 + a01*b10 + a02*b20
    #            C[0,1] = a00*b01 + a01*b11 + a02*b21
    #            C[1,0] = a10*b00 + a11*b10 + a12*b20
    #            C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # Winograd-style rank-6 decomposition for 2x3x2:
    # M1 = (a00 + a10) * (b00 + b01)
    # M2 = (a01 + a11) * (b10 + b11)
    # M3 = (a02 + a12) * (b20 + b21)
    # M4 = (a00 - a10) * b00
    # M5 = (a01 - a11) * b10
    # M6 = (a02 - a12) * b20
    
    # Then:
    # C[0,0] = M1 + M2 + M3 - M4 - M5 - M6  ... wait, let me recompute
    
    # Actually, let me use a proper rank-6 decomposition.
    # The standard Winograd for 2x2x2 is rank 7. For 2x3x2, we need to think differently.
    
    # Let me use a different approach: treat 2x3x2 as two 2x3x1 problems combined.
    # Actually, let me just verify if rank 6 is achievable.
    
    # For 2x3x2, the schoolbook rank is 12. Can we do better?
    # M(2,3,2) >= max(M(2,2,2), M(2,3,1)+M(2,3,1)) = max(7, 12) ... no that's not right.
    
    # Let me just use the schoolbook for each 2x3x2 block: rank 12 per block, total 48.
    # But that's the same as the parent. Let me try a different structured approach.
    
    # Actually, let me reconsider. For 2x3x8:
    # We can group the 8 columns of B into 4 pairs.
    # For each pair, we compute a 2x2 result from 2x3 * 3x2.
    
    # Let me use a rank-7 decomposition for each 2x2x2 sub-problem within the 2x3x2.
    # But that's more complex.
    
    # Let me just implement the schoolbook properly with clean code.
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
