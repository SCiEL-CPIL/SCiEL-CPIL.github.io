"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # F(4) Winograd-style algorithm for 4x4 matrix multiplication
    # The F(4) algorithm uses 9 multiplications for the 4x4x4 case.
    # We extend this to 4x4x5 by adding 16 more terms for the 5th column.
    
    # F(4) for 4x4x4:
    # The 9 products are:
    # m0 = (a00+a01+a02+a03)(b00+b10+b20+b30)
    # m1 = (a00-a01+a02-a03)(b00-b10+b20-b30)
    # m2 = (a00+a01-a02-a03)(b00+b10-b20-b30)
    # m3 = (a00-a01-a02+a03)(b00-b10-b20+b30)
    # m4 = (a10+a11+a12+a13)(b01+b11+b21+b31)
    # m5 = (a10-a11+a12-a13)(b01-b11+b21-b31)
    # m6 = (a10+a11-a12-a13)(b01+b11-b21-b31)
    # m7 = (a10-a11-a12+a13)(b01-b11-b21+b31)
    # m8 = (a20+a21+a22+a23)(b02+b12+b22+b32) -- This is not correct for standard F(4)
    
    # Actually, the standard F(4) algorithm for 4x4 matrix multiplication
    # uses a 4x4 Hadamard-like transform. Let me use the correct F(4) matrices.
    
    # F(4) algorithm:
    # U matrix (16x9) for A (4x4):
    # Row i*m+j corresponds to A[i][j]
    # V matrix (16x9) for B (4x4):
    # Row j*p+k corresponds to B[j][k]
    # W matrix (16x9) for C (4x4):
    # Row k*n+i corresponds to C[i][k]
    
    # The F(4) algorithm uses the following:
    # For each of the 4 rows of A and 4 columns of B, we have 4x4=16 entries.
    # But we only need 9 products total.
    
    # Let me use the known F(4) decomposition:
    # m0 = (a00+a01+a02+a03)(b00+b10+b20+b30)
    # m1 = (a00-a01+a02-a03)(b00-b10+b20-b30)
    # m2 = (a00+a01-a02-a03)(b00+b10-b20-b30)
    # m3 = (a00-a01-a02+a03)(b00-b10-b20+b30)
    # m4 = (a10+a11+a12+a13)(b01+b11+b21+b31)
    # m5 = (a10-a11+a12-a13)(b01-b11+b21-b31)
    # m6 = (a10+a11-a12-a13)(b01+b11-b21-b31)
    # m7 = (a10-a11-a12+a13)(b01-b11-b21+b31)
    # m8 = (a20+a21+a22+a23)(b02+b12+b22+b32)
    
    # Wait, this is not symmetric. The correct F(4) should be symmetric.
    
    # Let me use the standard Winograd F(4) algorithm:
    # The F(4) algorithm for 4x4 matrix multiplication uses 9 multiplications.
    # The U, V, W matrices are:
    
    # U (16x9):
    U_f4 = [
        [1, 1, 1, 1, 0, 0, 0, 0, 0],  # a00
        [1, -1, 1, -1, 0, 0, 0, 0, 0],  # a01
        [1, 1, -1, -1, 0, 0, 0, 0, 0],  # a02
        [1, -1, -1, 1, 0, 0, 0, 0, 0],  # a03
        [0, 0, 0, 0, 1, 1, 1, 1, 0],  # a10
        [0, 0, 0, 0, 1, -1, 1, -1, 0],  # a11
        [0, 0, 0, 0, 1, 1, -1, -1, 0],  # a12
        [0, 0, 0, 0, 1, -1, -1, 1, 0],  # a13
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a20
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a21
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a22
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a23
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a30
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a31
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a32
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a33
    ]
    
    # This is not correct. The F(4) algorithm should be symmetric.
    
    # Let me use a different approach. I'll use the schoolbook algorithm for 4x4x5,
    # which has rank 4*4*5 = 80. But that's too high.
    
    # Actually, let me use a simpler approach: use the rank-25 decomposition
    # by combining F(4) for the first 4 columns and schoolbook for the 5th column.
    
    # F(4) for 4x4x4 has rank 9.
    # Schoolbook for 4x4x1 has rank 4.
    # Total rank = 9 + 4 = 13.
    
    # But I need to construct the matrices correctly.
    
    # For F(4) 4x4x4:
    # The 9 products are:
    # m0 = (a00+a01+a02+a03)(b00+b10+b20+b30)
    # m1 = (a00-a01+a02-a03)(b00-b10+b20-b30)
    # m2 = (a00+a01-a02-a03)(b00+b10-b20-b30)
    # m3 = (a00-a01-a02+a03)(b00-b10-b20+b30)
    # m4 = (a10+a11+a12+a13)(b01+b11+b21+b31)
    # m5 = (a10-a11+a12-a13)(b01-b11+b21-b31)
    # m6 = (a10+a11-a12-a13)(b01+b11-b21-b31)
    # m7 = (a10-a11-a12+a13)(b01-b11-b21+b31)
    # m8 = (a20+a21+a22+a23)(b02+b12+b22+b32)
    
    # This is still not correct. Let me use the known F(4) matrices.
    
    # Actually, let me just use the schoolbook decomposition for now, which is guaranteed to work.
    # Rank = 4*4*5 = 80
    
    rank = n * m * p
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
