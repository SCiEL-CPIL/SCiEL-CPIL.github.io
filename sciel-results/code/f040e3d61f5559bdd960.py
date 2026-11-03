def construct():
    n, m, p = 2, 4, 6
    # Strategy: pad n from 2 to 3, use Laderman rank-13 for 3x3x3x3 blocks
    # Split p=6 into 3+3, m=4 into 3+1
    # Block 1: 3x3 x 3x3 (rows 0-2 of A, cols 0-2; rows 0-2 of B, cols 0-2) -> Laderman rank 13
    # Block 2: 3x1 x 1x3 (rows 0-2 of A, col 3; row 3 of B, cols 0-2) -> schoolbook rank 3
    # Block 3: 3x3 x 3x3 (rows 0-2 of A, cols 0-2; rows 0-2 of B, cols 3-5) -> Laderman rank 13
    # Block 4: 3x1 x 1x3 (rows 0-2 of A, col 3; row 3 of B, cols 3-5) -> schoolbook rank 3
    # Total rank: 13 + 3 + 13 + 3 = 32
    # But we only need n=2 rows in the output, so we pad A with a zero row.
    # The padded row in A contributes nothing, so we can ignore it.
    # However, Laderman operates on 3x3 blocks. We pad A to 3x4, B to 4x6.
    # The extra row in A (row 2) is zero, so any product involving it is zero.
    # We still need to use Laderman's 13 factors for the 3x3x3x3 blocks.
    
    # Laderman's rank-13 decomposition for 3x3 matrix multiplication.
    # A, B, C are 3x3 matrices.
    # A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    # C = A*B
    
    # The 13 factors are (u, v, w) where u is in A-space (9 dim), v in B-space (9 dim), w in C-space (9 dim)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # W convention: W[k*n+i] with n=3, so W[0]=C[0,0], W[1]=C[1,0], W[2]=C[2,0], W[3]=C[0,1], etc.
    # Actually W[k*3+i] for k=0,1,2 and i=0,1,2.
    
    # Laderman's algorithm:
    # p1 = (a11+a21)*(b11+b22)
    # p2 = a11*b22 - a21*b11
    # p3 = (a12+a13)*(b21+b31)
    # p4 = a13*b21 - a12*b31
    # p5 = (a22+a32)*(b13+b33)
    # p6 = a32*b13 - a22*b33
    # p7 = (a23+a33)*(b12+b32)
    # p8 = a33*b12 - a23*b32
    # p9 = (a11+a31)*(b23+b32)
    # p10 = a31*b23 - a11*b32
    # p11 = (a22+a33)*(b11+b21)
    # p12 = a33*b11 - a22*b21
    # p13 = (a13+a33)*(b22+b31)
    # p14 = a33*b22 - a13*b31  -- wait, that's 14. Let me check.
    
    # Actually Laderman has 13 products. Let me use a known correct set.
    # From various sources, here's one:
    # M1 = (a11 + a21) * (b11 + b22)
    # M2 = a11 * b22 - a21 * b11
    # M3 = (a12 + a13) * (b21 + b31)
    # M4 = a13 * b21 - a12 * b31
    # M5 = (a22 + a32) * (b13 + b33)
    # M6 = a32 * b13 - a22 * b33
    # M7 = (a23 + a33) * (b12 + b32)
    # M8 = a33 * b12 - a23 * b32
    # M9 = (a11 + a31) * (b23 + b32)
    # M10 = a31 * b23 - a11 * b32
    # M11 = (a22 + a33) * (b11 + b21)
    # M12 = a33 * b11 - a22 * b21
    # M13 = (a13 + a33) * (b22 + b31)
    # M14 = a33 * b22 - a13 * b31
    # That's 14. I need to find the correct 13.
    
    # Let me just use schoolbook for everything. Rank 2*4*6 = 48.
    # Or better, let me use Strassen tiling.
    
    # Actually, let me just implement schoolbook. It's guaranteed correct.
    
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]  # 8 x 48
    V = [[0] * rank for _ in range(m * p)]  # 24 x 48
    W = [[0] * rank for _ in range(p * n)]  # 12 x 48
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
