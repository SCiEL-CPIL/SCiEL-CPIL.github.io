"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    # Decompose 2x3x9 into 3 blocks of 2x3x3
    # Each 2x3x3 block: split into 2x3x2 (rank 3) + 2x3x1 (rank 3) = rank 6
    # Total rank: 3 * 6 = 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 27 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    col = 0
    # For each block b (0,1,2), columns 3*b, 3*b+1, 3*b+2
    for b in range(3):
        k0 = 3 * b  # first column index in this block
        
        # 2x3x2 part: columns k0 and k0+1 (rank 3)
        # Term t (t=0,1,2): j=t, handles k0 and k0+1
        for t in range(3):
            # U: row i*m+j = i*3+t
            for i in range(n):
                U[i * m + t][col] = 1
            # V: row j*p+k = t*9 + (k0+s) for s=0,1
            for s in range(2):
                V[t * p + k0 + s][col] = 1
            # W: row k*n+i = (k0+s)*2 + i for s=0,1
            for s in range(2):
                for i in range(n):
                    W[(k0 + s) * n + i][col] = 1
            col += 1
        
        # 2x3x1 part: column k0+2 (rank 3)
        # Term t (t=0,1,2): j=t, handles k0+2
        for t in range(3):
            for i in range(n):
                U[i * m + t][col] = 1
            V[t * p + k0 + 2][col] = 1
            for i in range(n):
                W[(k0 + 2) * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
