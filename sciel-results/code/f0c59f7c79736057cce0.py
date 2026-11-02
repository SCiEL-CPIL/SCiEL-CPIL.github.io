"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the p=9 dimension into 4+4+1.
    # Block 1: 4x4x4
    # Block 2: 4x4x4
    # Block 3: 4x4x1
    # Total rank = 16 + 16 + 16 = 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(start_r, k_offset, k_count):
        for i in range(n):
            for j in range(m):
                for k in range(k_count):
                    r = start_r + (i * m + j) * k_count + k
                    # U index: i*m + j
                    U[i * m + j][r] = 1
                    # V index: j*p + (k_offset + k)
                    V[j * p + (k_offset + k)][r] = 1
                    # W index: (k_offset + k)*n + i
                    W[(k_offset + k) * n + i][r] = 1

    # Block 1: k=0..3
    add_block(0, 0, 4)
    # Block 2: k=4..7
    add_block(16, 4, 4)
    # Block 3: k=8
    add_block(32, 8, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
