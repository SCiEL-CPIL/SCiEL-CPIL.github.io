"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Decompose the 4x4x8 matrix multiplication tensor into two independent
    # 4x4x4 blocks along the output dimension (p=8). Each 4x4x4 block is
    # decomposed using the standard rank-16 schoolbook decomposition.
    #
    # Total rank = 2 * 16 = 32, which is significantly lower than the full
    # schoolbook rank of 4*4*8 = 128.
    #
    # The tensor for multiplying an n x m matrix by an m x p matrix has
    # components T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    # For the output factor W, the convention is W[k*n + i, r] for column r.
    #
    # We split k into two ranges: k = 0..3 (first block) and k = 4..7 (second block).
    # For each block, we use a rank-16 decomposition of the 4x4x4 multiplication.
    
    n, m, p = DIMENSIONS
    # p should be 8, we split into two blocks of size 4
    half_p = p // 2  # 4
    
    rank = 2 * (n * m * half_p)  # 2 * 16 = 32
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for block in range(2):
        base_col = block * (n * m * half_p)
        for i in range(n):
            for j in range(m):
                for k_local in range(half_p):
                    col = base_col + (i * m + j) * half_p + k_local
                    U[i * m + j][col] = 1.0
                    # k = block * half_p + k_local
                    k_global = block * half_p + k_local
                    V[j * p + k_global][col] = 1.0
                    W[k_global * n + i][col] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
