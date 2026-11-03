"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # (16, 44)
    
    # Part 1: A1 * B1 where A1 is 2x1 (column 0 of A), B1 is 1x8 (row 0 of B)
    # Schoolbook: 2*1*8 = 16 terms
    col = 0
    for i in range(n):  # i in {0,1}
        for k in range(p):  # k in {0,...,7}
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: A2 * B2 where A2 is 2x2 (columns 1,2 of A), B2 is 2x8 (rows 1,2 of B)
    # Split B2 into 4 blocks of 2x2, use Strassen for each: 4*7 = 28 terms
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, -1, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    gamma = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    for s in range(4):
        for r in range(7):
            for i_prime in range(2):
                for j_prime in range(2):
                    U[i_prime * m + (1 + j_prime)][col] = alpha[r][i_prime * 2 + j_prime]
            for j_prime in range(2):
                for l_prime in range(2):
                    V[(1 + j_prime) * p + (2 * s + l_prime)][col] = beta[r][j_prime * 2 + l_prime]
            for l_prime in range(2):
                for i_prime in range(2):
                    W[(2 * s + l_prime) * n + i_prime][col] = gamma[r][l_prime * 2 + i_prime]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
