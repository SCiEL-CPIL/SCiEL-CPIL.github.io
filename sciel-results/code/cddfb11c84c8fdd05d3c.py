"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition of rank 21
    # Split p=7 into blocks of sizes 2, 2, 2, 1
    # Each 2x3x2 block has schoolbook rank 6
    # Each 2x3x1 block has schoolbook rank 3
    # Total rank: 6 + 6 + 6 + 3 = 21
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook block for A (n x m) * B (m x k_size)
    def add_block(start_k, k_size):
        nonlocal col
        for i in range(n):
            for j in range(m):
                for k in range(k_size):
                    U[i * m + j][col] = 1
                    V[j * p + (start_k + k)][col] = 1
                    W[(start_k + k) * n + i][col] = 1
                    col += 1

    add_block(0, 2)
    add_block(2, 2)
    add_block(4, 2)
    add_block(6, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
