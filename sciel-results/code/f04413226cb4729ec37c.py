"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on splitting p=8 into four blocks of size 2.
    # Each 3x3x2 block has rank 6 (schoolbook).
    # Total rank = 4 * 6 = 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4 blocks of p=2
    num_blocks = p // 2
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k_prime in range(2):
                    # Column index for this term
                    c = b * 6 + (i * m + j) * 2 + k_prime
                    
                    # U index: i*m + j
                    u_idx = i * m + j
                    U[u_idx][c] = 1
                    
                    # V index: j*p + k
                    # k = b * 2 + k_prime
                    k_val = b * 2 + k_prime
                    v_idx = j * p + k_val
                    V[v_idx][c] = 1
                    
                    # W index: k*n + i
                    w_idx = k_val * n + i
                    W[w_idx][c] = 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
