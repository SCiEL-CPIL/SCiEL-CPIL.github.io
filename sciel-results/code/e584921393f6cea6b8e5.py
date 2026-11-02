"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        j_r = r % m
        k_r = r // m
        
        # Fill U
        for i in range(n):
            for j in range(m):
                if j == j_r:
                    U[i * m + j][r] = 1
                    
        # Fill V
        for j in range(m):
            for k in range(p):
                if j == j_r and k == k_r:
                    V[j * p + k][r] = 1
                    
        # Fill W
        for k in range(p):
            for i in range(n):
                if k == k_r:
                    W[k * n + i][r] = 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
