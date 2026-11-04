"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 2 * 4 = 8
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U is identity matrix of size n*m
    for i in range(n * m):
        U[i][i] = 1
        
    # V construction
    # Row index r corresponds to j, k where r = j * p + k
    # Col index c corresponds to i, j where c = i * m + j
    # V[r, c] = 1 if j(r) == j(c)
    for r in range(m * p):
        j_r = r // p
        for c in range(rank):
            j_c = c % m
            if j_r == j_c:
                V[r][c] = 1
                
    # W construction
    # Row index r corresponds to k, i where r = k * n + i
    # Col index c corresponds to i, j where c = i * m + j
    # W[r, c] = 1 if i(r) == i(c)
    for r in range(p * n):
        i_r = r % n
        for c in range(rank):
            i_c = c // m
            if i_r == i_c:
                W[r][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
