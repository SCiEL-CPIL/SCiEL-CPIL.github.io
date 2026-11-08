"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Split m=3 into m1=1 and m2=2
    m1 = 1
    m2 = 2
    
    # Part 1: (n, m1, p) = (2, 1, 6) - schoolbook, rank = n*m1*p = 12
    rank1 = n * m1 * p
    # Part 2: (n, m2, p) = (2, 2, 6) - use Strassen for each of p/2 = 3 blocks, rank = 3*7 = 21
    rank2 = (p // 2) * 7  # 3 * 7 = 21
    
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 33)
    V = [[0] * total_rank for _ in range(m * p)]  # (18, 33)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 33)
    
    col = 0
    
    # Part 1: (2, 1, 6) schoolbook
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Strassen matrices for 2x2 * 2x2 -> 2x2
    # U: rows = [a11, a12, a21, a22], cols = [p1..p7]
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # a11
        [0, 0, 0, 0, 1, 0, 1],   # a12
        [0, 1, 0, 0, 0, 1, 0],   # a21
        [1, 1, 0, 1, 0, 0, -1],  # a22
    ]
    # V: rows = [b11, b12, b21, b22], cols = [p1..p7]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # b11
        [0, 0, 1, 0, 0, 1, 0],   # b12
        [0, 0, 0, 1, 0, 0, 1],   # b21
        [1, 0, -1, 0, 1, 0, 1],  # b22
    ]
    # W: rows = [c11, c12, c21, c22], cols = [p1..p7]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # c11
        [0, 0, 1, 0, 1, 0, 0],   # c12
        [0, 1, 0, 1, 0, 0, 0],   # c21
        [1, -1, 1, 0, 0, 1, 0],  # c22
    ]
    
    # Part 2: (2, 2, 6) using Strassen for each of the 3 blocks of size 2x2
    for block in range(3):
        for s in range(7):
            # U: A2 element at (i, j2) -> index i*m + (m1 + j2)
            for i in range(2):
                for j2 in range(2):
                    U[i * m + (m1 + j2)][col] = strassen_U[j2 * 2 + i][s]
            
            # V: B2 element at (j2, k2) -> index (m1 + j2)*p + (block*2 + k2)
            for j2 in range(2):
                for k2 in range(2):
                    V[(m1 + j2) * p + (block * 2 + k2)][col] = strassen_V[j2 * 2 + k2][s]
            
            # W: C element at (k2, i) -> index (block*2 + k2)*n + i
            for k2 in range(2):
                for i in range(2):
                    W[(block * 2 + k2) * n + i][col] = strassen_W[k2 * 2 + i][s]
            
            col += 1
    
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
