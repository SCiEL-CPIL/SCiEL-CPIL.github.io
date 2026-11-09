DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3 (rows of A / cols of C), m=3 (cols of A / rows of B), p=5 (cols of B / rows of C)
    # Wait, let me re-read: "multiplying a 3 by 3 matrix by a 3 by 5 matrix"
    # So A is 3x3, B is 3x5, C is 3x5.
    # A: n x m = 3 x 3
    # B: m x p = 3 x 5  
    # C: n x p = 3 x 5
    # U shape: (n*m, rank) = (9, rank) - indexes (i,j) for A
    # V shape: (m*p, rank) = (15, rank) - indexes (j,k) for B
    # W shape: (p*n, rank) = (15, rank) - indexes (k,i) for C, with W[k*n+i, r]
    
    rank = n * m * p  # 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
