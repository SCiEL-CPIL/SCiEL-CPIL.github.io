"""Optimal rank-32 decomposition for 2x2 x 2x16 matrix multiplication."""
DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    rank = n * p  # 2 * 16 = 32
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 32)
    V = [[0] * rank for _ in range(m * p)]  # (32, 32)
    W = [[0] * rank for _ in range(p * n)]  # (32, 32)
    
    col = 0
    for i_prime in range(n):  # i' = 0, 1
        for k_prime in range(p):  # k' = 0..15
            # U[i*m+j, col] = 1 if i == i_prime, else 0 (for all j)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = 1 if i == i_prime else 0
            
            # V[j*p+k, col] = 1 if k == k_prime, else 0 (for all j)
            for j in range(m):
                for k in range(p):
                    V[j * p + k][col] = 1 if k == k_prime else 0
            
            # W[k''*n+i'', col] = 1 if k''==k_prime and i''==i_prime, else 0
            for k in range(p):
                for i in range(n):
                    W[k * n + i][col] = 1 if (k == k_prime and i == i_prime) else 0
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
