import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Trivial rank: n * m * p = 4 * 4 * 6 = 96
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 96
    V = [[0] * rank for _ in range(m * p)] # 24 x 96
    W = [[0] * rank for _ in range(p * n)] # 24 x 96
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A element (i, j)
                u_idx = i * m + j
                # B element (j, k)
                v_idx = j * p + k
                # C element (i, k) -> W index k * n + i
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
