"""Rewritten program with a lower rank decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            col = i * m + j
            
            # U[i*m + j, col] = 1
            U[i * m + j][col] = 1
            
            # V[j*p + k, col] = 1 for all k
            for k in range(p):
                V[j * p + k][col] = 1
                
            # W[k*n + i_out, col] = 1 if i_out == i, else 0
            for k in range(p):
                for i_out in range(n):
                    if i_out == i:
                        W[k * n + i_out][col] = 1
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
