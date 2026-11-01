"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Attempt Strassen (Rank 7)
    # U: (n*m, rank) -> (4, 7)
    # V: (m*p, rank) -> (4, 7)
    # W: (p*n, rank) -> (4, 7)
    
    U_strassen = [
        [1, 0, 0, 0, 1, 0, 0],  # a
        [0, 0, 1, 0, 0, 1, 0],  # b
        [0, 1, 0, 1, 0, 0, 1],  # c
        [1, 1, 0, 0, 0, 1, 1]   # d
    ]
    
    V_strassen = [
        [1, 1, 0, 0, 1, 0, 0],  # e
        [0, 0, 1, 0, 0, 0, 1],  # f
        [0, 0, 0, 1, -1, 0, 0], # g
        [1, 0, -1, -1, 0, 1, -1] # h
    ]
    
    W_strassen = [
        [1, 0, 0, 1, 0, 0, 0],  # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 1, 0],  # C01 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
    ]
    
    # Verification
    # Tensor T_{u,v,w} = 1 if u=2i+j, v=2j+k, w=2k+i
    # Decomposition: sum_r U[u,r] * V[v,r] * W[w,r]
    
    def verify(U, V, W):
        rank = len(U[0])
        # Build target tensor
        T = [[[0]* (p*n) for _ in range(m*p)] for _ in range(n*m)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u][v][w] = 1
        
        # Compute decomposition
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    val = 0
                    for r in range(rank):
                        val += U[u][r] * V[v][r] * W[w][r]
                    if val != T[u][v][w]:
                        return False
        return True

    if verify(U_strassen, V_strassen, W_strassen):
        return {"U": U_strassen, "V": V_strassen, "W": W_strassen}
    else:
        # Fallback to Schoolbook (Rank 8)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
