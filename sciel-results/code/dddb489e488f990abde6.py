    def construct():
        n, m, p = 3, 3, 5
        # Split into 3x3x3 and 3x3x2
        p1 = 3
        p2 = 2
        
        # 3x3x3 schoolbook
        rank1 = n * m * p1
        U1 = [[0] * rank1 for _ in range(n * m)]
        V1 = [[0] * rank1 for _ in range(m * p1)]
        W1 = [[0] * rank1 for _ in range(p1 * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p1):
                    U1[i * m + j][col] = 1
                    V1[j * p1 + k][col] = 1
                    W1[k * n + i][col] = 1
                    col += 1
        
        # 3x3x2 schoolbook
        rank2 = n * m * p2
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p2)]
        W2 = [[0] * rank2 for _ in range(p2 * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    U2[i * m + j][col] = 1
                    V2[j * p2 + k][col] = 1
                    W2[k * n + i][col] = 1
                    col += 1
        
        # Combine
        total_rank = rank1 + rank2
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for i in range(n * m):
            for c in range(rank1):
                U[i][c] = U1[i][c]
            for c in range(rank2):
                U[i][rank1 + c] = U2[i][c]
        
        # V: rows 0..8 are V1, rows 9..14 are V2
        for i in range(m * p1):
            for c in range(rank1):
                V[i][c] = V1[i][c]
        for i in range(m * p2):
            for c in range(rank2):
                V[m * p1 + i][rank1 + c] = V2[i][c]
        
        # W: rows 0..8 are W1, rows 9..14 are W2
        for i in range(p1 * n):
            for c in range(rank1):
                W[i][c] = W1[i][c]
        for i in range(p2 * n):
            for c in range(rank2):
                W[p1 * n + i][rank1 + c] = W2[i][c]
        
        return {"U": U, "V": V, "W": W}
    