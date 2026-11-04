    def construct():
        n, m, p = 3, 5, 7
        rank = m # 5
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(m):
            # U: column r is 1s, others 0
            for i in range(n):
                U[i * m + r][r] = 1
            
            # V: row r is 1s, others 0
            for k in range(p):
                V[r * p + k][r] = 1
            
            # W: all 1s
            for k in range(p):
                for i in range(n):
                    W[k * n + i][r] = 1
        
        return {"U": U, "V": V, "W": W}
    