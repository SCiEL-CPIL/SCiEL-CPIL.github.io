import functools

def construct():
    n, m, p = 3, 5, 5
    
    @functools.lru_cache(maxsize=None)
    def get_decomposition(n, m, p):
        # Returns U, V, W as tuples of tuples for hashability, or just lists if we don't cache
        # Since we need to return lists, we can't cache directly if we mutate them.
        # But we can cache the rank and the split choice, or just recompute.
        # Given small sizes, recomputation is fine.
        
        if n == 1 or m == 1 or p == 1:
            rank = n * m * p
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
            return U, V, W

        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(4)]
            
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[1][1] = 1; W[3][1] = -1
            
            U[0][2] = 1
            V[2][2] = 1; V[3][2] = -1
            W[2][2] = 1; W[3][2] = 1
            
            U[3][3] = 1
            V[1][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[1][3] = 1
            
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[2][4] = 1
            
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[3][6] = 1
            
            return U, V, W

        best_rank = n * m * p
        best_U, best_V, best_W = None, None, None
        
        # Helper to combine
        def combine(U1, V1, W1, U2, V2, W2, n, m, p, split_dim, split_idx):
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            new_U = [[0] * total_rank for _ in range(n * m)]
            new_V = [[0] * total_rank for _ in range(m * p)]
            new_W = [[0] * total_rank for _ in range(p * n)]
            
            if split_dim == 'm':
                m1 = split_idx
                m2 = m - m1
                for i in range(n):
                    for j in range(m1):
                        for c in range(rank1):
                            new_U[i * m + j][c] = U1[i * m1 + j][c]
                for i in range(n):
                    for j in range(m2):
                        for c in range(rank2):
                            new_U[i * m + (m1 + j)][rank1 + c] = U2[i * m2 + j][c]
                for j in range(m1):
                    for k in range(p):
                        for c in range(rank1):
                            new_V[j * p + k][c] = V1[j * p + k][c]
                for j in range(m2):
                    for k in range(p):
                        for c in range(rank2):
                            new_V[(m1 + j) * p + k][rank1 + c] = V2[j * p + k][c]
                for k in range(p):
                    for i in range(n):
                        for c in range(rank1):
                            new_W[k * n + i][c] = W1[k * n + i][c]
                for k in range(p):
                    for i in range(n):
                        for c in range(rank2):
                            new_W[k * n + i][rank1 + c] = W2[k * n + i][c]
            elif split_dim == 'n':
                n1 = split_idx
                n2 = n - n1
                for i in range(n1):
                    for j in range(m):
                        for c in range(rank1):
                            new_U[i * m + j][c] = U1[i * m + j][c]
                for i in range(n2):
                    for j in range(m):
                        for c in range(rank2):
                            new_U[(n1 + i) * m + j][rank1 + c] = U2[i * m + j][c]
                for j in range(m):
                    for k in range(p):
                        for c in range(rank1):
                            new_V[j * p + k][c] = V1[j * p + k][c]
                for j in range(m):
                    for k in range(p):
                        for c in range(rank2):
                            new_V[j * p + k][rank1 + c] = V2[j * p + k][c]
                for k in range(p):
                    for i in range(n1):
                        for c in range(rank1):
                            new_W[k * n + i][c] = W1[k * n1 + i][c]
                for k in range(p):
                    for i in range(n2):
                        for c in range(rank2):
                            new_W[k * n + (n1 + i)][rank1 + c] = W2[k * n2 + i][c]
            elif split_dim == 'p':
                p1 = split_idx
                p2 = p - p1
                for i in range(n):
                    for j in range(m):
                        for c in range(rank1):
                            new_U[i * m + j][c] = U1[i * m + j][c]
                for i in range(n):
                    for j in range(m):
                        for c in range(rank2):
                            new_U[i * m + j][rank1 + c] = U2[i * m + j][c]
                for j in range(m):
                    for k in range(p1):
                        for c in range(rank1):
                            new_V[j * p + k][c] = V1[j * p1 + k][c]
                for j in range(m):
                    for k in range(p2):
                        for c in range(rank2):
                            new_V[j * p + (p1 + k)][rank1 + c] = V2[j * p2 + k][c]
                for k in range(p1):
                    for i in range(n):
                        for c in range(rank1):
                            new_W[k * n + i][c] = W1[k * n + i][c]
                for k in range(p2):
                    for i in range(n):
                        for c in range(rank2):
                            new_W[(p1 + k) * n + i][rank1 + c] = W2[k * n + i][c]
                            
            return new_U, new_V, new_W, total_rank

        # Split m
        for m1 in range(1, m):
            U1, V1, W1 = get_decomposition(n, m1, p)
            U2, V2, W2 = get_decomposition(n, m - m1, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                U, V, W, r = combine(U1, V1, W1, U2, V2, W2, n, m, p, 'm', m1)
                best_U, best_V, best_W, best_rank = U, V, W, r

        # Split n
        for n1 in range(1, n):
            U1, V1, W1 = get_decomposition(n1, m, p)
            U2, V2, W2 = get_decomposition(n - n1, m, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                U, V, W, r = combine(U1, V1, W1, U2, V2, W2, n, m, p, 'n', n1)
                best_U, best_V, best_W, best_rank = U, V, W, r

        # Split p
        for p1 in range(1, p):
            U1, V1, W1 = get_decomposition(n, m, p1)
            U2, V2, W2 = get_decomposition(n, m, p - p1)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                U, V, W, r = combine(U1, V1, W1, U2, V2, W2, n, m, p, 'p', p1)
                best_U, best_V, best_W, best_rank = U, V, W, r

        return best_U, best_V, best_W

    U, V, W = get_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
