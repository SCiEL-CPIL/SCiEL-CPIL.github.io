def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    def verify_fast(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for c in range(rank):
            W3c = W3[:,:,c].T
            T += U3[:,:,c][:, :, None] * V3[:,:,c][None, :, :] * W3c[None, :, :]
        return np.allclose(T, 1.0, atol=1e-10)
    
    def add_strassen_block(i0, j0, k0, U, V, W, col):
        c = col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    # Try multiple block placements
    block_sets = [
        [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)],
        [(0, 0, 0), (0, 3, 0), (0, 0, 3), (0, 3, 3)],
        [(0, 0, 0), (0, 2, 0), (1, 0, 2), (1, 2, 2)],
        [(0, 0, 0), (0, 2, 2), (1, 0, 0), (1, 2, 0)],
        [(0, 0, 0), (0, 3, 2), (1, 0, 0), (1, 2, 3)],
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for blocks in block_sets:
        if time.time() > DEADLINE:
            break
        
        num_blocks = len(blocks)
        total_rank = num_blocks * 7 + (n*m*p - num_blocks*8)
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for i0, j0, k0 in blocks:
            add_strassen_block(i0, j0, k0, U, V, W, col)
            col += 7
        
        covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered_3d[i0+di][j0+dj][k0+dk] = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered_3d[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        if verify_fast(U, V, W):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = {"U": U, "V": V, "W": W}
    
    if best_result is not None:
        return best_result
    else:
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
