import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = start_time + 110
    
    T = np.ones((n, m, p))
    
    for rank in [27, 26]:
        if time.time() > deadline:
            break
        
        for trial in range(100):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 7919 + rank * 1000)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            best_err = float('inf')
            for it in range(3000):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        M = np.zeros((rank, p))
                        for k in range(p):
                            M[:, k] = V[j * p + k, :] * W[k * n + i, :]
                        U[a, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        b = j * p + k
                        M2 = np.zeros((rank, n))
                        for i in range(n):
                            M2[:, i] = U[i * m + j, :] * W[k * n + i, :]
                        V[b, :] = np.linalg.lstsq(M2.T, T[:, j, k], rcond=None)[0]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        c = k * n + i
                        M3 = np.zeros((rank, m))
                        for j in range(m):
                            M3[:, j] = U[i * m + j, :] * V[j * p + k, :]
                        W[c, :] = np.linalg.lstsq(M3.T, T[i, :, k], rcond=None)[0]
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0) ** 2
                
                if err < 1e-12:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                if err < best_err:
                    best_err = err
                    if best_err < 1e-6:
                        # Try to polish
                        for polish_it in range(1000):
                            if time.time() > deadline:
                                break
                            # Same updates
                            for i in range(n):
                                for j in range(m):
                                    a = i * m + j
                                    M = np.zeros((rank, p))
                                    for k in range(p):
                                        M[:, k] = V[j * p + k, :] * W[k * n + i, :]
                                    U[a, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                            for j in range(m):
                                for k in range(p):
                                    b = j * p + k
                                    M2 = np.zeros((rank, n))
                                    for i in range(n):
                                        M2[:, i] = U[i * m + j, :] * W[k * n + i, :]
                                    V[b, :] = np.linalg.lstsq(M2.T, T[:, j, k], rcond=None)[0]
                            for k in range(p):
                                for i in range(n):
                                    c = k * n + i
                                    M3 = np.zeros((rank, m))
                                    for j in range(m):
                                        M3[:, j] = U[i * m + j, :] * V[j * p + k, :]
                                    W[c, :] = np.linalg.lstsq(M3.T, T[i, :, k], rcond=None)[0]
                            err = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                                        err += (val - 1.0) ** 2
                            if err < 1e-12:
                                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                if it % 200 == 0 and err > 1e1:
                    break
    
    # Fallback: rank-28 Strassen
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1
    
    add_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
    add_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
    add_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
    add_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])
    
    return {"U": U, "V": V, "W": W}
