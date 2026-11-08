"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    np.random.seed(777)
    for _ in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        C = np.zeros((n, p))
        for t in range(rank):
            Ut = np.array([U[i*m+j][t] for i in range(n) for j in range(m)]).reshape(n, m)
            Vt = np.array([V[j*p+k][t] for j in range(m) for k in range(p)]).reshape(m, p)
            Wt = np.array([W[k*n+i][t] for k in range(p) for i in range(n)]).reshape(p, n)
            L = np.sum(Ut * A, axis=1)
            R = np.sum(Vt * B, axis=1)
            C += np.outer(L, R) * Wt
        if not np.allclose(C, C_exp, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110
    
    # Try known low-rank constructions
    # For 2x3 x 3x3, try rank 12, 13, 14, 15
    
    for rank in [12, 13, 14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        # Randomized search with gradient descent
        n_restarts = 30
        for rs in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(rank * 1000 + rs)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Gradient descent
            lr = 0.05
            best_l = float('inf')
            best = (U.copy(), V.copy(), W.copy())
            
            for it in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Compute loss and gradient using random test matrices
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                l = 0.0
                
                for _ in range(2):
                    A = np.random.randn(n, m)
                    B = np.random.randn(m, p)
                    C_exp = A @ B
                    C = np.zeros((n, p))
                    Ls = []
                    Rs = []
                    
                    for t in range(rank):
                        Ut = U.reshape(n, m, rank)[:, :, t]
                        Vt = V.reshape(m, p, rank)[:, :, t]
                        Wt = W.reshape(p, n, rank)[:, :, t]
                        L = np.sum(Ut * A, axis=1)
                        R = np.sum(Vt * B, axis=1)
                        Ls.append(L)
                        Rs.append(R)
                        C += np.outer(L, R) * Wt
                    
                    err = C - C_exp
                    l += np.sum(err ** 2)
                    
                    for t in range(rank):
                        Wt = W.reshape(p, n, rank)[:, :, t]
                        L = Ls[t]
                        R = Rs[t]
                        gW.reshape(p, n, rank)[:, :, t] += 2 * err * np.outer(L, R)
                        
                        Ut = U.reshape(n, m, rank)[:, :, t]
                        Vt = V.reshape(m, p, rank)[:, :, t]
                        
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    gU[i*m+j, t] += 2 * err[i,k] * Wt[k,i] * A[i,j] * R[k]
                        
                        for j in range(m):
                            for k in range(p):
                                for i in range(n):
                                    gV[j*p+k, t] += 2 * err[i,k] * Wt[k,i] * L[i] * B[j,k]
                
                if l < best_l:
                    best_l = l
                    best = (U.copy(), V.copy(), W.copy())
                
                if best_l < 1e-12:
                    break
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                lr *= 0.95
            
            if best_l < 1e-10:
                U_list = [[round(float(best[0][i][t]), 6) for t in range(rank)] for i in range(n*m)]
                V_list = [[round(float(best[1][i][t]), 6) for t in range(rank)] for i in range(m*p)]
                W_list = [[round(float(best[2][i][t]), 6) for t in range(rank)] for i in range(p*n)]
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
