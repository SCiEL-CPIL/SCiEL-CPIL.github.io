"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-8):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    for target_rank in [30, 40, 50, 55]:
        if time.time() > deadline - 15:
            break
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(seed + target_rank)
            U = rng.randn(n*m, target_rank) * 0.5
            V = rng.randn(m*p, target_rank) * 0.5
            W = rng.randn(p*n, target_rank) * 0.5
            
            found = False
            for it in range(50):
                for i in range(n):
                    for j in range(m):
                        M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                        try:
                            U[i*m+j] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                        try:
                            V[j*p+k] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                        try:
                            W[k*n+i] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                if it % 10 == 9:
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
            if found:
                break
        if best["U"] != U_sb:
            break
    
    return best
# EVOLVE-BLOCK-END
