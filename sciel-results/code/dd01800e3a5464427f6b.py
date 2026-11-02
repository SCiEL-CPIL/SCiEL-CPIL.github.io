import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 105)
    rank_school = n * m * p
    
    U_school = np.zeros((n * m, rank_school), dtype=np.float64)
    V_school = np.zeros((m * p, rank_school), dtype=np.float64)
    W_school = np.zeros((p * n, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    deadline = time.time() + 115
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    def verify(U, V, W, tol=1e-10):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(r))
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition using random search with local optimization
    for r in range(70, 25, -1):
        if time.time() > deadline:
            break
        
        for init in range(200):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            def err(U, V, W):
                e = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(r))
                            e += (s - 1.0) ** 2
                return e
            
            curr = err(U, V, W)
            for _ in range(50):
                if time.time() > deadline:
                    break
                if curr < 1e-12:
                    break
                idx = np.random.randint(0, 3)
                if idx == 0:
                    U += np.random.randn(n * m, r) * 0.1
                elif idx == 1:
                    V += np.random.randn(m * p, r) * 0.1
                else:
                    W += np.random.randn(p * n, r) * 0.1
                new = err(U, V, W)
                if new < curr:
                    curr = new
            
            if curr < 1e-12 and verify(U, V, W):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
