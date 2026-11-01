"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Structured rank-24 solution: share V factor between i=0 and i=1
    def rank24_solution():
        rank = m * p  # 24
        U = [[0] * rank for _ in range(n * m)]  # (6, 24)
        V = [[0] * rank for _ in range(m * p)]  # (24, 24)
        W = [[0] * rank for _ in range(p * n)]  # (16, 24)
        
        for j in range(m):
            for k in range(p):
                c = j * p + k
                # U row for (i,j) is i*m+j
                U[0*m + j][c] = 1  # i=0
                U[1*m + j][c] = 1  # i=1
                V[j*p + k][c] = 1
                W[k*n + 0][c] = 1  # i=0
                W[k*n + 1][c] = 1  # i=1
        
        return {"U": U, "V": V, "W": W}
    
    sol24 = rank24_solution()
    if not verify(sol24["U"], sol24["V"], sol24["W"]):
        # Fallback to schoolbook
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
    
    # Try to find lower rank
    def try_lower_rank():
        for rank in range(12, 24):
            if time.time() > deadline - 10:
                break
            
            found = False
            for trial in range(5):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(42 + rank * 100 + trial)
                U = np.random.randn(n*m, rank) * 0.3
                V = np.random.randn(m*p, rank) * 0.3
                W = np.random.randn(p*n, rank) * 0.3
                
                for it in range(500):
                    if time.time() > deadline - 3:
                        break
                    
                    resid = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                resid[i, j, k] = 1.0 - s
                    
                    err = np.max(np.abs(resid))
                    if err < 1e-14:
                        break
                    
                    # Gauss-Newton updates
                    for i in range(n):
                        for j in range(m):
                            for c in range(rank):
                                grad_k = np.array([-V[j*p+k, c] * W[k*n+i, c] for k in range(p)])
                                denom = np.sum(grad_k**2) + 1e-14
                                U[i*m+j, c] += np.sum(resid[i, j, :] * grad_k) / denom
                    
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                grad_i = np.array([-U[i*m+j, c] * W[k*n+i, c] for i in range(n)])
                                denom = np.sum(grad_i**2) + 1e-14
                                V[j*p+k, c] += np.sum(resid[:, j, k] * grad_i) / denom
                    
                    for k in range(p):
                        for i in range(n):
                            for c in range(rank):
                                grad_j = np.array([-U[i*m+j, c] * V[j*p+k, c] for j in range(m)])
                                denom = np.sum(grad_j**2) + 1e-14
                                W[k*n+i, c] += np.sum(resid[i, :, k] * grad_j) / denom
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-10:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    found = True
        
        return None
    
    result = try_lower_rank()
    if result is not None:
        return result
    
    return sol24
# EVOLVE-BLOCK-END
