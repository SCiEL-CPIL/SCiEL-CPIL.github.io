"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(s - 1) > 1e-8:
                        return False
        return True
    
    def round_to_half(x):
        return round(x * 2) / 2
    
    # Try to find a lower rank decomposition
    # Use iterative method: fix two factors, solve for the third
    best_result = None
    best_rank = rank_school
    
    for rank in range(21, min(36, rank_school + 1)):
        if time.time() > deadline - 40:
            break
        
        np.random.seed(rank)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        for iteration in range(200):
            if time.time() > deadline - 30:
                break
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(rank)] for j in range(m)])
                    try:
                        x = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        W[k*n+i, :] = x
                    except:
                        pass
            
            if time.time() > deadline - 30:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.array([[V[j*p+k, c] * W[k*n+i, c] for c in range(rank)] for k in range(p)])
                    try:
                        x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        U[i*m+j, :] = x
                    except:
                        pass
            
            if time.time() > deadline - 30:
                break
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(rank)] for i in range(n)])
                    try:
                        x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        V[j*p+k, :] = x
                    except:
                        pass
        
        # Check convergence and round to half-integers
        U_rounded = np.array([[round_to_half(x) for x in row] for row in U])
        V_rounded = np.array([[round_to_half(x) for x in row] for row in V])
        W_rounded = np.array([[round_to_half(x) for x in row] for row in W])
        
        if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), rank):
            best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            best_rank = rank
            break
    
    if best_result is not None:
        return best_result
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

