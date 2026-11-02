import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 140.0
    
    def schoolbook():
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
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def solve_for_U(V, W, rank):
        U = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i*m+j, :] = x
        return U
    
    def solve_for_V(U, W, rank):
        V = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j*p+k, :] = x
        return V
    
    def solve_for_W(U, V, rank):
        W = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k*n+i, :] = x
        return W
    
    def check_error(U, V, W):
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    # Try ranks from 24 upward
    for rank in range(24, 37):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(100):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 1000 + rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating least squares
            U = solve_for_U(V, W, rank)
            for iteration in range(300):
                if time.time() > deadline - 30:
                    break
                
                V = solve_for_V(U, W, rank)
                W = solve_for_W(U, V, rank)
                U = solve_for_U(V, W, rank)
                
                # Check convergence
                err = check_error(U, V, W)
                
                if err < 1e-10:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break
        
    return schoolbook()
