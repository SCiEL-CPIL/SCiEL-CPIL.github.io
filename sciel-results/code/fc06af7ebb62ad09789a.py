import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_TIME_LIMIT = 140

def check_time_remaining():
    return _START_TIME + _TIME_LIMIT - time.time()

def validate_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Validate that U, V, W form a valid decomposition with half-integer entries"""
    rank = len(U[0])
    # Check half-integer constraint
    for i in range(len(U)):
        for s in range(rank):
            if abs(U[i][s] * 2 - round(U[i][s] * 2)) > tol:
                return False
    for i in range(len(V)):
        for s in range(rank):
            if abs(V[i][s] * 2 - round(V[i][s] * 2)) > tol:
                return False
    for i in range(len(W)):
        for s in range(rank):
            if abs(W[i][s] * 2 - round(W[i][s] * 2)) > tol:
                return False
    
    # Check the 1-equations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s] for s in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    
    # Check the 0-equations
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                continue
                            s = sum(U[i*m+j][s] * V[jp*p+k][s] * W[kp*n+ip][s] for s in range(rank))
                            if abs(s) > tol:
                                return False
    return True

def schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = schoolbook_decomposition(n, m, p)
    
    # Precompute index lists
    idx_1 = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    idx_0 = [(i, j, jp, k, kp, ip) for i in range(n) for j in range(m) 
             for jp in range(m) for k in range(p) for kp in range(p) for ip in range(n)
             if not (i == ip and j == jp and k == kp)]
    
    # Try to find lower rank decompositions
    for target_rank in range(59, 55, -1):
        if check_time_remaining() < 30:
            break
            
        # Initialize with schoolbook truncated
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i*m*p + j*p + k
                    if col < target_rank:
                        U_init[i*m+j, col] = 1
                        V_init[j*p+k, col] = 1
                        W_init[k*n+i, col] = 1
        
        def residual(x):
            U = x[:n*m*target_rank].reshape(n*m, target_rank)
            V = x[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
            W = x[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
            
            res = []
            for (i, j, k) in idx_1:
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                res.append(val - 1.0)
            for (i, j, jp, k, kp, ip) in idx_0:
                val = np.dot(U[i*m+j], V[jp*p+k] * W[kp*n+ip])
                res.append(val)
            return np.array(res)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            sol = least_squares(residual, x0, method='lm', max_nfev=200)
            if np.max(np.abs(sol.fun)) < 1e-5:
                U_sol = sol.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = sol.x[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
                W_sol = sol.x[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
                
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if validate_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-4):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
    
    return result
