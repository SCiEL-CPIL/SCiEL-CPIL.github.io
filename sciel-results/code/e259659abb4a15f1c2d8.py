import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        # Check if U, V, W are valid half-integer matrices
        if not np.allclose(U * 2, np.round(U * 2)) or not np.allclose(V * 2, np.round(V * 2)) or not np.allclose(W * 2, np.round(W * 2)):
            return False
        # Check decomposition
        # Conditions:
        # 1. sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
        # 2. sum_r U[i,j,r] V[j,k,r] W[k,i',r] == 0 for i' != i
        # 3. sum_r U[i,j,r] V[j,k,r] W[k',i,r] == 0 for k' != k
        
        U_mat = U.reshape(n, m, -1)
        V_mat = V.reshape(m, p, -1)
        W_mat = W.reshape(p, n, -1)
        
        # Compute tensor T[i, j, k, i', k'] = sum_r U[i,j,r] V[j,k,r] W[k',i',r]
        # We only need to check specific slices.
        
        # Check 1: i' = i, k' = k
        # T[i, j, k, i, k] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i, j, :] * V_mat[j, k, :] * W_mat[k, i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        
        # Check 2: i' != i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        if ip == i: continue
                        val = np.sum(U_mat[i, j, :] * V_mat[j, k, :] * W_mat[k, ip, :])
                        if not np.isclose(val, 0.0, atol=1e-9):
                            return False
        
        # Check 3: k' != k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for kp in range(p):
                        if kp == k: continue
                        val = np.sum(U_mat[i, j, :] * V_mat[j, k, :] * W_mat[kp, i, :])
                        if not np.isclose(val, 0.0, atol=1e-9):
                            return False
        return True

    def residual(x):
        R = len(x) // (n*m + m*p + p*n)
        if R == 0: return []
        U = x[:n*m*R].reshape(n, m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
        W = x[n*m*R + m*p*R :].reshape(p, n, R)
        
        res = []
        # 1. Diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                    res.append(val - 1.0)
        
        # 2. Off-diagonal i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        if ip == i: continue
                        val = np.sum(U[i, j, :] * V[j, k, :] * W[k, ip, :])
                        res.append(val)
        
        # 3. Off-diagonal k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for kp in range(p):
                        if kp == k: continue
                        val = np.sum(U[i, j, :] * V[j, k, :] * W[kp, i, :])
                        res.append(val)
                        
        return np.array(res)

    best_solution = (U_school.tolist(), V_school.tolist(), W_school.tolist())
    
    # Try to find lower rank
    for R in range(95, 40, -1):
        if time.time() > deadline:
            break
            
        # Initialize with random values
        np.random.seed(42 + R)
        x0 = np.random.randn(n*m*R + m*p*R + p*n*R)
        
        try:
            sol = least_squares(residual, x0, method='lm', max_nfev=1000)
            if sol.success or sol.cost < 1e-8:
                x_opt = sol.x
                U_opt = x_opt[:n*m*R].reshape(n, m, R)
                V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
                W_opt = x_opt[n*m*R + m*p*R :].reshape(p, n, R)
                
                # Round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    best_solution = (U_round.flatten().tolist(), V_round.flatten().tolist(), W_round.flatten().tolist())
                    break # Found a lower rank, stop
        except Exception:
            pass

    U_final, V_final, W_final = best_solution
    return {"U": U_final, "V": V_final, "W": W_final}
