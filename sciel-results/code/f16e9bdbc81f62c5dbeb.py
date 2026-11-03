"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Theoretical lower bound is max(n*m, m*p, n*p) = max(6,12,8) = 12
    min_rank = max(n * m, m * p, n * p)
    
    # Precompute the tensor entries we need to satisfy
    # T[(i,j), (jp,k), (ip,kp)] = 1 iff i==ip and j==jp and k==kp
    # We need: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    
    # For each rank, try to find a decomposition
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 20:
            break
        
        success = False
        for attempt in range(15):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with structured bias
            np.random.seed(attempt * 1000 + rank)
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Also try a structured initialization based on schoolbook subset
            if attempt % 3 == 0:
                # Start from schoolbook and randomly zero out some columns
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < rank:
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
                            else:
                                break
                        if col >= rank:
                            break
                    if col >= rank:
                        break
            
            def residual(params):
                U_flat, V_flat, W_flat = params[:n*m*rank], params[n*m*rank:n*m*rank+m*p*rank], params[n*m*rank+m*p*rank:]
                U_mat = U_flat.reshape(n*m, rank)
                V_mat = V_flat.reshape(m*p, rank)
                W_mat = W_flat.reshape(p*n, rank)
                
                res = []
                for i in range(n):
                    for ip in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    for kp in range(p):
                                        target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                        val = 0.0
                                        for r in range(rank):
                                            val += W_mat[kp*n+ip, r] * U_mat[i*m+j, r] * V_mat[jp*p+k, r]
                                        res.append(val - target)
                return np.array(res)
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead for small problems
            try:
                # First try a quick gradient-based approach
                def objective(x):
                    r = residual(x)
                    return np.sum(r**2)
                
                result = minimize(objective, params0, 
                                  method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                
                max_err = max(abs(r) for r in residual(result.x)) if len(result.x) > 0 else float('inf')
                
                if max_err < 1e-8:
                    U_mat = result.x[:n*m*rank].reshape(n*m, rank)
                    V_mat = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_mat = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_mat * 2) / 2
                    V_snap = np.round(V_mat * 2) / 2
                    W_snap = np.round(W_mat * 2) / 2
                    
                    # Verify the snapped version
                    def verify_snapped(U_s, V_s, W_s):
                        for i in range(n):
                            for ip in range(n):
                                for j in range(m):
                                    for jp in range(m):
                                        for k in range(p):
                                            for kp in range(p):
                                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                                val = 0.0
                                                for r in range(rank):
                                                    val += W_s[kp*n+ip, r] * U_s[i*m+j, r] * V_s[jp*p+k, r]
                                                if abs(val - target) > 1e-6:
                                                    return False
                        return True
                    
                    if verify_snapped(U_snap, V_snap, W_snap):
                        U_list = [[val for val in row] for row in U_snap]
                        V_list = [[val for val in row] for row in V_snap]
                        W_list = [[val for val in row] for row in W_snap]
                        
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        success = True
                        break
            except Exception:
                continue
        
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
