import scipy
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Try ranks from 14 to 18
    for rank in range(14, 19):
        if time.time() > deadline - 10:
            break
            
        success = try_find_decomposition(n, m, p, rank, deadline)
        if success:
            return success
            
    # Fallback to schoolbook
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

def try_find_decomposition(n, m, p, rank, deadline):
    # Shapes
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # We need to find U, V, W such that sum_r U[i*m+k, r] * V[k*p+l, r] * W[l*n+i, r] == 1 for all i,k,l
    # This is a system of n*m*p equations.
    
    # Random restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            return None
            
        # Initialize with random values
        # Scale matters
        U = np.random.randn(size_U, rank) * 0.5
        V = np.random.randn(size_V, rank) * 0.5
        W = np.random.randn(size_W, rank) * 0.5
        
        # Flatten for optimization
        # Actually, we can just use numpy operations in the objective function.
        
        def objective(params):
            U_flat = params[:size_U * rank]
            V_flat = params[size_U * rank : size_U * rank + size_V * rank]
            W_flat = params[size_U * rank + size_V * rank :]
            
            U_mat = U_flat.reshape(size_U, rank)
            V_mat = V_flat.reshape(size_V, rank)
            W_mat = W_flat.reshape(size_W, rank)
            
            error = 0.0
            # Calculate sum for each i, k, l
            # C[i, l] = sum_k A[i, k] B[k, l]
            # Our tensor equation: sum_r U[i*m+k, r] V[k*p+l, r] W[l*n+i, r] = 1
            for i in range(n):
                for k in range(m):
                    for l in range(p):
                        u_idx = i * m + k
                        v_idx = k * p + l
                        w_idx = l * n + i
                        val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                        error += (val - 1.0) ** 2
            return error
            
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use Nelder-Mead or BFGS
        # BFGS is faster for smooth problems
        try:
            res = scipy.optimize.minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000})
            if res.fun < 1e-8:
                # Check if rounding to half-integers works
                U_opt = res.x[:size_U * rank].reshape(size_U, rank)
                V_opt = res.x[size_U * rank : size_U * rank + size_V * rank].reshape(size_V, rank)
                W_opt = res.x[size_U * rank + size_V * rank :].reshape(size_W, rank)
                
                # Round to nearest 0.5
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for k in range(m):
                        for l in range(p):
                            u_idx = i * m + k
                            v_idx = k * p + l
                            w_idx = l * n + i
                            val = np.dot(U_r[u_idx, :], V_r[v_idx, :] * W_r[w_idx, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception:
            pass
            
    return None
