import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def solve_block(n, m, p, target_rank, time_limit=10):
    """
    Attempts to find a rank target_rank decomposition for n x m x p matrix multiplication.
    Returns (U, V, W) if successful, else None.
    U shape (n*m, target_rank)
    V shape (m*p, target_rank)
    W shape (p*n, target_rank)
    """
    start_time = time.time()
    
    # Indices for the tensor T_{i,j,k} = 1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let's define the tensor elements
    # Actually, we can just loop over i, j, k
    
    def error(params):
        U = params[0:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :]) # No, it's element-wise product sum
                    # val = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    err += (val - 1.0)**2
        return err

    # Initial random guess
    # We can try to initialize with schoolbook and add noise, or just random
    # Random might be better to escape local minima of schoolbook
    
    best_rank = None
    best_params = None
    best_err = float('inf')
    
    max_restarts = 5
    for r in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random init
        np.random.seed(r)
        # Scale: values should be O(1)
        init = np.random.normal(0, 1, size=(n*m*target_rank + m*p*target_rank + p*n*target_rank))
        
        from scipy.optimize import minimize
        
        # Use L-BFGS-B
        res = minimize(error, init, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 1000})
        
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
            
        if best_err < 1e-8:
            break
            
    if best_err > 1e-6:
        return None
        
    # Extract U, V, W
    U = best_params[0:n*m*target_rank].reshape(n*m, target_rank)
    V = best_params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
    W = best_params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
    
    # Round to nearest 0.5
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return None
                    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    p1, p2 = 3, 4
    
    # Solve Block 1: 2 x 3 x 3
    try:
        U1, V1, W1 = solve_block(n, m, p1, 16, time_limit=20)
    except:
        U1, V1, W1 = None, None, None
        
    if U1 is None:
        # Schoolbook for block 1
        rank1 = n * m * p1
        U1 = np.zeros((n*m, rank1))
        V1 = np.zeros((m*p1, rank1))
        W1 = np.zeros((p1*n, rank1))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p1):
                    U1[i*m+j, col] = 1
                    V1[j*p1+k, col] = 1
                    W1[k*n+i, col] = 1
                    col += 1
    else:
        rank1 = 16

    # Solve Block 2: 2 x 3 x 4
    try:
        U2, V2, W2 = solve_block(n, m, p2, 22, time_limit=20)
    except:
        U2, V2, W2 = None, None, None
        
    if U2 is None:
        # Schoolbook for block 2
        rank2 = n * m * p2
        U2 = np.zeros((n*m, rank2))
        V2 = np.zeros((m*p2, rank2))
        W2 = np.zeros((p2*n, rank2))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    U2[i*m+j, col] = 1
                    V2[j*p2+k, col] = 1
                    W2[k*n+i, col] = 1
                    col += 1
    else:
        rank2 = 22

    # Combine
    U_total = np.hstack([U1, U2])
    
    # V needs shifting for indices
    V_total = np.zeros((m*p, rank1 + rank2))
    # Block 1 V uses k in [0, p1-1]
    for j in range(m):
        for k in range(p1):
            V_total[j*p+k, 0:rank1] = V1[j*p1+k, :]
            
    # Block 2 V uses k in [p1, p-1]
    for j in range(m):
        for k in range(p1, p):
            local_k = k - p1
            V_total[j*p+k, rank1:rank1+rank2] = V2[j*p2+local_k, :]
            
    # W needs shifting for indices
    W_total = np.zeros((p*n, rank1 + rank2))
    # Block 1 W uses k in [0, p1-1]
    for k in range(p1):
        for i in range(n):
            W_total[k*n+i, 0:rank1] = W1[k*n+i, :]
            
    # Block 2 W uses k in [p1, p-1]
    for k in range(p1, p):
        local_k = k - p1
        for i in range(n):
            W_total[k*n+i, rank1:rank1+rank2] = W2[local_k*n+i, :]
            
    # Convert to list of lists for return format?
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # The seed returns lists. I should return lists or arrays?
    # "return {'U': U, 'V': V, 'W': W}"
    # Seed returns lists of lists.
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
