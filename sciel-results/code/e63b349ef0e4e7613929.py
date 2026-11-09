"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def construct():
    global START_TIME
    if 'START_TIME' not in globals() or START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 48
    
    # Build the target tensor T[i,j,k] = 1 for valid multiplication
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # W[k*n + i, c]
    
    # Try to find lower rank decompositions
    # Start with schoolbook and try to reduce
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Initialize schoolbook
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = schoolbook_rank
    
    # Try to find lower rank by optimization
    # For each candidate rank, try random init + gradient descent
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the tensor."""
        r = U.shape[1]
        # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For valid (i,j,k): should be 1
        # For invalid: but actually all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid
        # The tensor is 1 for all such triples
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j,:], V[j*p+k,:]*W[k*n+i,:])
                    # Actually: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                    # Let me compute properly
                    prod = U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]
                    val = np.sum(prod)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank(rank, num_trials=5, max_iter=2000):
        """Try to find a rank-r decomposition."""
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        for trial in range(num_trials):
            if time.time() - START_TIME > TIME_LIMIT:
                return None
            
            # Use Adam-like optimization
            lr = 0.01
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iter):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                
                # Compute loss: sum over all (i,j,k) of (T[i,j,k] - target)^2
                # target = 1 for all valid (i,j,k)
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                            prod = U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]
                            val = np.sum(prod)
                            err = val - 1.0
                            loss += err*err
                            
                            # d(loss)/d(U[i,j,c]) = 2*err * V[j,k,c] * W[k,i,c]
                            dU[i*m+j,:] += 2.0 * err * V[j*p+k,:] * W[k*n+i,:]
                            dV[j*p+k,:] += 2.0 * err * U[i*m+j,:] * W[k*n+i,:]
                            dW[k*n+i,:] += 2.0 * err * U[i*m+j,:] * V[j*p+k,:]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_loss < 1e-12:
                    break
            
            if best_loss < 1e-10:
                # Try to round to half-integers
                U_round = np.round(best_U * 2) / 2
                V_round = np.round(best_V * 2) / 2
                W_round = np.round(best_W * 2) / 2
                err = check_decomposition(U_round, V_round, W_round)
                if err < 1e-9:
                    return U_round, V_round, W_round
                else:
                    # Use unrounded if close enough
                    err2 = check_decomposition(best_U, best_V, best_W)
                    if err2 < 1e-9:
                        return best_U, best_V, best_W
        
        return None
    
    # Try ranks from 47 down to some minimum
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Also rank >= max(n, m, p) * something...
    # Let's try from 47 down
    
    for rank in range(schoolbook_rank - 1, 23, -1):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        result = try_rank(rank, num_trials=3, max_iter=1000)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            # Found a valid decomposition, try even lower
            continue
    
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
