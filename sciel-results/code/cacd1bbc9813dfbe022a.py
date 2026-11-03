import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Try to find a better decomposition for 3x3x3
    # If found, use it twice for 3x3x6
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Search for 3x3x3 decomposition
    # Target: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k in 0..2
    # We try ranks from 20 to 26
    
    def find_3x3x3(rank):
        # Variables: U (9xR), V (9xR), W (9xR)
        # Flatten to vector
        size = 27 * rank
        x0 = np.random.uniform(-1, 1, size)
        
        # Initialize with schoolbook-like structure to help convergence?
        # Just random for now, with multiple restarts
        
        def error(x):
            U = x[:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:27*rank].reshape(9, rank)
            
            # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # i,j,k in 0..2
            err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*3 + j, c] * V[j*3 + k, c] * W[k*3 + i, c]
                        err += (s - 1.0)**2
            return err
        
        # Simple gradient descent
        # Or use scipy if available, but let's stick to simple to avoid dependency issues if scipy is slow
        # We'll use a few steps of gradient descent
        
        x = x0.copy()
        lr = 0.01
        for step in range(500):
            if time.time() > deadline:
                return None
            
            # Compute gradient numerically or analytically?
            # Analytical gradient is better.
            U = x[:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:27*rank].reshape(9, rank)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
                        diff = s - 1.0
                        for c in range(rank):
                            grad_U[i*3+j, c] += 2 * diff * V[j*3+k, c] * W[k*3+i, c]
                            grad_V[j*3+k, c] += 2 * diff * U[i*3+j, c] * W[k*3+i, c]
                            grad_W[k*3+i, c] += 2 * diff * U[i*3+j, c] * V[j*3+k, c]
            
            g = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            x -= lr * g
            
            # Check error
            e = error(x)
            if e < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for c in range(rank):
                                s += U_r[i*3+j, c] * V_r[j*3+k, c] * W_r[k*3+i, c]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return U_r, V_r, W_r
        
        return None

    # Try to find 3x3x3 decomposition
    for r in range(20, 27):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
            res = find_3x3x3(r)
            if res is not None:
                U1, V1, W1 = res
                # Construct 3x3x6
                total_rank = 2 * r
                U_new = np.zeros((9, total_rank))
                V_new = np.zeros((18, total_rank))
                W_new = np.zeros((18, total_rank))
                
                # Block 1: k=0,1,2
                U_new[:, 0:r] = U1
                V_new[0:9, 0:r] = V1
                # V[9:18, 0:r] = 0
                W_new[0:9, 0:r] = W1
                # W[9:18, 0:r] = 0
                
                # Block 2: k=3,4,5
                U_new[:, r:2*r] = U1
                # V[0:9, r:2*r] = 0
                V_new[9:18, r:2*r] = V1
                # W[0:9, r:2*r] = 0
                W_new[9:18, r:2*r] = W1
                
                # Verify final solution
                valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(6):
                            s = 0.0
                            for c in range(total_rank):
                                s += U_new[i*3+j, c] * V_new[j*6+k, c] * W_new[k*3+i, c]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_rank = total_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    found = True
                    break
        
        if found:
            break

    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
