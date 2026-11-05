"""Matrix multiplication 2x3x5 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[i*m+j, j'*p+k, k'*n+i']
    # T = delta[j,j'] * delta[i,i'] * delta[k,k']
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition (rank 30)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try to find lower rank decomposition
    # We'll try ranks from 28 down to 24
    best_result = None
    best_rank = n * m * p  # 30
    
    def try_rank(r, max_iters=200, n_trials=5):
        """Try to find rank-r decomposition using iterative refinement."""
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Initialize randomly
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Use gradient descent / Newton-like method
            # Objective: minimize sum over (a,b,c) of (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
            # This is equivalent to minimizing ||T - U*V*W||_F^2 where * denotes the tensor product
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute current decomposition
                # D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Efficient computation:
                # D = np.einsum('ar,br,cr->abc', U, V, W)
                D = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Compute residual
                R = D - T  # R[a,b,c]
                
                # Check convergence
                err = np.linalg.norm(R)
                if err < 1e-10:
                    return U, V, W, err
                
                # Compute gradients
                # d/dU[a,r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
                # d/dV[b,r] = sum_{a,c} R[a,b,c] * U[a,r] * W[c,r]
                # d/dW[c,r] = sum_{a,b} R[a,b,c] * U[a,r] * V[b,r]
                
                # dU[a,r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
                # = einsum('abc,br,cr->ar', R, V, W)
                dU = np.einsum('abc,br,cr->ar', R, V, W)
                dV = np.einsum('abc,ar,cr->br', R, U, W)
                dW = np.einsum('abc,ar,br->cr', R, U, V)
                
                # Adaptive learning rate
                # Use a simple backtracking line search
                lr = 0.1
                for _ in range(10):
                    U_new = U - lr * dU
                    V_new = V - lr * dV
                    W_new = W - lr * dW
                    D_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    err_new = np.linalg.norm(D_new - T)
                    if err_new < err:
                        break
                    lr *= 0.5
                
                U, V, W = U_new, V_new, W_new
                err = err_new
            
            final_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
            if final_err < 1e-10:
                return U, V, W, final_err
        
        return None, None, None, np.inf
    
    # Try ranks from 28 down
    for r in range(28, 23, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=100, n_trials=3)
        if result[0] is not None:
            U, V, W, err = result
            if err < 1e-10:
                best_result = (U, V, W)
                best_rank = r
                break
    
    # If we found a better rank, use it; otherwise fall back to schoolbook
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = schoolbook()
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
