import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Build the target tensor indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
    
    def compute_error(U, V, W, rank):
        error = 0.0
        for a, b, c in indices:
            val = np.sum(U[a] * V[b] * W[c])
            error += (val - 1.0)**2
        return error
    
    def gradient_descent(U, V, W, rank, max_iter=500, lr_init=0.05, tol=1e-8):
        """Adaptive gradient descent with learning rate decay"""
        lr = lr_init
        best_err = float('inf')
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for a, b, c in indices:
                residual = np.sum(U[a] * V[b] * W[c]) - 1.0
                grad_U[a] += 2 * residual * V[b] * W[c]
                grad_V[b] += 2 * residual * U[a] * W[c]
                grad_W[c] += 2 * residual * U[a] * V[b]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check error periodically
            if iteration % 10 == 0:
                err = compute_error(U, V, W, rank)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                # Adaptive learning rate
                if err < best_err * 0.99:
                    lr *= 1.05
                else:
                    lr *= 0.9
                
                if err < tol:
                    break
            
            if time.time() > deadline:
                break
        
        return best_U, best_V, best_W, best_err
    
    best_rank = n*m*p
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find a lower rank decomposition
    for rank in range(80, 97):
        if time.time() > deadline:
            break
        
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Gradient descent with adaptive learning rate
            U, V, W, err = gradient_descent(U, V, W, rank, max_iter=800, lr_init=0.03, tol=1e-10)
            
            if err < 1e-6:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                err_rounded = compute_error(U_rounded, V_rounded, W_rounded, rank)
                if err_rounded < 1e-6:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
        
        if best_U is not None:
            break
    
    if best_U is None:
        # Schoolbook
        best_rank = n*m*p
        best_U = np.zeros((n*m, best_rank))
        best_V = np.zeros((m*p, best_rank))
        best_W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
