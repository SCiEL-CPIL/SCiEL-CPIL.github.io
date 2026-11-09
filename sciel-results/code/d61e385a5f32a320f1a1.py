import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition (rank = n*m*p = 54)
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
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_schoolbook
    
    # Efficient verification using NumPy
    def is_exact(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Reshape to 3D
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # For each (i,j,k), compute sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is a 3-index contraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Strategy: for each candidate rank, try structured initializations and local search
    
    def try_rank(r, num_restarts=3):
        """Try to find a rank-r decomposition."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return False
            
            # Different initialization strategies
            if restart == 0:
                # Random initialization
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            elif restart == 1:
                # Structured: try to exploit the all-ones tensor
                # Initialize with small values that might sum to 1
                U = np.ones((n * m, r)) / np.sqrt(r)
                V = np.ones((m * p, r)) / np.sqrt(r)
                W = np.ones((p * n, r)) / np.sqrt(r)
            else:
                # Sparse initialization
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for c in range(r):
                    # Each column has a few non-zero entries
                    idx_u = np.random.choice(n * m, size=min(3, n * m), replace=False)
                    idx_v = np.random.choice(m * p, size=min(3, m * p), replace=False)
                    idx_w = np.random.choice(p * n, size=min(3, p * n), replace=False)
                    U[idx_u, c] = 1.0
                    V[idx_v, c] = 1.0
                    W[idx_w, c] = 1.0
            
            # Local search with gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    return False
                
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                
                # Compute error
                error = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            error[i, j, k] = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :]) - 1.0
                
                max_err = np.max(np.abs(error))
                if max_err < 1e-9:
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if is_exact(U_r, V_r, W_r):
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        best_rank = r
                        return True
                    else:
                        # Continue with unrounded
                        pass
                
                # Compute gradients
                grad_U = np.zeros((n, m, r))
                grad_V = np.zeros((m, p, r))
                grad_W = np.zeros((p, n, r))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = error[i, j, k]
                            if abs(e) > 1e-12:
                                grad_U[i, j, :] += 2 * e * V3[j, k, :] * W3[k, i, :]
                                grad_V[j, k, :] += 2 * e * U3[i, j, :] * W3[k, i, :]
                                grad_W[k, i, :] += 2 * e * U3[i, j, :] * V3[j, k, :]
                
                # Update with learning rate decay
                U -= lr * grad_U.reshape(n * m, r)
                V -= lr * grad_V.reshape(m * p, r)
                W -= lr * grad_W.reshape(p * n, r)
                
                # Decay learning rate
                lr *= 0.99
            
            # Final check after local search
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if is_exact(U_r, V_r, W_r):
                best_U = U_r
                best_V = V_r
                best_W = W_r
                best_rank = r
                return True
        
        return False
    
    # Try ranks from 53 down to 40
    for r in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline - 10:
            break
        if try_rank(r, num_restarts=2):
            # Found a valid decomposition at this rank, try lower
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
