import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave margin for 150s limit
    
    # Target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def matricize(U, V, W, r):
        """Compute the tensor from factor matrices."""
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] = np.dot(U[i*m+j, :r], V[j*p+k, :r] * W[k*n+i, :r])
        return result
    
    def check_exact(U, V, W, r):
        """Check if the decomposition is exact (all entries = 1)."""
        T = matricize(U, V, W, r)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def round_half(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2
    
    # Start with schoolbook
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try to find lower rank decompositions
    for target_rank in range(rank0 - 1, 0, -1):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                T = matricize(U, V, W, target_rank)
                error = T - 1.0
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = error[i, j, k]
                            dU[i*m+j, :] += e * V[j*p+k, :] * W[k*n+i, :]
                            dV[j*p+k, :] += e * U[i*m+j, :] * W[k*n+i, :]
                            dW[k*n+i, :] += e * U[i*m+j, :] * V[j*p+k, :]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Round to half-integers
            U_r = round_half(U)
            V_r = round_half(V)
            W_r = round_half(W)
            
            # Check if rounded solution is exact
            if check_exact(U_r, V_r, W_r, target_rank):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if found:
            # Try to go even lower
            continue
        else:
            # Can't find this rank, stop
            break
    
    # Convert to lists
    U_list = best_U[:best_rank].tolist()
    V_list = best_V[:best_rank].tolist()
    W_list = best_W[:best_rank].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
