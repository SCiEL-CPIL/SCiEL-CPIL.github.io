import numpy as np
import time

def construct():
    n, m, p = 2, 5, 5
    
    # Target tensor T of shape (n*m, m*p, p*n) is not directly formed,
    # but we check the condition:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (val - 1.0) ** 2
        return err

    def verify(U, V, W, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Schoolbook decomposition
    r_sb = n * m * p
    U_sb = np.zeros((n*m, r_sb))
    V_sb = np.zeros((m*p, r_sb))
    W_sb = np.zeros((p*n, r_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
                
    best_rank = r_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    deadline = time.time() + 140
    
    # Try to find lower rank using gradient descent
    # We will try ranks from r_sb - 1 down to a lower bound
    # Lower bound for 2x5x5 is likely > 25
    
    for r in range(r_sb - 1, 20, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Simple gradient descent
            # Error E = sum_{i,j,k} (sum_r U_{ij}^r V_{jk}^r W_{ki}^r - 1)^2
            # This is expensive to compute gradient analytically for large r,
            # but dimensions are small.
            
            # Instead of full GD, we can try to solve for W given U, V?
            # Or just use a few steps of GD.
            
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                # Compute gradients numerically or analytically?
                # Analytical gradient is better.
                # Let S_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r
                # E = sum (S_{i,j,k} - 1)^2
                # dE/dU_{i,j}^r = 2 * sum_{k} (S_{i,j,k} - 1) V_{j,k}^r W_{k,i}^r
                # Similarly for V and W.
                
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                S = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for rr in range(r):
                                val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                            S[i,j,k] = val
                            
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = S[i,j,k] - 1.0
                            for rr in range(r):
                                U_grad[i*m+j, rr] += 2 * diff * V[j*p+k, rr] * W[k*n+i, rr]
                                V_grad[j*p+k, rr] += 2 * diff * U[i*m+j, rr] * W[k*n+i, rr]
                                W_grad[k*n+i, rr] += 2 * diff * U[i*m+j, rr] * V[j*p+k, rr]
                                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                if compute_error(U, V, W) < 1e-4:
                    if verify(U, V, W):
                        found = True
                        best_rank = r
                        best_U = U
                        best_V = V
                        best_W = W
                        break
            if found:
                break
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
