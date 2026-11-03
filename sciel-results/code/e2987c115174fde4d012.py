import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # We want to find U, V, W such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound for M_4,5,5 is not known exactly, but likely > 20
    # Start from 30 to be safe, go up to 100
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    for R in range(30, 101):
        if time.time() > deadline:
            break
            
        # Multiple random restarts for each rank
        for restart in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 100 + restart)
            
            # Initialize with small random values
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute error
                # E[i, j, k] = 1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                
                # Efficient computation using einsum or loops
                # U shape (n*m, R), V (m*p, R), W (p*n, R)
                
                # Sum over r: U_ik V_jk W_ki ... wait indices
                # Let's compute the reconstructed tensor C_recon
                # C_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # We can do this by iterating over r or using broadcasting
                # U_reshaped: (n, m, R)
                # V_reshaped: (m, p, R)
                # W_reshaped: (p, n, R)
                
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                
                # Product term for each r: U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:]
                # This is tricky to broadcast directly without expanding R dimension in output.
                # We can sum over R using einsum
                # C_recon[i, j, k] = sum_r U_r[i, j, r] V_r[j, k, r] W_r[k, i, r]
                C_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                Error = C_recon - T
                Loss = np.sum(Error**2)
                
                # Gradients
                # dL/dU[i*m+j, r] = 2 * sum_{j,k} Error[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                # dL/dV[j*p+k, r] = 2 * sum_{i,k} Error[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                # dL/dW[k*n+i, r] = 2 * sum_{i,j} Error[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                
                # Compute gradients using einsum
                # grad_U[i, j, r] = sum_{j,k} Error[i, j, k] V[j, k, r] W[k, i, r]
                # Note: V is indexed by j*p+k, so V_r[j, k, r]
                # W is indexed by k*n+i, so W_r[k, i, r]
                
                grad_U_r = np.einsum('ijk,jkr,kir->ijr', Error, V_r, W_r)
                grad_V_r = np.einsum('ijk,ijr,kir->jkr', Error, U_r, W_r)
                grad_W_r = np.einsum('ijk,ijr,jkr->kir', Error, U_r, V_r)
                
                grad_U = grad_U_r.reshape(-1, R)
                grad_V = grad_V_r.reshape(-1, R)
                grad_W = grad_W_r.reshape(-1, R)
                
                # Update
                U -= lr * 2 * grad_U
                V -= lr * 2 * grad_V
                W -= lr * 2 * grad_W
                
                # Optional: check convergence
                if Loss < 1e-8:
                    break
            
            # Check if solution is good enough
            if Loss < 1e-6:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check exactness
                U_r = U_rounded.reshape(n, m, R)
                V_r = V_rounded.reshape(m, p, R)
                W_r = W_rounded.reshape(p, n, R)
                C_check = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                if np.allclose(C_check, T, atol=1e-9):
                    # Convert to list of lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
    
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
