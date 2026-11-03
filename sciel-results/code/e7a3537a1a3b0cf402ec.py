import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build the full tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try rank 43: 7 for 3x3x3 + 36 for 3x3x4
    # First, try to find a 7-term decomposition for the 3x3x3 part
    T1 = np.zeros((n*m, m*3, 3*n))  # 3x3x3 part
    for i in range(n):
        for j in range(m):
            for k in range(3):
                T1[i*m+j, j*3+k, k*n+i] = 1
    
    # Try to find a 7-term decomposition
    rank1 = 7
    best_error = float('inf')
    best_U1 = None
    best_V1 = None
    best_W1 = None
    
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        U1 = np.random.randn(n*m, rank1)
        V1 = np.random.randn(m*3, rank1)
        W1 = np.random.randn(3*n, rank1)
        
        def residual(params):
            U = params[:n*m*rank1].reshape(n*m, rank1)
            V = params[n*m*rank1:n*m*rank1+m*3*rank1].reshape(m*3, rank1)
            W = params[n*m*rank1+m*3*rank1:].reshape(3*n, rank1)
            result = np.einsum('ar,br,cr->abc', U, V, W)
            return (result - T1).flatten()
        
        params0 = np.concatenate([U1.flatten(), V1.flatten(), W1.flatten()])
        try:
            result = least_squares(residual, params0, method='lm', max_nfev=1000)
            U1 = result.x[:n*m*rank1].reshape(n*m, rank1)
            V1 = result.x[n*m*rank1:n*m*rank1+m*3*rank1].reshape(m*3, rank1)
            W1 = result.x[n*m*rank1+m*3*rank1:].reshape(3*n, rank1)
            result_tensor = np.einsum('ar,br,cr->abc', U1, V1, W1)
            error = np.max(np.abs(result_tensor - T1))
            if error < best_error:
                best_error = error
                best_U1 = U1.copy()
                best_V1 = V1.copy()
                best_W1 = W1.copy()
        except:
            pass
    
    if best_error < 1e-6:
        # Use the 7-term decomposition for the 3x3x3 part
        rank = rank1 + n*m*4  # 7 + 36 = 43
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Copy the 7-term decomposition
        U[:, :rank1] = best_U1
        V[:, :rank1] = best_V1
        W[:, :rank1] = best_W1
        
        # Schoolbook for the 3x3x4 part
        col = rank1
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        # Verify
        T_result = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_result, T, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
