import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # 1. Try to find a low-rank decomposition for 3x3x3
    # Target: sum_r U[r,i,j] V[r,j',k] W[r,i,k] = delta[j,j']
    
    n, m, p = 3, 3, 3
    R_target = 23
    
    def solve_333(R):
        # Variables: U (R, 9), V (R, 9), W (R, 9)
        # Flatten to vector x of size 27*R
        # Residual: 81 values
        
        # Initialize
        best_error = float('inf')
        best_x = None
        
        for seed in range(10):
            if time.time() > deadline - 10: break
            random.seed(seed)
            np.random.seed(seed)
            
            # Random init
            U = np.random.randn(R, 9)
            V = np.random.randn(R, 9)
            W = np.random.randn(R, 9)
            
            # Scale to avoid large numbers
            U /= np.linalg.norm(U) * np.sqrt(R)
            V /= np.linalg.norm(V) * np.sqrt(R)
            W /= np.linalg.norm(W) * np.sqrt(R)
            
            x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use scipy.optimize
            from scipy.optimize import minimize
            
            def objective(x):
                U = x[:R*9].reshape(R, 9)
                V = x[R*9:2*R*9].reshape(R, 9)
                W = x[2*R*9:].reshape(R, 9)
                
                # Compute residual
                # res[i, j, j', k] = sum_r U[r,i,j] V[r,j',k] W[r,i,k] - delta[j,j']
                # We can vectorize this.
                # U: (R, 3, 3)
                # V: (R, 3, 3)
                # W: (R, 3, 3)
                
                U3 = U.reshape(R, 3, 3)
                V3 = V.reshape(R, 3, 3)
                W3 = W.reshape(R, 3, 3)
                
                # Sum over r: U[r,i,j] V[r,j',k] W[r,i,k]
                # This is a 4-tensor T[i,j,j',k]
                # T = np.einsum('rij,rjk,rik->ijjk', U3, V3, W3)
                
                T = np.einsum('rij,rjk,rik->ijjk', U3, V3, W3)
                
                # Target: delta[j,j'] for all i,k
                # Target tensor D[i,j,j',k] = delta[j,j']
                D = np.zeros((3, 3, 3, 3))
                for i in range(3):
                    for k in range(3):
                        D[i, :, :, k] = np.eye(3)
                        
                error = np.sum((T - D) ** 2)
                return error
            
            res = minimize(objective, x, method='BFGS', options={'maxiter': 100})
            
            if res.fun < best_error:
                best_error = res.fun
                best_x = res.x
                
            if best_error < 1e-10:
                break
                
        if best_x is None or best_error > 1e-6:
            return None
            
        U = best_x[:R*9].reshape(R, 9)
        V = best_x[R*9:2*R*9].reshape(R, 9)
        W = best_x[2*R*9:].reshape(R, 9)
        
        # Try to snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify snapped solution
        U3 = U_snap.reshape(R, 3, 3)
        V3 = V_snap.reshape(R, 3, 3)
        W3 = W_snap.reshape(R, 3, 3)
        T = np.einsum('rij,rjk,rik->ijjk', U3, V3, W3)
        D = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for k in range(3):
                D[i, :, :, k] = np.eye(3)
                
        if np.allclose(T, D, atol=1e-9):
            return U_snap, V_snap, W_snap
            
        return None

    U333, V333, W333 = solve_333(23)
    
    if U333 is None:
        # Fallback to schoolbook 63
        n, m, p = DIMENSIONS
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
        
    # Construct 3x3x7 decomposition
    # Use two copies of 333 and one 331
    R333 = 23
    R331 = 9
    TotalRank = R333 + R333 + R331
    
    n, m, p = 3, 3, 7
    U = [[0] * TotalRank for _ in range(n * m)]
    V = [[0] * TotalRank for _ in range(m * p)]
    W = [[0] * TotalRank for _ in range(p * n)]
    
    # Fill first 23 columns for B cols 0-2
    for r in range(R333):
        for i in range(3):
            for j in range(3):
                U[i * 3 + j][r] = U333[r, i * 3 + j]
                V[j * 7 + (i if False else 0)][r] = 0 # Placeholder
        # V indexing: V[j*p + k, r]
        # For the first 3 columns of B (k=0,1,2), we use V333
        # V333 is (23, 9) where index is j*3 + k
        for j in range(3):
            for k in range(3):
                V[j * 7 + k][r] = V333[r, j * 3 + k]
        # W indexing: W[k*n + i, r]
        # W333 is (23, 9) where index is k*3 + i
        for k in range(3):
            for i in range(3):
                W[k * 3 + i][r] = W333[r, k * 3 + i]
                
    # Fill next 23 columns for B cols 3-5
    offset = R333
    for r in range(R333):
        for i in range(3):
            for j in range(3):
                U[i * 3 + j][r + offset] = U333[r, i * 3 + j]
        for j in range(3):
            for k in range(3):
                # k in 3,4,5 corresponds to k' in 0,1,2
                V[j * 7 + (k + 3)][r + offset] = V333[r, j * 3 + k]
        for k in range(3):
            for i in range(3):
                # k in 3,4,5 corresponds to k' in 0,1,2
                W[(k + 3) * 3 + i][r + offset] = W333[r, k * 3 + i]
                
    # Fill last 9 columns for B col 6
    offset = 2 * R333
    for i in range(3):
        for j in range(3):
            r = offset + i * 3 + j
            U[i * 3 + j][r] = 1
            V[j * 7 + 6][r] = 1
            W[6 * 3 + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
