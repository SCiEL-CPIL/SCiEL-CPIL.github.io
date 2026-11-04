import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave margin for safety
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def solve_for_W(U, V, r):
        W = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = W_col
                except:
                    W[k*n+i, :] = 0
        return W
    
    def solve_for_U(U, V, W, r):
        U_new = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = U_col
                except:
                    U_new[i*m+j, :] = 0
        return U_new
    
    def solve_for_V(U, V, W, r):
        V_new = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = V_col
                except:
                    V_new[j*p+k, :] = 0
        return V_new
    
    def compute_loss(U, V, W):
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    total += (val - 1.0) ** 2
        return total
    
    def compute_gradients(U, V, W, r):
        """Compute gradients for U, V, W"""
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err = val - 1.0
                    # dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                    # dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                    # dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
                    dU[i*m+j] += 2 * err * V[j*p+k] * W[k*n+i]
                    dV[j*p+k] += 2 * err * U[i*m+j] * W[k*n+i]
                    dW[k*n+i] += 2 * err * U[i*m+j] * V[j*p+k]
        
        return dU, dV, dW
    
    def gradient_polish(U, V, W, r, lr=0.01, max_iter=200):
        """Polish solution using gradient descent"""
        best_loss = compute_loss(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > timeout - 15:
                break
            
            dU, dV, dW = compute_gradients(U, V, W, r)
            
            U = U - lr * dU
            V = V - lr * dV
            W = W - lr * dW
            
            loss = compute_loss(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
        
        return best_U, best_V, best_W, best_loss
    
    def try_rank(r, max_sweeps=20, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        best_overall = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Multiple random restarts with different initialization strategies
        for attempt in range(5):
            if time.time() - start_time > timeout - 20:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Standard random
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
            elif attempt == 1:
                # Sparse initialization
                U = np.random.randn(n * m, r) * 0.1
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
            elif attempt == 2:
                # Block structure hint
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i * m + j < r:
                                U[i * m + j, i * m + j] = 1.0
                            if j * p + k < r:
                                V[j * p + k, j * p + k] = 1.0
                            if k * n + i < r:
                                W[k * n + i, k * n + i] = 1.0
                U += np.random.randn(n * m, r) * 0.01
                V += np.random.randn(m * p, r) * 0.01
                W += np.random.randn(p * n, r) * 0.01
            elif attempt == 3:
                # Low-rank structure
                U = np.random.randn(n * m, min(r, 10))
                V = np.random.randn(m * p, min(r, 10))
                W = np.random.randn(p * n, min(r, 10))
                # Pad with zeros
                if r > 10:
                    U = np.hstack([U, np.zeros((n * m, r - 10))])
                    V = np.hstack([V, np.zeros((m * p, r - 10))])
                    W = np.hstack([W, np.zeros((p * n, r - 10))])
            else:
                # Mixed scale
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 2.0
            
            best_loss = float('inf')
            best_U_local, best_V_local, best_W_local = U.copy(), V.copy(), W.copy()
            
            for sweep in range(max_sweeps):
                if time.time() - start_time > timeout - 20:
                    break
                
                W = solve_for_W(U, V, r)
                U = solve_for_U(U, V, W, r)
                V = solve_for_V(U, V, W, r)
                
                loss = compute_loss(U, V, W)
                if loss < best_loss:
                    best_loss = loss
                    best_U_local, best_V_local, best_W_local = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-12:
                    break
            
            # Gradient polish
            if best_loss < 1e-6:
                U_pol, V_pol, W_pol, loss_pol = gradient_polish(
                    best_U_local, best_V_local, best_W_local, r, 
                    lr=0.005, max_iter=100
                )
                if loss_pol < best_loss:
                    best_loss = loss_pol
                    best_U_local, best_V_local, best_W_local = U_pol, V_pol, W_pol
            
            if best_loss < best_overall:
                best_overall = best_loss
                best_U, best_V, best_W = best_U_local, best_V_local, best_W_local
            
            if best_overall < 1e-10:
                # Try to round to half-integers
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    return {
                        "U": [[float(x) for x in row] for row in U_r],
                        "V": [[float(x) for x in row] for row in V_r],
                        "W": [[float(x) for x in row] for row in W_r]
                    }
        
        return None
    
    # Try ranks from 71 down to 40
    for r in range(71, 39, -1):
        if time.time() - start_time > timeout - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            return result
    
    return schoolbook()
