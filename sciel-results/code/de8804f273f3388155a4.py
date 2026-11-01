"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Tensor-train inspired approach with randomized gradient descent
    # Target: T[i,j,k,l] = delta_{jk} for i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r] = delta_{jk}
    
    def compute_tensor(U, V, W):
        """Compute T[i,j,k,l] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r]"""
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
        return T
    
    def compute_gradients(U, V, W, T_current, T_target):
        """Compute gradients of loss = ||T_current - T_target||^2"""
        error = T_current - T_target
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        R = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        e = error[i,j,k,l]
                        for r in range(R):
                            dU[i*m+j, r] += 2 * e * V[j*p+k, r] * W[k*n+l, r]
                            dV[j*p+k, r] += 2 * e * U[i*m+j, r] * W[k*n+l, r]
                            dW[k*n+l, r] += 2 * e * U[i*m+j, r] * V[j*p+k, r]
        return dU, dV, dW
    
    def try_rank(R, seed=42):
        """Try to find a decomposition of rank R using randomized gradient descent."""
        np.random.seed(seed)
        
        # Tensor-train inspired initialization: structured random init
        U = np.random.randn(n * m, R) * 0.3
        V = np.random.randn(m * p, R) * 0.3
        W = np.random.randn(p * n, R) * 0.3
        
        # Target tensor
        T_target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_target[i,j,k,l] = 1.0 if j == k else 0.0
        
        # Adaptive learning rate gradient descent
        lr = 0.05
        min_lr = 1e-6
        lr_decay = 0.999
        patience = 100
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        no_improve = 0
        
        for iteration in range(20000):
            T_current = compute_tensor(U, V, W)
            loss = np.sum((T_current - T_target)**2)
            
            if loss < 1e-12:
                break
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience and lr > min_lr:
                lr *= 0.5
                no_improve = 0
            
            # Compute gradients
            dU, dV, dW = compute_gradients(U, V, W, T_current, T_target)
            
            # Gradient descent with momentum-like update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            lr *= lr_decay
        
        T_final = compute_tensor(best_U, best_V, best_W)
        max_err = np.max(np.abs(T_final - T_target))
        
        if max_err < 1e-8:
            return best_U, best_V, best_W, max_err
        return None, None, None, max_err
    
    # Try rank 10 first (as per tensor-train heuristic), then lower
    for R in [10, 9, 8]:
        # Try multiple seeds for robustness
        for seed in [42, 123, 456, 789, 101]:
            U, V, W, err = try_rank(R, seed)
            if U is not None:
                # Convert to exact integer form using rational approximation
                all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                # Try to find integer scaling
                best_scale = None
                best_max_err = float('inf')
                
                for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                    scaled_U = U * scale
                    scaled_V = V * scale
                    scaled_W = W * scale
                    
                    U_int = np.round(scaled_U)
                    V_int = np.round(scaled_V)
                    W_int = np.round(scaled_W)
                    
                    U_err = np.max(np.abs(U_int - scaled_U))
                    V_err = np.max(np.abs(V_int - scaled_V))
                    W_err = np.max(np.abs(W_int - scaled_W))
                    
                    max_err_scaled = max(U_err, V_err, W_err)
                    if max_err_scaled < 1e-6 and max_err_scaled < best_max_err:
                        best_max_err = max_err_scaled
                        best_scale = scale
                        best_U_int = U_int
                        best_V_int = V_int
                        best_W_int = W_int
                
                if best_scale is not None:
                    U_final = best_U_int.astype(float)
                    V_final = best_V_int.astype(float)
                    W_final = best_W_int.astype(float)
                    
                    # Verify exactness
                    T_check = compute_tensor(U_final, V_final, W_final)
                    T_target_check = np.zeros((n, m, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(m):
                                for l in range(p):
                                    T_target_check[i,j,k,l] = 1.0 if j == k else 0.0
                    
                    if np.max(np.abs(T_check - T_target_check)) < 1e-10:
                        U_list = [[int(x) for x in row] for row in U_final]
                        V_list = [[int(x) for x in row] for row in V_final]
                        W_list = [[int(x) for x in row] for row in W_final]
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook
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
# EVOLVE-BLOCK-END
