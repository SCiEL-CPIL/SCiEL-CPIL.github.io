import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = [[0.0] * rank_sb for _ in range(n * m)]
    V_sb = [[0.0] * rank_sb for _ in range(m * p)]
    W_sb = [[0.0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    def verify(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[idx_u][r] * V[idx_v][r] * W[idx_w][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_np(U, V, W, tol=1e-8):
        """Verify using numpy for speed"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_to_half_int(arr, tol=1e-4):
        """Try to round to nearest half-integer and check validity"""
        rounded = np.round(arr * 2) / 2.0
        if verify_np(rounded, V_r, W_r, tol=1e-6):
            return rounded
        return None
    
    def try_find_solution(rank, max_steps=5000, lr_init=0.01):
        """Try to find a rank-r solution using Adam optimization"""
        if time.time() > deadline:
            return None
        
        # Multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                return None
            
            np.random.seed(attempt * 12345 + rank)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Adam optimizer state
            mU = np.zeros_like(U)
            vU = np.zeros_like(U)
            mV = np.zeros_like(V)
            vV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vW = np.zeros_like(W)
            t = 0
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            lr = lr_init
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                t += 1
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                err = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            
                            prod = U[idx_u] * V[idx_v] * W[idx_w]
                            s = np.sum(prod)
                            diff = s - 1.0
                            err += diff * diff
                            
                            grad_U[idx_u] += 2 * diff * V[idx_v] * W[idx_w]
                            grad_V[idx_v] += 2 * diff * U[idx_u] * W[idx_w]
                            grad_W[idx_w] += 2 * diff * U[idx_u] * V[idx_v]
                
                # Adam update
                mU = beta1 * mU + (1 - beta1) * grad_U
                vU = beta2 * vU + (1 - beta2) * grad_U ** 2
                mU_hat = mU / (1 - beta1 ** t)
                vU_hat = vU / (1 - beta2 ** t)
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                
                mV = beta1 * mV + (1 - beta1) * grad_V
                vV = beta2 * vV + (1 - beta2) * grad_V ** 2
                mV_hat = mV / (1 - beta1 ** t)
                vV_hat = vV / (1 - beta2 ** t)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                
                mW = beta1 * mW + (1 - beta1) * grad_W
                vW = beta2 * vW + (1 - beta2) * grad_W ** 2
                mW_hat = mW / (1 - beta1 ** t)
                vW_hat = vW / (1 - beta2 ** t)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                # Decay learning rate
                lr = lr_init * (0.999 ** (step // 100))
                
                if err < 1e-10:
                    break
            
            if time.time() > deadline:
                return None
            
            # Check if we found a good solution
            if err < 1e-6:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                
                if verify_np(U_r, V_r, W_r, tol=1e-6):
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist(), rank)
                
                # Try with smaller tolerance for rounding
                for tol in [1e-5, 1e-4, 1e-3]:
                    U_r2 = np.round(U / tol) * tol
                    V_r2 = np.round(V / tol) * tol
                    W_r2 = np.round(W / tol) * tol
                    if verify_np(U_r2, V_r2, W_r2, tol=1e-3):
                        # Check if entries are close to half-integers
                        U_close = np.all(np.abs(U_r2 * 2 - np.round(U_r2 * 2)) < 0.01)
                        V_close = np.all(np.abs(V_r2 * 2 - np.round(V_r2 * 2)) < 0.01)
                        W_close = np.all(np.abs(W_r2 * 2 - np.round(W_r2 * 2)) < 0.01)
                        if U_close and V_close and W_close:
                            U_final = np.round(U_r2 * 2) / 2.0
                            V_final = np.round(V_r2 * 2) / 2.0
                            W_final = np.round(W_r2 * 2) / 2.0
                            if verify_np(U_final, V_final, W_final, tol=1e-4):
                                return (U_final.tolist(), V_final.tolist(), W_final.tolist(), rank)
        
        return None
    
    # Try ranks from low to high
    for rank in [18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline:
            break
        
        result = try_find_solution(rank)
        if result is not None:
            U_list, V_list, W_list, r = result
            best_U = U_list
            best_V = V_list
            best_W = W_list
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
