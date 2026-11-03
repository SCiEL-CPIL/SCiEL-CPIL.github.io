import time
import numpy as np
import random
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]
START_TIME = time.time()
DEADLINE = START_TIME + 140

def construct():
    n, m, p = DIMENSIONS
    U_rows = n * m  # 9
    V_rows = m * p  # 12
    W_rows = p * n  # 12
    
    def schoolbook():
        U = np.zeros((U_rows, n*m*p))
        V = np.zeros((V_rows, n*m*p))
        W = np.zeros((W_rows, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        return err

    def objective(params):
        r = len(params) // (U_rows + V_rows + W_rows)
        U_flat = params[:U_rows*r].reshape(U_rows, r)
        V_flat = params[U_rows*r : U_rows*r + V_rows*r].reshape(V_rows, r)
        W_flat = params[U_rows*r + V_rows*r :].reshape(W_rows, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                    err += (s - 1.0) ** 2
        return err

    def gradient(params):
        r = len(params) // (U_rows + V_rows + W_rows)
        U_flat = params[:U_rows*r].reshape(U_rows, r)
        V_flat = params[U_rows*r : U_rows*r + V_rows*r].reshape(V_rows, r)
        W_flat = params[U_rows*r + V_rows*r :].reshape(W_rows, r)
        
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                    diff = 2 * (s - 1.0)
                    
                    grad_U[i*m+j, :] += diff * (V_flat[j*p+k, :] * W_flat[k*n+i, :])
                    grad_V[j*p+k, :] += diff * (U_flat[i*m+j, :] * W_flat[k*n+i, :])
                    grad_W[k*n+i, :] += diff * (U_flat[i*m+j, :] * V_flat[j*p+k, :])
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def alternating_solve(U, V, W, max_iter=50, tol=1e-10):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = len(U[0])
        
        prev_err = compute_error(U, V, W)
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, r))
                    b_vec = np.ones(p)
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    b_vec = np.ones(n)
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    b_vec = np.ones(m)
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            cur_err = compute_error(U, V, W)
            if abs(prev_err - cur_err) < tol:
                break
            prev_err = cur_err
        
        return U, V, W, cur_err

    def verify(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half(U, V, W):
        U = np.round(2 * U) / 2
        V = np.round(2 * V) / 2
        W = np.round(2 * W) / 2
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    ranks_to_try = list(range(25, 14, -1))
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts and time.time() < DEADLINE - 10:
            attempts += 1
            np.random.seed(attempts * 7919 + target_rank * 1000)
            
            U = np.random.randn(U_rows, target_rank) * 0.5
            V = np.random.randn(V_rows, target_rank) * 0.5
            W = np.random.randn(W_rows, target_rank) * 0.5
            
            U_opt, V_opt, W_opt, err = alternating_solve(U, V, W, max_iter=20)
            
            if err < 1e-3:
                params = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.success or res.fun < 1e-8:
                    U_opt = res.x[:U_rows*target_rank].reshape(U_rows, target_rank)
                    V_opt = res.x[U_rows*target_rank : U_rows*target_rank + V_rows*target_rank].reshape(V_rows, target_rank)
                    W_opt = res.x[U_rows*target_rank + V_rows*target_rank :].reshape(W_rows, target_rank)
                    
                    U_opt, V_opt, W_opt, err = alternating_solve(U_opt, V_opt, W_opt, max_iter=50)
                    
                    if err < 1e-6:
                        U_snap, V_snap, W_snap = snap_to_half(U_opt, V_opt, W_opt)
                        if verify(U_snap, V_snap, W_snap, tol=1e-6):
                            best_U, best_V, best_W = U_snap, V_snap, W_snap
                            best_rank = target_rank
                            break
                        else:
                            U_int = np.round(U_opt)
                            V_int = np.round(V_opt)
                            W_int = np.round(W_opt)
                            if verify(U_int, V_int, W_int, tol=1e-6):
                                best_U, best_V, best_W = U_int, V_int, W_int
                                best_rank = target_rank
                                break
                            else:
                                best_U, best_V, best_W = U_opt, V_opt, W_opt
                                best_rank = target_rank
                                break

    if not verify(best_U, best_V, best_W, tol=1e-6):
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    
    if not verify(U_out, V_out, W_out, tol=1e-6):
        U_out = np.round(best_U)
        V_out = np.round(best_V)
        W_out = np.round(best_W)
        
    if not verify(U_out, V_out, W_out, tol=1e-6):
        for denom in [3, 4, 6, 12]:
            U_out = np.round(best_U * denom) / denom
            V_out = np.round(best_V * denom) / denom
            W_out = np.round(best_W * denom) / denom
            if verify(U_out, V_out, W_out, tol=1e-6):
                break
        
    if not verify(U_out, V_out, W_out, tol=1e-6):
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
