# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _verify_rounded(U, V, W, n, m, p, tol=1e-6):
    # Check if U, V, W exactly decompose the tensor
    # D_{i, j, k, l} = sum_r W[k*n+i, r] U[i*m+j, r] V[l*p+k, r]
    # Target: delta_{j, l}
    
    # We can check a few random samples or all if small
    # n=3, m=4, p=4. Total 192 entries in D? No, D is 4th order.
    # But we can check C_{i, k} for random A, B?
    # No, we need exact tensor match.
    
    # Check all i, k
    for i in range(n):
        for k in range(p):
            w = W[k*n+i, :]
            U_i = U[i*m : (i+1)*m, :]
            V_k = V[np.arange(m)*p + k, :]
            
            U_w = U_i * w[None, :]
            D = U_w @ V_k.T # m x m
            
            # D should be Identity
            if not np.allclose(D, np.eye(m), atol=tol):
                return False
    return True

def _optimize_decomposition():
    n, m, p = 3, 4, 4
    
    def loss_and_grad(params, r):
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        loss = 0.0
        grad = np.zeros_like(params)
        
        for i in range(n):
            for k in range(p):
                w = W[k*n+i, :]
                U_i = U[i*m : (i+1)*m, :]
                V_k = V[np.arange(m)*p + k, :]
                
                U_w = U_i * w[None, :]
                D = U_w @ V_k.T
                
                M = np.eye(m)
                Err = D - M
                loss += np.sum(Err**2)
                
                grad_U_i = 2 * (Err @ V_k) * w[None, :]
                grad_V_k = 2 * (Err.T @ U_i) * w[None, :]
                
                grad_W = np.zeros(r)
                for r_idx in range(r):
                    grad_W[r_idx] = 2 * np.sum(Err * U_i[:, r_idx][:, None] * V_k[:, r_idx][None, :])
                
                grad[i*m : (i+1)*m, :] += grad_U_i
                grad[np.arange(m)*p + k, :] += grad_V_k
                grad[k*n+i, :] += grad_W
        
        return loss, grad

    start_time = time.time()
    best_rank = n*m*p
    best_factors = None
    
    random.seed(42)
    
    for rank in range(48, 25, -1):
        if time.time() - start_time > 100:
            break
            
        for restart in range(10):
            if time.time() - start_time > 100:
                break
                
            U0 = np.random.randn(n*m, rank)
            V0 = np.random.randn(m*p, rank)
            W0 = np.random.randn(p*n, rank)
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(
                    lambda p: loss_and_grad(p, rank)[0],
                    params0,
                    jac=lambda p: loss_and_grad(p, rank)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if res.success and res.fun < 1e-10:
                    U_opt, V_opt, W_opt = res.x[:n*m*rank].reshape(n*m, rank), \
                                          res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank), \
                                          res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if _verify_rounded(U_round, V_round, W_round, n, m, p):
                        best_rank = rank
                        best_factors = (U_round, V_round, W_round)
                        break
                        
            except Exception:
                pass

    if best_factors:
        U, V, W = best_factors
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m+j
                    v_idx = j*p+k
                    w_idx = k*n+i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

def construct():
    return _optimize_decomposition()
# EVOLVE-BLOCK-END
